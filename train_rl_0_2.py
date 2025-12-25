import os
import json
import random
import datetime
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import QUERY_PATH_RL, RELATIONAL_GAT
from dataloader import RelGraphDataset2


# --------------------------- Local JSONL logger ---------------------------
def setup_local_logger(name, log_dir="logs"):
    logs_dir = Path(log_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{name}.jsonl"

    def _safe(val):
        if isinstance(val, torch.Tensor):
            return val.item() if val.numel() == 1 else val.detach().cpu().tolist()
        try:
            json.dumps(val)
            return val
        except Exception:
            return str(val)

    def _log(entry: dict):
        entry = {k: _safe(v) for k, v in entry.items()}
        entry = {"timestamp": datetime.datetime.now().isoformat(), **entry}
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    _log({"event": "logger_initialized", "log_path": str(log_path)})
    return _log, str(log_path)


# --------------------------- Discounted returns ---------------------------
def compute_returns(rewards, gamma):
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns


# --------------------------- Training loop ---------------------------
def train(
    dataset_path: str = "dataset/traintestamr.jsonl",
    save_path: str = "./checkpoints_baseline_ppo",
    tag: str = "hotpotqa",
    encoder: str = "bert",
    model_version: str = "QueryPathRLV0",
    rgat_version: str = "RelationalGATV1",
    epochs: int = 10,
    num_hops: int = 20,
    gamma: float = 0.99,
    lr: float = 3e-4,
    ppo_clip = 0.1,
    ppo_epochs = 1,          # can start with 1 if unstable
    value_coef = 0.5,
    entropy_coef = 0.02,
    test_run: bool = False,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):

    print("[INFO] Starting baseline RL training")
    os.makedirs(save_path, exist_ok=True)
    device = torch.device(device)
    logs_path = "logs"

    run_name = f"baseline_rl_ppo_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    run_dir = Path(logs_path) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    local_log, log_file = setup_local_logger(run_name, log_dir=run_dir)
    print(f"[INFO] Logging to {log_file}")

    local_log({
        "event": "input_parameters",
        **locals().copy()
    })

    checkpoint_path = save_path + "/" + run_name
    model_save_path = checkpoint_path + "/model"
    gat_save_path = checkpoint_path + "/rgat"
    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(gat_save_path, exist_ok=True)


    # ---------------- Dataset ----------------
    def load_dataset(path, encoder_name, tag="hotpotqa", test_run=False, split='train'):
            test_samples = 50
            dataset = []
            with open(path, 'r', encoding='utf-8') as file:
                for line in file:
                    dataset.append(json.loads(line))
            dataset = [d for d in dataset if d['split'] == split and d['tag']==tag]
            if test_run:
                dataset = dataset[:test_samples]
            return RelGraphDataset2(raw_data=dataset, encoder=encoder_name, num_samples=6000, max_nodes=200)

    dataset = load_dataset(dataset_path, encoder, tag, test_run)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # ---------------- Model ----------------
    model = QUERY_PATH_RL[model_version](
        encoder=encoder,
        num_hops=num_hops,
        device=device,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Optional RGAT encoder (same as your hierarchical setup)
    gat_encoder = None
    gat_cls = RELATIONAL_GAT.get(rgat_version, None)
    if gat_cls is not None:
        gat_encoder = gat_cls(in_dim=model.in_dim).to(device)
        print("[INFO] Initialized RGAT encoder:", rgat_version)

    # ---------------- Training ----------------
    global_step = 0
    for epoch in range(epochs):
        epoch_reward = 0.0
        epoch_success = 0
        epoch_episodes = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", ascii=" .-=#")

        for batch in pbar:
            adj = batch["adj"].to(device)
            rel_feat = batch["rel_feat"].to(device)
            node_feat = batch["node_feat"].to(device)
            nodes = batch["nodes"]
            question = batch["question"][0]

            if isinstance(adj, (list, tuple)): adj = adj[0]
            if adj.dim() == 3 and adj.size(0) == 1: adj = adj.squeeze(0)
            if isinstance(node_feat, (list, tuple)): node_feat = node_feat[0]
            if node_feat.dim() == 3 and node_feat.size(0) == 1: node_feat = node_feat.squeeze(0)

            # compute rgat nodes (if encoder provided) or take node_feat as precomputed embeddings
            rgat_nodes = node_feat
            if gat_encoder is not None:
                with torch.no_grad():
                    out = gat_encoder(adj.unsqueeze(0), rel_feat, node_feat.unsqueeze(0))
                    rgat_nodes = out[0].squeeze(0) if isinstance(out, (tuple, list)) else out.squeeze(0)

            for qury in batch["query"]:
                start_node = qury[0][0]
                target_node = qury[-1][-1]

                start_idx = nodes.index(start_node)
                target_idx = nodes.index(target_node)

                ep = model.run_episode(
                    start_idx=start_idx,
                    question=question,
                    adj=adj,
                    rgat_nodes=rgat_nodes,
                    target_idx=target_idx,
                    deterministic=False,
                )

                # ---------- PPO UPDATE ----------
                rewards = ep["rewards"]
                logps_old = torch.stack(ep["logps_old"]).detach()
                values_old = torch.stack(ep["values"]).detach()

                returns = torch.tensor(
                    compute_returns(rewards, gamma),
                    dtype=torch.float32,
                    device=device,
                )

                advantages = returns - values_old
                if advantages.numel() > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                else:
                    advantages = advantages.detach()

                for _ in range(ppo_epochs):
                    new_values = []
                    new_logps = []
                    entropies = []

                    # Recompute log πθ(at | st)
                    q = model.get_embedding(question)
                    cur_idx = start_idx
                    h_t = rgat_nodes[cur_idx].to(device)
                    v = model.value(torch.cat([q, h_t], dim=0))
                    new_values.append(v.squeeze(0))

                    for next_idx in ep["idx_list"]:
                        neighbors = torch.nonzero(adj[cur_idx], as_tuple=False).squeeze(-1)
                        cand_indices = neighbors.tolist()
                        cand_h = rgat_nodes[cand_indices].to(device)

                        logits = model.score_actions(q, h_t, cand_h)
                        probs = F.softmax(logits, dim=0)
                        probs = torch.clamp(probs, min=1e-8)
                        probs = probs / probs.sum()
                        dist = torch.distributions.Categorical(probs)

                        action = cand_indices.index(next_idx)
                        action = torch.tensor(action, device=device)

                        new_logps.append(dist.log_prob(action))
                        entropies.append(dist.entropy())

                        cur_idx = next_idx
                        h_t = rgat_nodes[cur_idx].to(device)

                    new_logps = torch.stack(new_logps)
                    entropies = torch.stack(entropies)

                    ratio = torch.exp(new_logps - logps_old)

                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1 - ppo_clip, 1 + ppo_clip) * advantages

                    policy_loss = -torch.min(surr1, surr2).mean()
                    new_values = torch.stack(new_values)
                    value_loss = F.mse_loss(new_values, returns)
                    entropy_loss = -entropies.mean()

                    loss = (
                        policy_loss
                        + value_coef * value_loss
                        + entropy_coef * entropy_loss
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()


                episode_reward = sum(rewards)
                epoch_reward += episode_reward
                epoch_success += int(ep["success"])
                epoch_episodes += 1
                global_step += 1

                _mean_reward = epoch_reward / max(1, epoch_episodes)
                _success_rate = epoch_success / max(1, epoch_episodes)

                local_log({
                    "event": "train_step",
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "mean_reward": _mean_reward,
                    "success_rate": _success_rate,
                })

        mean_reward = epoch_reward / max(1, epoch_episodes)
        success_rate = epoch_success / max(1, epoch_episodes)

        local_log({
            "event": "epoch_end",
            "epoch": epoch + 1,
            "global_step": global_step,
            "mean_reward": mean_reward,
            "success_rate": success_rate,
            "episodes": epoch_episodes,
        })

        print(
            f"[Epoch {epoch+1}] "
            f"mean_reward={mean_reward:.4f} "
            f"success_rate={success_rate:.4f}"
        )

        # Save checkpoint
        torch.save(
            model.state_dict(),
            model_save_path + "/" + f"baseline_rl_querypathrl_epoch{epoch+1}.pt",
        )
        if gat_encoder is not None:
            torch.save(
                gat_encoder.state_dict(),
                gat_save_path + "/" + f"baseline_rl_rgat_epoch{epoch+1}.pt",
            )

    print("[INFO] Baseline RL training finished.")


if __name__ == "__main__":
    train(
        epochs=10,
        num_hops=20,
        gamma=0.99,
        encoder="bert",
        model_version="QueryPathRLV02",
        rgat_version="RelationalGATV1",
        lr=3e-4,
        tag="hotpotqa",
        test_run=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
