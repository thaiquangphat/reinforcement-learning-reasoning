"""
train_baseline_ppo_gae.py

Correct PPO + GAE baseline for KG multihop reasoning.
STRUCTURALLY MATCHES baseline_rl.py
"""

import os
import json
import datetime
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

from src import QUERY_PATH_RL, RELATIONAL_GAT
from src.dataloader import RelGraphDataset2


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


# --------------------------- GAE ---------------------------
def compute_gae(rewards, values, gamma, lam):
    """
    rewards: List[float]
    values: List[float]
    """
    advantages = []
    gae = 0.0

    values = values + [0.0]  # V_{T+1} = 0

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)

    returns = [adv + v for adv, v in zip(advantages, values[:-1])]
    return advantages, returns


# --------------------------- Training loop ---------------------------
def train(
    dataset_path: str = "dataset/traintestamr.jsonl",
    save_path: str = "./checkpoints_baseline",
    tag: str = "hotpotqa",
    encoder: str = "bert",
    model_version: str = "QueryPathRLV05",
    rgat_version: str = "RelationalGATV1",
    epochs: int = 10,
    num_hops: int = 20,
    gamma: float = 0.99,
    lam: float = 0.95,
    lr: float = 3e-4,
    ppo_clip: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    test_run: bool = False,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):

    print("[INFO] Starting PPO + GAE baseline training")
    os.makedirs(save_path, exist_ok=True)
    device = torch.device(device)

    run_name = f"baseline_ppo_gae_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    run_dir = Path("logs") / run_name
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
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(gat_save_path, exist_ok=True)

    # ---------------- Dataset ----------------
    def load_dataset(path, encoder_name, tag="hotpotqa", test_run=False, split="train"):
        test_samples = 50
        dataset = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                dataset.append(json.loads(line))
        dataset = [d for d in dataset if d["split"] == split and d["tag"] == tag]
        if test_run:
            dataset = dataset[:test_samples]
        return RelGraphDataset2(
            raw_data=dataset,
            encoder=encoder_name,
            num_samples=6000,
            max_nodes=200,
        )

    dataset = load_dataset(dataset_path, encoder, tag, test_run)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # ---------------- Model ----------------
    model = QUERY_PATH_RL[model_version](
        encoder=encoder,
        num_hops=num_hops,
        device=device,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ---------------- Optional RGAT ----------------
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

            if isinstance(adj, (list, tuple)):
                adj = adj[0]
            if adj.dim() == 3:
                adj = adj.squeeze(0)

            if isinstance(node_feat, (list, tuple)):
                node_feat = node_feat[0]
            if node_feat.dim() == 3:
                node_feat = node_feat.squeeze(0)

            # RGAT encoding
            rgat_nodes = node_feat
            if gat_encoder is not None:
                with torch.no_grad():
                    out = gat_encoder(
                        adj.unsqueeze(0),
                        rel_feat,
                        node_feat.unsqueeze(0),
                    )
                    rgat_nodes = out[0].squeeze(0) if isinstance(out, (tuple, list)) else out.squeeze(0)

            for query in batch["query"]:
                start_node = query[0][0]
                target_node = query[-1][-1]

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

                rewards = ep["rewards"]
                logps = torch.stack(ep["logps"])
                values = torch.stack(ep["values"]).squeeze(-1)
                values_list = values.detach().cpu().flatten().tolist()
                entropies = torch.stack(ep["entropies"])

                advantages, returns = compute_gae(
                    rewards,
                    values_list,
                    gamma,
                    lam,
                )

                advantages = torch.tensor(advantages, device=device)
                returns = torch.tensor(returns, device=device)

                # PPO
                old_logps = logps.detach()
                ratio = torch.exp(logps - old_logps)

                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - ppo_clip, 1 + ppo_clip) * advantages

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, returns)
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

                local_log({
                    "event": "train_step",
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "mean_reward": epoch_reward / epoch_episodes,
                    "success_rate": epoch_success / epoch_episodes,
                })

        local_log({
            "event": "epoch_end",
            "epoch": epoch + 1,
            "global_step": global_step,
            "mean_reward": epoch_reward / epoch_episodes,
            "success_rate": epoch_success / epoch_episodes,
            "episodes": epoch_episodes,
        })

        print(
            f"[Epoch {epoch+1}] "
            f"mean_reward={epoch_reward / epoch_episodes:.4f} "
            f"success_rate={epoch_success / epoch_episodes:.4f}"
        )

        torch.save(
            model.state_dict(),
            model_save_path + f"/baseline_ppo_gae_epoch{epoch+1}.pt",
        )
        if gat_encoder is not None:
            torch.save(
                gat_encoder.state_dict(),
                gat_save_path + f"/baseline_ppo_gae_rgat_epoch{epoch+1}.pt",
            )

    print("[INFO] PPO + GAE baseline training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="hotpotqa")
    parser.add_argument("--test_run", action="store_true")
    args = parser.parse_args()

    train(
        save_path="./checkpoints_baseline_ppo_gae",
        epochs=10,
        num_hops=20,
        gamma=0.99,
        lam=0.95,
        encoder="bert",
        model_version="QueryPathRLV05",
        rgat_version="RelationalGATV1",
        lr=3e-4,
        tag=args.dataset,
        test_run=args.test_run,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
