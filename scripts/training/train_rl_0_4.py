"""
train_rl_lstm_ac.py

Training script for LSTM Actor-Critic baseline
for multihop QA on knowledge graphs.

This script is STRUCTURALLY IDENTICAL to train_rl_0.py
and differs ONLY in:
- model_version (QueryPathRLV04)
- internal recurrence handled inside the model

This guarantees fair comparison.
"""

import os
import json
import random
import datetime
from pathlib import Path
from typing import Any, Dict, List
import argparse

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import QUERY_PATH_RL, RELATIONAL_GAT
from src.dataloader import RelGraphDataset2

DATASET_SAMPLES = {
    "hotpotqa": -1,
    "2wikiqa": 9000,
    "musique": -1,
}

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

# --- add this utility ---
def compute_gae(rewards, values, gamma, lam):
    """
    rewards: list[float]
    values:  tensor[T]
    """
    T = len(rewards)
    values = torch.cat([values, torch.zeros(1, device=values.device)])
    advantages = torch.zeros(T, device=values.device)

    gae = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae

    returns = advantages + values[:-1]
    return advantages.detach(), returns.detach()


# --------------------------- Training loop ---------------------------
def train(
    dataset_path: str = "dataset/traintestamr.jsonl",
    save_path: str = "./checkpoints_baseline",
    tag: str = "hotpotqa",
    encoder: str = "bert",
    model_version: str = "QueryPathRLV04",   # <<< LSTM-AC
    rgat_version: str = "RelationalGATV1",
    epochs: int = 10,
    num_hops: int = 20,
    gamma: float = 0.99,
    lr: float = 1e-4,                        # lower LR for recurrent AC
    test_run: bool = False,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):

    print("[INFO] Starting baseline RL training (LSTM-AC)")
    os.makedirs(save_path, exist_ok=True)
    device = torch.device(device)
    logs_path = "logs"

    run_name = f"baseline_rl_lstmac_{datetime.datetime.now():%Y%m%d_%H%M%S}"
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
    def load_dataset(path, encoder_name, tag="hotpotqa", test_run=False, split="train"):
        test_samples = 50
        dataset = []
        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                dataset.append(json.loads(line))
        dataset = [d for d in dataset if d["split"] == split and d["tag"] == tag]
        if test_run:
            dataset = dataset[:test_samples]
        return RelGraphDataset2(
            raw_data=dataset,
            encoder=encoder_name,
            num_samples=DATASET_SAMPLES[tag],
            max_nodes=200,
        )

    dataset = load_dataset(dataset_path, encoder, tag, test_run)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # ---------------- Model ----------------
    assert model_version in QUERY_PATH_RL, f"Unknown model version: {model_version}"

    model = QUERY_PATH_RL[model_version](
        encoder=encoder,
        num_hops=num_hops,
        device=device,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Optional RGAT encoder (IDENTICAL to baseline)
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

            # ---- EXACT tensor sanitation as baseline ----
            if isinstance(adj, (list, tuple)): adj = adj[0]
            if adj.dim() == 3 and adj.size(0) == 1: adj = adj.squeeze(0)
            if isinstance(node_feat, (list, tuple)): node_feat = node_feat[0]
            if node_feat.dim() == 3 and node_feat.size(0) == 1: node_feat = node_feat.squeeze(0)

            # ---- RGAT embeddings (if provided) ----
            rgat_nodes = node_feat
            if gat_encoder is not None:
                with torch.no_grad():
                    out = gat_encoder(
                        adj.unsqueeze(0),
                        rel_feat,
                        node_feat.unsqueeze(0),
                    )
                    rgat_nodes = (
                        out[0].squeeze(0)
                        if isinstance(out, (tuple, list))
                        else out.squeeze(0)
                    )

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

                rewards = ep["rewards"]
                logps = torch.stack(ep["logps"])
                values = torch.stack(ep["values"])

                advantages, value_targets = compute_gae(
                    rewards=rewards,
                    values=values,
                    gamma=gamma,
                    lam=0.95,
                )

                policy_loss = -(logps * advantages).mean()
                value_loss = F.mse_loss(values, value_targets)
                loss = policy_loss + 0.5 * value_loss


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
                    "mean_reward": epoch_reward / max(1, epoch_episodes),
                    "success_rate": epoch_success / max(1, epoch_episodes),
                })

        # ---------------- Epoch end ----------------
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

        # ---------------- Save checkpoint ----------------
        torch.save(
            model.state_dict(),
            model_save_path + f"/baseline_rl_{model_version}_epoch{epoch+1}.pt",
        )
        if gat_encoder is not None:
            torch.save(
                gat_encoder.state_dict(),
                gat_save_path + f"/baseline_rl_rgat_epoch{epoch+1}.pt",
            )

    print("[INFO] LSTM Actor-Critic training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="hotpotqa")
    parser.add_argument("--test_run", action="store_true")
    args = parser.parse_args()

    train(
        epochs=10,
        num_hops=20,
        gamma=0.99,
        encoder="bert",
        save_path="./checkpoints_baseline_lstmac",
        model_version="QueryPathRLV04",
        rgat_version="RelationalGATV1",
        lr=1e-4,
        tag=args.dataset,
        test_run=args.test_run,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
