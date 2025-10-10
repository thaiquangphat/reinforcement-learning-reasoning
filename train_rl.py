"""
train_rl.py (with JSONL logger)

Training script for QueryPathRL using Actor-Critic (A2C) with GAE and local JSONL logging.
"""
import os
import math
import json
import datetime
from pathlib import Path
from typing import List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import RELATIONAL_GAT, QUERY_PATH_RL
from dataloader import RelGraphDataset


# ----------- Local JSONL Logger (logs/) -----------
def setup_local_logger(name, log_dir="logs"):
    logs_dir = Path(log_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{name}.jsonl"

    def _safe_serialize(val):
        try:
            if isinstance(val, torch.Tensor):
                return val.item() if val.numel() == 1 else val.detach().cpu().tolist()
            elif isinstance(val, (datetime.datetime,)):
                return val.isoformat()
            else:
                json.dumps(val)
                return val
        except Exception:
            return str(val)

    def _log(entry: dict):
        safe_entry = {}
        for k, v in entry.items():
            safe_entry[k] = _safe_serialize(v)
        entry_with_ts = {"timestamp": datetime.datetime.now().isoformat(), **safe_entry}
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry_with_ts, ensure_ascii=False) + "\n")

    _log({"event": "logger_initialized", "log_path": str(log_path)})
    return _log, str(log_path)


def compute_gae(rewards: List[float], values: List[float], gamma: float, lam: float, last_value: float = 0.0):
    T = len(rewards)
    values = values + [last_value]
    advantages = [0.0] * T
    gae = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = [advantages[t] + values[t] for t in range(T)]
    return returns, advantages


def train(
    save_path: str = "./checkpoints_rl",
    rgat_version: str = "RelationalGATV1",
    query_path_version: str = "QueryPathRLV1",
    querypath_cfg: Dict[str, Any] = None,
    epochs: int = 10,
    batch_size: int = 8,
    episodes_per_update: int = 16,
    gamma: float = 0.99,
    lam: float = 0.95,
    lr: float = 3e-4,
    vf_coeff: float = 0.5,
    entropy_coeff: float = 0.01,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    os.makedirs(save_path, exist_ok=True)
    device = torch.device(device)

    run_name = f"run_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    local_log, log_file = setup_local_logger(run_name, log_dir="logs/train")
    print(f"[INFO] Logging to {log_file}")

    model = QUERY_PATH_RL[query_path_version](device=device, **(querypath_cfg or {}))
    model.to(device)

    gat_encoder = None
    if RELATIONAL_GAT is not None:
        gat_cls = RELATIONAL_GAT.get(rgat_version, None)
        if gat_cls is not None:
            gat_encoder = gat_cls(in_dim=model.in_dim).to(device)
            print("[INFO] Initialized GAT encoder:", rgat_version)
    optimizer = optim.Adam(list(model.parameters()) + ([] if gat_encoder is None else list(gat_encoder.parameters())), lr=lr)

    dataset = None
    dataloader = None
    if RelGraphDataset is not None:
        dataset = RelGraphDataset()
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    global_step = 0
    for epoch in range(epochs):
        if dataloader is None:
            print("[INFO] No dataloader configured; exiting training loop.")
            break

        epoch_reward = 0.0
        epoch_success = 0
        epoch_steps = 0
        batch_episodes = []
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for idx, batch in enumerate(pbar):
            try:
                adj, node_feat, rel_feat, nodes, query = batch
            except Exception:
                adj, node_feat, rel_feat, nodes, query = batch[0]

            if isinstance(adj, (list, tuple)): adj = adj[0]
            if adj.dim() == 3 and adj.size(0) == 1: adj = adj.squeeze(0)
            if isinstance(node_feat, (list, tuple)): node_feat = node_feat[0]
            if node_feat.dim() == 3 and node_feat.size(0) == 1: node_feat = node_feat.squeeze(0)

            rgat_nodes = node_feat
            if gat_encoder is not None:
                with torch.no_grad():
                    out = gat_encoder(adj.unsqueeze(0), rel_feat, node_feat.unsqueeze(0))
                    rgat_nodes = out[0].squeeze(0) if isinstance(out, (tuple, list)) else out.squeeze(0)

            for qury in query:
                try:
                    start_node = qury[0][0]; target_node = qury[-1][-1]
                    start_idx = nodes.index(start_node); target_idx = nodes.index(target_node)
                except Exception:
                    start_idx = int(qury[0][0]); target_idx = int(qury[-1][-1])
                question_text = " ".join([str(x) for x in qury])

                ep = model.run_episode(
                    start_idx=start_idx,
                    question=question_text,
                    adj=adj.to(device),
                    rgat_nodes=rgat_nodes.to(device),
                    num_hops=model.num_hops,
                    target_idx=target_idx,
                    deterministic=False,
                    mask_visited=True,
                )
                batch_episodes.append(ep)
                epoch_reward += sum(ep["rewards"])
                epoch_success += int(ep["success"])
                epoch_steps += len(ep["rewards"])

                if len(batch_episodes) >= episodes_per_update:
                    all_logps, all_values, all_returns, all_advantages, all_entropies = [], [], [], [], []
                    for ep_item in batch_episodes:
                        rewards = ep_item["rewards"]
                        values = [v.item() if isinstance(v, torch.Tensor) else float(v) for v in ep_item["values"]]
                        last_value = 0.0
                        if len(values) > 0:
                            with torch.no_grad():
                                last_value = float(model.get_value(ep_item["q_list"][-1].to(device), ep_item["e_list"][-1].to(device)).item())
                        returns, advantages = compute_gae(rewards, values, gamma, lam, last_value)
                        all_returns += returns; all_advantages += advantages
                        all_logps += [lp for lp in ep_item["logps"]]
                        all_entropies += ep_item["entropies"]
                        all_values += values

                    advantages_t = torch.tensor(all_advantages, dtype=torch.float32, device=device)
                    returns_t = torch.tensor(all_returns, dtype=torch.float32, device=device)
                    logps_t = torch.stack(all_logps).to(device) if len(all_logps) > 0 else torch.tensor([], device=device)
                    values_t = torch.tensor(all_values, dtype=torch.float32, device=device)
                    advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

                    actor_loss = -(logps_t * advantages_t).mean() if logps_t.numel() > 0 else torch.tensor(0.0, device=device)
                    value_loss = F.mse_loss(values_t, returns_t) if values_t.numel() > 0 else torch.tensor(0.0, device=device)
                    entropy_bonus = torch.tensor(sum(all_entropies) / max(1, len(all_entropies)), device=device)
                    total_loss = actor_loss + vf_coeff * value_loss - entropy_coeff * entropy_bonus + model.get_regularization_loss()

                    optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(list(model.parameters()) + ([] if gat_encoder is None else list(gat_encoder.parameters())), 1.0)
                    optimizer.step()

                    mean_reward = epoch_reward / max(1, global_step + 1)
                    success_rate = epoch_success / max(1, len(batch_episodes))
                    avg_len = epoch_steps / max(1, len(batch_episodes))

                    local_log({
                        "event": "train_step",
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "actor_loss": actor_loss.detach().cpu().item(),
                        "value_loss": value_loss.detach().cpu().item(),
                        "entropy": entropy_bonus.detach().cpu().item(),
                        "total_loss": total_loss.detach().cpu().item(),
                        "mean_reward": mean_reward,
                        "success_rate": success_rate,
                        "avg_episode_len": avg_len,
                        "episodes_in_batch": len(batch_episodes),
                    })

                    global_step += 1
                    batch_episodes = []

        local_log({
            "event": "epoch_end",
            "epoch": epoch + 1,
            "mean_reward": epoch_reward / max(1, len(dataloader)),
            "success_rate": epoch_success / max(1, len(dataloader)),
            "total_steps": epoch_steps,
        })

        torch.save(model.state_dict(), os.path.join(save_path, f"querypathrl_epoch{epoch+1}.pt"))
        if gat_encoder is not None:
            torch.save(gat_encoder.state_dict(), os.path.join(save_path, f"gat_encoder_epoch{epoch+1}.pt"))
        print(f"[INFO] Epoch {epoch+1} done. Logged to {log_file}")


if __name__ == "__main__":
    train(
        save_path="./checkpoints_rl",
        rgat_version="RelationalGATV3",
        query_path_version="QueryPathRLV1",
        querypath_cfg={"encoder": "sbert", "num_hops": 20},
        epochs=5,
        batch_size=1,
        episodes_per_update=8,
        gamma=0.99,
        lam=0.95,
        lr=3e-4,
        vf_coeff=0.5,
        entropy_coeff=0.01,
    )