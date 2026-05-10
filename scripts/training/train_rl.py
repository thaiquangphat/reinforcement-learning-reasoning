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
from src.dataloader import RelGraphDataset

# ----------- Dataset Loader -----------
def load_dataset(path, encoder_name, tag="hotpotqa", test=False):
    test_samples = 50
    dataset = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            dataset.append(json.loads(line))
    dataset = [d for d in dataset if d['tag']==tag]
    if test:
        dataset = dataset[:test_samples]

    return RelGraphDataset(
        raw_data=dataset,
        encoder=encoder_name,
        num_samples=-1,
        max_nodes=50
    )

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


def compute_gae(rewards, values, gamma, lam, last_value=0.0):
    if not values:
        values = [0.0] * len(rewards)

    rewards = [float(r) for r in rewards]
    values = [float(v) for v in values] + [float(last_value)]

    T = len(rewards)
    advantages = [0.0] * T
    gae = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae

    returns = [advantages[t] + values[t] for t in range(T)]
    return returns, advantages



def train(
    save_path: str = "./checkpoints",
    rgat_version: str = "RelationalGATV1",
    query_path_version: str = "QueryPathRLV1",
    querypath_cfg: Dict[str, Any] = None,
    epochs: int = 10,
    episodes_per_update: int = 16,
    gamma: float = 0.99,
    lam: float = 0.95,
    lr: float = 3e-4,
    vf_coeff: float = 0.5,
    entropy_coeff: float = 0.01,
    test_run: bool = False,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    os.makedirs(save_path, exist_ok=True)
    device = torch.device(device)

    run_name = f"run_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    run_dir = "logs/" + run_name
    os.makedirs(run_dir, exist_ok=True)
    local_log, log_file = setup_local_logger(run_name, log_dir=run_dir)
    print(f"[INFO] Logging to {log_file}")

    checkpoint_path = save_path + "/" + run_name
    model_save_path = checkpoint_path + "/model"
    gat_save_path = checkpoint_path + "/rgat"

    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(gat_save_path, exist_ok=True)

    local_log({
        "event": "save_paths",
        "checkpoints": checkpoint_path,
        "model_checkpoints": model_save_path,
        "rgat_checkpoints": gat_save_path
    })

    model = QUERY_PATH_RL[query_path_version](device=device, **(querypath_cfg or {}))
    model.to(device)
    print("[INFO] Initialized Query path model:", query_path_version)

    model_reward_config = model.reward_config
    local_log({
        "event": "querypath_model_reward_config",
        **model_reward_config
    })

    gat_encoder = None
    gat_cls = RELATIONAL_GAT.get(rgat_version, None)
    if gat_cls is not None:
        gat_encoder = gat_cls(in_dim=model.in_dim).to(device)
        print("[INFO] Initialized GAT encoder:", rgat_version)

    local_log({
        "event": "base_model",
        "model": query_path_version,
        "rgat_model": rgat_version
    })
    
    optimizer = optim.Adam(list(model.parameters()) + ([] if gat_encoder is None else list(gat_encoder.parameters())), lr=lr)

    local_log({
        "event": "hyperparameters",
        "epochs": epochs,
        "episodes_per_update": episodes_per_update,
        "gamma": gamma,
        "lam": lam,
        "lr": lr,
        "vf_coeff": vf_coeff,
        "entropy_coeff": entropy_coeff,
        "test_run": test_run,
    })

    dataset = None
    dataloader = None
    tag = "hotpotqa"
    
    dataset = load_dataset(path='dataset/train.jsonl', encoder_name='bert', tag=tag, test=test_run)
    local_log({
        "event": "dataset",
        "path": "dataset/train.jsonl",
        "encoder_name": "bert",
        "tag": tag,
    })
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
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", ascii=" .-=#")

        for idx, batch in enumerate(pbar):
            adj = batch["adj"].to(device)
            rel_feat = batch["rel_feat"].to(device)
            node_feat = batch["node_feat"].to(device)
            nodes = batch["nodes"]
            query = batch["query"]
            question = batch["question"][0]

            if isinstance(adj, (list, tuple)): adj = adj[0]
            if adj.dim() == 3 and adj.size(0) == 1: adj = adj.squeeze(0)
            if isinstance(node_feat, (list, tuple)): node_feat = node_feat[0]
            if node_feat.dim() == 3 and node_feat.size(0) == 1: node_feat = node_feat.squeeze(0)

            rgat_nodes = node_feat
            with torch.no_grad():
                out = gat_encoder(adj.unsqueeze(0), rel_feat, node_feat.unsqueeze(0))
                rgat_nodes = out[0].squeeze(0) if isinstance(out, (tuple, list)) else out.squeeze(0)

            for qury in query:
                start_node = qury[0][0]
                target_node = qury[-1][-1]

                start_idx = nodes.index(start_node)
                target_idx = nodes.index(target_node)

                ep = model.run_episode(
                    start_idx=start_idx,
                    question=question,
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
                        if len(values) < len(rewards):
                            missing = len(rewards) - len(values)
                            values = values + [0.0] * missing
                        with torch.no_grad():
                            if len(values) > 0 and len(ep_item["q_list"]) > 0 and len(ep_item["e_list"]) > 0:
                                last_value = float(
                                    model.get_value(
                                        ep_item["q_list"][-1].to(device),
                                        ep_item["e_list"][-1].to(device)
                                    ).item()
                                )
                            else:
                                last_value = 0.0
                        returns, advantages = compute_gae(rewards, values, gamma, lam, last_value)
                        all_returns += returns; all_advantages += advantages
                        all_logps += [lp for lp in ep_item["logps"]]
                        all_entropies += ep_item["entropies"]
                        all_values += values

                    min_len = min(len(all_logps), len(all_advantages), len(all_returns), len(all_values))
                    if min_len == 0:
                        continue
                    all_logps = all_logps[:min_len]
                    all_advantages = all_advantages[:min_len]
                    all_returns = all_returns[:min_len]
                    all_values = all_values[:min_len]

                    logps_t = torch.stack(all_logps).to(device)
                    advantages_t = torch.tensor(all_advantages, dtype=torch.float32, device=device)
                    returns_t = torch.tensor(all_returns, dtype=torch.float32, device=device)
                    values_t = torch.tensor(all_values, dtype=torch.float32, device=device)

                    if advantages_t.numel() > 1:
                        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
                    else:
                        advantages_t = advantages_t * 0.0

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

        torch.save(model.state_dict(), os.path.join(model_save_path, f"querypathrl_epoch{epoch+1}.pt"))
        if gat_encoder is not None:
            torch.save(gat_encoder.state_dict(), os.path.join(gat_save_path, f"gatencoder_epoch{epoch+1}.pt"))
        print(f"[INFO] Epoch {epoch+1} done. Logged to {log_file}")


if __name__ == "__main__":
    train(
        save_path="./checkpoints",
        rgat_version="RelationalGATV1",
        query_path_version="QueryPathRLV1",
        querypath_cfg={"encoder": "bert", "num_hops": 100},
        epochs=10,
        episodes_per_update=6,
        gamma=0.99,
        lam=0.97,
        lr=1e-6,
        vf_coeff=0.4,
        entropy_coeff=0.01,
        test_run=False
    )