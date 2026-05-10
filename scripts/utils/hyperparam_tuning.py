# hyperparam_search.py
"""
Random-search hyperparameter tuning script based on the provided train_rl.py.
For each sampled hyperparameter configuration we run 3 epochs on test subset
(50 samples as in load_dataset(..., test=True)) and print results to terminal.

Inspired / adapted from user's train_rl.py. See original for details.
"""
import os
import math
import json
import datetime
import random
from pathlib import Path
from typing import List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Importing model classes and dataset loader from your project.
# Ensure pythonpath includes the project root or run from project root.
from src import RELATIONAL_GAT, QUERY_PATH_RL
from src.dataloader import RelGraphDataset

# ----------------- Utilities copied/adapted from original -----------------
def load_dataset(path, encoder_name, tag="hotpotqa", test=False):
    test_samples = 50
    dataset = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            dataset.append(json.loads(line))
    dataset = [d for d in dataset if d.get('tag')==tag]
    if test:
        dataset = dataset[:test_samples]

    return RelGraphDataset(
        raw_data=dataset,
        encoder=encoder_name,
        num_samples=-1,
        max_nodes=50
    )

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

# ----------------- Hyperparameter tuning core -----------------
def single_run_train(
    cfg: Dict[str, Any],
    base_save_path: str,
    run_name_prefix: str = "hpt",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Run a short training loop for one hyperparameter configuration.
    cfg contains keys: rgat_version, query_path_version, querypath_cfg, epochs,
    episodes_per_update, gamma, lam, lr, vf_coeff, entropy_coeff, test_run
    Returns a dict with summary metrics.
    """
    # Unpack
    rgat_version = cfg.get("rgat_version")
    query_path_version = cfg.get("query_path_version")
    querypath_cfg = cfg.get("querypath_cfg") or {}
    epochs = int(cfg.get("epochs", 3))
    episodes_per_update = int(cfg.get("episodes_per_update", 16))
    gamma = float(cfg.get("gamma", 0.99))
    lam = float(cfg.get("lam", 0.95))
    lr = float(cfg.get("lr", 3e-4))
    vf_coeff = float(cfg.get("vf_coeff", 0.5))
    entropy_coeff = float(cfg.get("entropy_coeff", 0.01))
    test_run = bool(cfg.get("test_run", True))

    # Setup logging and dirs
    run_name = f"{run_name_prefix}_{datetime.datetime.now():%Y%m%d_%H%M%S}_{random.randint(0,9999)}"
    run_dir = os.path.join("logs", f"hpt_{datetime.datetime.now():%Y%m%d_%H%M%S}", run_name)
    os.makedirs(run_dir, exist_ok=True)
    local_log, log_file = setup_local_logger(run_name, log_dir=run_dir)

    device = torch.device(device)

    # Build model + (optional) gat encoder
    model = QUERY_PATH_RL[query_path_version](device=device, **(querypath_cfg or {}))
    model.to(device)
    gat_encoder = None
    gat_cls = RELATIONAL_GAT.get(rgat_version, None)
    if gat_cls is not None:
        gat_encoder = gat_cls(in_dim=model.in_dim).to(device)

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

    # Load dataset with test=True to get 50 samples as requested
    tag = "hotpotqa"
    dataset = load_dataset(path='dataset/train.jsonl', encoder_name='bert', tag=tag, test=test_run)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    global_step = 0
    last_epoch_summary = {"mean_reward": -9999.0, "success_rate": 0.0, "total_steps": 0}
    for epoch in range(epochs):
        if dataloader is None:
            break

        epoch_reward = 0.0
        epoch_success = 0
        epoch_steps = 0
        batch_episodes = []
        pbar = tqdm(dataloader, desc=f"HP-Run {run_name} Epoch {epoch+1}/{epochs}", ascii=" .-=#")

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
                if gat_encoder is not None:
                    out = gat_encoder(adj.unsqueeze(0), rel_feat, node_feat.unsqueeze(0))
                    rgat_nodes = out[0].squeeze(0) if isinstance(out, (tuple, list)) else out.squeeze(0)

            for qury in query:
                start_node = qury[0][0]
                target_node = qury[-1][-1]

                # defensive: if nodes not found, skip episode
                try:
                    start_idx = nodes.index(start_node)
                    target_idx = nodes.index(target_node)
                except ValueError:
                    continue

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
                        batch_episodes = []
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

                    # Log each training step similarly to original file (so downstream parsing stays consistent)
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

        # end epoch
        mean_reward_epoch = epoch_reward / max(1, len(dataloader))
        success_rate_epoch = epoch_success / max(1, len(dataloader))
        local_log({
            "event": "epoch_end",
            "epoch": epoch + 1,
            "mean_reward": mean_reward_epoch,
            "success_rate": success_rate_epoch,
            "total_steps": epoch_steps,
        })

        last_epoch_summary = {
            "mean_reward": mean_reward_epoch,
            "success_rate": success_rate_epoch,
            "total_steps": epoch_steps,
        }

    # return chosen cfg + summary
    result = {"cfg": cfg, "summary": last_epoch_summary, "log_file": log_file}
    return result

def hyperparameter_tuning(
    save_path: str = "./checkpoints",
    rgat_version: List[str] = None,
    query_path_version: List[str] = None,
    querypath_cfg: List[Dict[str,Any]] = None,
    epochs: List[int] = None,
    episodes_per_update: List[int] = None,
    gamma: List[float] = None,
    lam: List[float] = None,
    lr: List[float] = None,
    vf_coeff: List[float] = None,
    entropy_coeff: List[float] = None,
    test_run: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_trials: int = 20,
    seed: int = 42,
):
    """
    Main hyperparameter tuning function.
    Each hyperparam argument should be an iterable (list/tuple). We'll sample
    randomly from each list num_trials times (with replacement).
    """
    random.seed(seed)
    torch.manual_seed(seed)

    # Default small search space if some lists are None
    rgat_version = rgat_version or ["RelationalGATV1"]
    query_path_version = query_path_version or ["QueryPathRLV1"]
    querypath_cfg = querypath_cfg or [ {"encoder": "bert", "num_hops": 100} ]
    epochs = epochs or [3]  # each trial runs 3 epochs by default
    episodes_per_update = episodes_per_update or [6, 8, 16]
    gamma = gamma or [0.99]
    lam = lam or [0.95, 0.97]
    lr = lr or [1e-4, 3e-4, 1e-5]
    vf_coeff = vf_coeff or [0.4, 0.5]
    entropy_coeff = entropy_coeff or [0.01, 0.0]

    all_results = []
    total_possible = num_trials
    print(f"[HPT] Starting random search with {total_possible} trials (seed={seed})")
    for t in range(total_possible):
        sampled_cfg = {
            "rgat_version": random.choice(rgat_version),
            "query_path_version": random.choice(query_path_version),
            "querypath_cfg": random.choice(querypath_cfg),
            "epochs": random.choice(epochs),
            "episodes_per_update": random.choice(episodes_per_update),
            "gamma": random.choice(gamma),
            "lam": random.choice(lam),
            "lr": random.choice(lr),
            "vf_coeff": random.choice(vf_coeff),
            "entropy_coeff": random.choice(entropy_coeff),
            "test_run": test_run,
        }
        print(f"\n[HPT] Trial {t+1}/{total_possible} - sampled cfg:")
        for k,v in sampled_cfg.items():
            print(f"   {k}: {v}")

        try:
            result = single_run_train(sampled_cfg, base_save_path=save_path, run_name_prefix=f"hpt_trial{t+1}", device=device)
            print(f"[HPT] Trial {t+1} finished. Summary: {result['summary']}. Logs: {result['log_file']}")
            all_results.append(result)
        except Exception as e:
            print(f"[HPT] Trial {t+1} failed with exception: {e}")
            all_results.append({"cfg": sampled_cfg, "summary": {"mean_reward": -9999.0, "success_rate": 0.0, "total_steps": 0}, "error": str(e)})

    # Sort by mean_reward descending
    all_results_sorted = sorted(all_results, key=lambda x: x["summary"].get("mean_reward", -9999.0), reverse=True)

    # Print aggregated results
    print("\n[HPT] All trials finished. Top 10 results:")
    for i, res in enumerate(all_results_sorted[:10]):
        cfg = res["cfg"]
        s = res["summary"]
        print(f"Rank {i+1}: mean_reward={s['mean_reward']:.4f}, success_rate={s['success_rate']:.4f}, total_steps={s['total_steps']}")
        print(f"   cfg: lr={cfg.get('lr')}, epochs={cfg.get('epochs')}, episodes_per_update={cfg.get('episodes_per_update')}, lam={cfg.get('lam')}, vf_coeff={cfg.get('vf_coeff')}, entropy_coeff={cfg.get('entropy_coeff')}")
        print(f"   logs: {res.get('log_file')}")

    return all_results_sorted

if __name__ == "__main__":
    # Example usage: adjust lists as you wish
    results = hyperparameter_tuning(
        save_path="./checkpoints_hpt",
        rgat_version=["RelationalGATV1"],
        query_path_version=["QueryPathRLV1"],
        querypath_cfg=[{"encoder": "bert", "num_hops": 100}],
        epochs=[3],  # each trial runs 3 epochs
        episodes_per_update=[6, 8],
        gamma=[0.99],
        lam=[0.95, 0.97],
        lr=[2e-6, 1e-6, 1e-5],
        vf_coeff=[0.4, 0.5],
        entropy_coeff=[0.01, 0.0],
        test_run=True,   # IMPORTANT: dataset loaded with test=True => 50 samples
        device="cuda" if torch.cuda.is_available() else "cpu",
        num_trials=10,
        seed=42,
    )
    # optionally save summary to json
    with open("hpt_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print("[HPT] Summary saved to hpt_summary.json")
