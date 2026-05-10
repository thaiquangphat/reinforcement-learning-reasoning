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
def load_dataset(path, encoder_name, tag="2wikiqa", test=False):
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

# ----------- Load Checkpoint -----------
def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda'))
    model.load_state_dict(checkpoint)
    return model

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



def inference(
    checkpoint_path: str = "./checkpoints",
    using_epoch: int = 1,
    rgat_version: str = "RelationalGATV1",
    query_path_version: str = "QueryPathRLV1",
    test_run: bool = False,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    device = torch.device(device)
    querypath_cfg = None

    run_name = f"inference_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    run_dir = "logs/" + run_name
    os.makedirs(run_dir, exist_ok=True)
    local_log, log_file = setup_local_logger(run_name, log_dir=run_dir)
    print(f"[INFO] Logging to {log_file}")

    model_checkpoint_path = checkpoint_path + "/model" + f"querypathrl_epoch{using_epoch}.pt"
    rgat_checkpoint_path = checkpoint_path + "/rgat" + f"gatencoder_epoch{using_epoch}.pt"

    model = QUERY_PATH_RL[query_path_version](device=device, **(querypath_cfg or {}))
    model.to(device)
    print("[INFO] Initialized Query path model:", query_path_version)

    gat_encoder = None
    gat_cls = RELATIONAL_GAT.get(rgat_version, None)
    if gat_cls is not None:
        gat_encoder = gat_cls(in_dim=model.in_dim).to(device)
        print("[INFO] Initialized GAT encoder:", rgat_version)

    local_log({
        "event": "initialized_model",
        "checkpoints": checkpoint_path,
        "model_checkpoints": query_path_version,
        "rgat_checkpoints": rgat_version
    })

    model = load_checkpoint(model, model_checkpoint_path)
    print("[INFO] Loaded Query path model checkpoint:", model_checkpoint_path)
    gat_encoder = load_checkpoint(gat_encoder, rgat_checkpoint_path)
    print("[INFO] Loaded GAT encoder checkpoint:", rgat_checkpoint_path)

    local_log({
        "event": "loaded_model",
        "model": model_checkpoint_path,
        "rgat_model": rgat_checkpoint_path
    })

    dataset = None
    dataloader = None
    
    dataset = load_dataset(path='dataset/test.jsonl', encoder_name='bert', test=test_run)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    if dataloader is None:
        print("[INFO] No dataloader configured; exiting training loop.")

    pbar = tqdm(dataloader, desc="Inference ", ascii=" .-=#")

    true_samples = 0
    false_samples = 0

    for _, batch in enumerate(pbar):
        adj = batch["adj"].to(device)
        rel_adj = batch["rel_adj"]
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

            found = False
            idx_list_ret = None

            for n_hops in range(20):
                with torch.no_grad():
                    ep = model.run_inference(
                        start_idx=start_idx,
                        question=question,
                        adj=adj.to(device),
                        rgat_nodes=rgat_nodes.to(device),
                        num_hops=model.num_hops,
                        mask_visited=True,
                    )

                idx_list = ep["idx_list"]
                if target_idx in idx_list:
                    found = True
                    idx_list_ret = idx_list
                    break
                elif n_hops==19 and target_idx not in idx_list:
                    idx_list_ret = idx_list

            if found:
                true_samples = true_samples + 1
            else:
                false_samples = false_samples + 1

            reasoning_path = []
            for i in range(len(idx_list_ret) - 1):
                head_idx = idx_list_ret[i]
                tail_idx = idx_list_ret[i+1]

                head = nodes[start_idx]
                tail = nodes[tail_idx]
                relation = rel_adj[head_idx][tail_idx]

                reasoning_path.append([head, relation, tail])

            local_log({
                "event": "inference_step",
                "is_correct": found,
                "question": question,
                "target_answer": target_node,
                "reasoning_path": reasoning_path
            })
            
    accuracy = true_samples / (true_samples + false_samples)
    local_log({
        "event": "inference_end",
        "true_samples": true_samples,
        "false_samples": false_samples,
        "accuracy": accuracy 
    })

    print(f"[INFO] Inference done. Logged to {log_file}")


if __name__ == "__main__":
    inference(
        checkpoint_path = "./checkpoints",
        using_epoch=5,
        rgat_version="RelationalGATV1",
        query_path_version="QueryPathRLV1",
        test_run=True,
    )