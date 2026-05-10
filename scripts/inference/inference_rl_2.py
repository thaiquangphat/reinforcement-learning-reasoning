import os
import math
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

# Import registries / dataset loader from existing repo (as in train_rl.py)
from src import RELATIONAL_GAT, QUERY_PATH_RL  # if you maintain registry
from src.dataloader import RelGraphDataset  # assume same dataloader

# --------------------------- Local JSONL logger ---------------------------
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

# ----------- Load Checkpoint -----------
def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda'))
    model.load_state_dict(checkpoint)
    return model

def inference(
    save_path: str = "./inference",
    rgat_version: str = "RelationalGATV1",
    rgat_checkpoint: str = "epoch_1.pt",
    query_path_version: str = "QueryPathRLV2",
    query_path_cfg: Dict[str, Any] = None,
    query_path_checkpoint: str = "epoch_1.pt",
    test_run: bool = False,
    tag: str="musique",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    args_dict = {
        "event": "input_parameters",
        **locals().copy()
    }
    print(json.dumps(args_dict, indent=2, ensure_ascii=False))

    os.makedirs(save_path, exist_ok=True)
    device = torch.device(device)

    run_name = f"inference_run_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    run_dir = "logs/" + run_name
    os.makedirs(run_dir, exist_ok=True)
    local_log, log_file = setup_local_logger(run_name, log_dir=run_dir)
    print(f"[INFO] Logging to {log_file}")

    local_log(args_dict)

    checkpoint_path = save_path + "/" + run_name
    model_save_path = checkpoint_path + "/model"
    gat_save_path = checkpoint_path + "/rgat"
    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(gat_save_path, exist_ok=True)

    # Instantiate model
    model = QUERY_PATH_RL[query_path_version](device=device, **(query_path_cfg or {}))
    model.to(device)
    print("[INFO] Initialized hierarchical model.")

    # instantiate GAT encoder if available
    gat_encoder = None
    gat_cls = RELATIONAL_GAT.get(rgat_version, None)
    if gat_cls is not None:
        gat_encoder = gat_cls(in_dim=model.in_dim).to(device)
        print("[INFO] Initialized GAT encoder:", rgat_version)

    model = load_checkpoint(model, query_path_checkpoint)
    print("[INFO] Loaded Query path model checkpoint:", query_path_checkpoint)
    gat_encoder = load_checkpoint(gat_encoder, rgat_checkpoint)
    print("[INFO] Loaded GAT encoder checkpoint:", rgat_checkpoint)

    local_log({
        "event": "loaded_model",
        "model": query_path_checkpoint,
        "rgat_model": rgat_checkpoint
    })

    local_log({"event": "train_inference", "run_name": run_name})

    # dataset loader (reuse same function as original repo)
    dataset = None
    dataloader = None
    try:
        # use same loader signature as original train_rl.py
        def load_dataset(path, encoder_name, tag="hotpotqa", test=test_run):
            test_samples = 50
            dataset = []
            with open(path, 'r', encoding='utf-8') as file:
                for line in file:
                    dataset.append(json.loads(line))
            dataset = [d for d in dataset if d['tag']==tag]
            if test_run:
                dataset = dataset[:test_samples]
            return RelGraphDataset(raw_data=dataset, encoder=encoder_name, num_samples=-1, max_nodes=200)
        dataset = load_dataset(path='dataset/test.jsonl', encoder_name='bert', tag=tag, test=False)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    except Exception as e:
        print("[WARN] Dataset loader failed; ensure dataloader exists.", e)
        dataloader = None

    pbar = tqdm(dataloader, desc=f"[INFERENCE]", ascii=" .-=#")

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
        if gat_encoder is not None:
            with torch.no_grad():
                out = gat_encoder(adj.unsqueeze(0), rel_feat, node_feat.unsqueeze(0))
                rgat_nodes = out[0].squeeze(0) if isinstance(out, (tuple, list)) else out.squeeze(0)

        for qury in query:
            start_node = qury[0][0]
            start_idx = nodes.index(start_node)
            target_node = qury[-1][-1]
            target_idx = nodes.index(target_node)

            trials = 5
            trial_paths = []
            found = False
            for _ in range(trials):
                with torch.no_grad():
                    ep = model.run_episode(
                        start_idx=start_idx,
                        question=question,
                        adj=adj.to(device),
                        rgat_nodes=rgat_nodes.to(device),
                        num_hops=model.num_hops,
                        target_idx=None,
                        deterministic=True,
                        mask_visited=True,
                    )

                idx_list = ep["idx_list"]
                if target_idx in idx_list:
                    found = True
                    
                reasoning_path = []
                for i in range(len(idx_list) - 1):
                    head_idx = idx_list[i]
                    tail_idx = idx_list[i+1]

                    head = nodes[start_idx]
                    tail = nodes[tail_idx]
                    relation = rel_adj[head_idx][tail_idx]

                    reasoning_path.append([head, relation, tail])

                trial_paths.append(reasoning_path)

            local_log({
                "event": "inference_step",
                "is_correct": found,
                "question": question,
                "target_answer": target_node,
                "reasoning_path_trials": trial_paths
            })

            if found:
                true_samples += 1
            else:
                false_samples += 1

    accuracy = true_samples / (true_samples + false_samples)
    local_log({
        "event": "inference_end",
        "true_samples": true_samples,
        "false_samples": false_samples,
        "accuracy": accuracy 
    })

if __name__ == "__main__":
    inference(
        checkpoint_path = "./inference",
        rgat_version="RelationalGATV1",
        rgat_checkpoint="epoch_1.pt",
        query_path_version="QueryPathRLV2",
        query_path_cfg={"encoder": "bert", "num_hops": 20, "manager_horizon": 4, "num_prototypes": 128},
        query_path_checkpoint="epoch_1.pt",
        test_run=True,
        tag='hotpotqa',
        device="cuda" if torch.cuda.is_available() else "cpu",
    )