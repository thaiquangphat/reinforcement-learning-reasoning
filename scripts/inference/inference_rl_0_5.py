import os
import json
import datetime
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

from src import RELATIONAL_GAT, QUERY_PATH_RL
from dataloader import RelGraphDataset2


# --------------------------- Config ---------------------------
BEST_CHECKPOINT = {
    "hotpotqa": {
        "model": "checkpoints_baseline_ppo_gae/baseline_ppo_gae_20260101_180526/model/baseline_ppo_gae_epoch8.pt",
        "rgat": "checkpoints_baseline_ppo_gae/baseline_ppo_gae_20260101_180526/rgat/baseline_ppo_gae_rgat_epoch8.pt",
    },
    "2wikiqa": {
        "model": "--",
        "rgat": "--",
    },
    "musique": {
        "model": "--",
        "rgat": "--",
    },
}


# --------------------------- Local JSONL logger ---------------------------
def setup_local_logger(name, log_dir="logs"):
    logs_dir = Path(log_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{name}.jsonl"

    def _safe(val):
        try:
            if isinstance(val, torch.Tensor):
                return val.item() if val.numel() == 1 else val.detach().cpu().tolist()
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


# --------------------------- Checkpoint loader ---------------------------
def load_checkpoint(model, ckpt_path):
    state = torch.load(ckpt_path, map_location="cuda")
    model.load_state_dict(state)
    return model


# --------------------------- Inference ---------------------------
def inference(
    save_path: str = "./inference_baseline_ppo",
    dataset_path: str = "dataset/traintestamr.jsonl",
    tag: str = "hotpotqa",
    encoder: str = "bert",
    model_version: str = "QueryPathRLV05",
    rgat_version: str = "RelationalGATV1",
    model_checkpoint: str = "",
    rgat_checkpoint: str = "",
    trials: int = 10,
    num_hops: int = 20,
    test_run: bool = False,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):

    device = torch.device(device)
    os.makedirs(save_path, exist_ok=True)

    run_name = f"inference_baseline_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    run_dir = Path(save_path) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    local_log, log_file = setup_local_logger(run_name, log_dir=run_dir)
    print(f"[INFO] Logging to {log_file}")

    local_log({
        "event": "input_parameters",
        **locals()
    })

    # ---------------- Model ----------------
    model = QUERY_PATH_RL[model_version](
        encoder=encoder,
        num_hops=num_hops,
        device=device,
    ).to(device)

    gat_encoder = None
    gat_cls = RELATIONAL_GAT.get(rgat_version, None)
    if gat_cls is not None:
        gat_encoder = gat_cls(in_dim=model.in_dim).to(device)

    model = load_checkpoint(model, model_checkpoint)
    if gat_encoder is not None:
        gat_encoder = load_checkpoint(gat_encoder, rgat_checkpoint)

    model.eval()
    if gat_encoder is not None:
        gat_encoder.eval()

    local_log({
        "event": "loaded_model",
        "model_checkpoint": model_checkpoint,
        "rgat_checkpoint": rgat_checkpoint,
    })

    # ---------------- Dataset ----------------
    def load_dataset(path, encoder_name, tag, test_run, split="test"):
        test_samples = 50
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                if item["split"] == split and item["tag"] == tag:
                    data.append(item)
        if test_run:
            data = data[:test_samples]
        return RelGraphDataset2(
            raw_data=data,
            encoder=encoder_name,
            num_samples=6000,
            max_nodes=200,
        )

    dataset = load_dataset(dataset_path, encoder, tag, test_run)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # ---------------- Inference loop ----------------
    true_samples = 0
    false_samples = 0

    pbar = tqdm(dataloader, desc="[BASELINE INFERENCE]", ascii=" .-=#")

    for batch in pbar:
        try:
            adj = batch["adj"].to(device)
            rel_adj = batch["rel_adj"]
            rel_feat = batch["rel_feat"].to(device)
            node_feat = batch["node_feat"].to(device)
            nodes = batch["nodes"]
            question = batch["question"][0]

            if isinstance(adj, (list, tuple)): adj = adj[0]
            if adj.dim() == 3: adj = adj.squeeze(0)
            if isinstance(node_feat, (list, tuple)): node_feat = node_feat[0]
            if node_feat.dim() == 3: node_feat = node_feat.squeeze(0)

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

                found = False
                trial_paths = []
                first_found_path = None

                for _ in range(trials):
                    with torch.no_grad():
                        ep = model.run_episode(
                            start_idx=start_idx,
                            question=question,
                            adj=adj,
                            rgat_nodes=rgat_nodes,
                            target_idx=target_idx,
                            deterministic=False,
                        )

                    idx_list = ep["idx_list"]
                    if target_idx in idx_list:
                        found = True

                    reasoning_path = []
                    path = [start_idx] + idx_list
                    for i in range(len(path) - 1):
                        h = path[i]
                        t = path[i + 1]
                        relation = rel_adj[h][t]
                        reasoning_path.append([nodes[h][0], relation[0], nodes[t][0]])

                    trial_paths.append(reasoning_path)

                    if found and first_found_path is None and len(reasoning_path) > 0:
                        first_found_path = reasoning_path

                local_log({
                    "event": "inference_step",
                    "is_correct": found,
                    "question": question,
                    "target_answer": target_node,
                    "reasoning_path_trials": trial_paths,
                    "first_correct_reasoning_path": first_found_path,
                })

                if found:
                    true_samples += 1
                else:
                    false_samples += 1

                acc = true_samples / max(1, (true_samples + false_samples))
                pbar.set_postfix(acc=f"{acc:.4f}", true=true_samples, false=false_samples)

        except Exception as e:
            print(f"[WARN] Skipping sample: {e}")
            continue

    accuracy = true_samples / max(1, (true_samples + false_samples))
    local_log({
        "event": "inference_end",
        "true_samples": true_samples,
        "false_samples": false_samples,
        "accuracy": accuracy,
    })

    print(f"[INFO] Inference finished. Accuracy={accuracy:.4f}")


# --------------------------- Main ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="hotpotqa")
    args = parser.parse_args()

    model_path = BEST_CHECKPOINT[args.dataset]["model"]
    rgat_path = BEST_CHECKPOINT[args.dataset]["rgat"]

    inference(
        save_path="./inference_baseline_ppo_gae",
        tag=args.dataset,
        model_checkpoint=model_path,
        rgat_checkpoint=rgat_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
