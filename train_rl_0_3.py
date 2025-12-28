"""
train_dqn_rl.py

DQN baseline for multihop QA on knowledge graphs.

- Sparse terminal reward (+1 if target reached)
- Experience replay
- Target network
- Variable action space handled via neighbor scoring
"""

import os
import json
import random
import datetime
from pathlib import Path
from collections import deque
from typing import Dict, Any

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


# --------------------------- Replay Buffer ---------------------------
class ReplayBuffer:
    def __init__(self, capacity: int = 50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition: Dict[str, Any]):
        self.buffer.append(transition)

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# --------------------------- Training Loop ---------------------------
def train(
    dataset_path: str = "dataset/traintestamr.jsonl",
    save_path: str = "./checkpoints_dqn",
    tag: str = "hotpotqa",
    encoder: str = "bert",
    model_version: str = "QueryPathRLV03",
    rgat_version: str = "RelationalGATV1",
    epochs: int = 10,
    num_hops: int = 20,
    gamma: float = 0.99,
    lr: float = 1e-4,
    batch_size: int = 32,
    replay_capacity: int = 50000,
    target_update: int = 1000,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.1,
    epsilon_decay: float = 0.995,
    test_run: bool = False,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):

    print("[INFO] Starting DQN baseline training")
    os.makedirs(save_path, exist_ok=True)
    device = torch.device(device)

    run_name = f"dqn_rl_{datetime.datetime.now():%Y%m%d_%H%M%S}"
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

    # ---------------- Models ----------------
    q_net = QUERY_PATH_RL[model_version](
        encoder=encoder,
        num_hops=num_hops,
        device=device,
    ).to(device)

    target_net = QUERY_PATH_RL[model_version](
        encoder=encoder,
        num_hops=num_hops,
        device=device,
    ).to(device)

    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(replay_capacity)

    gat_encoder = None
    gat_cls = RELATIONAL_GAT.get(rgat_version, None)
    if gat_cls is not None:
        gat_encoder = gat_cls(in_dim=q_net.in_dim).to(device)
        print("[INFO] Initialized RGAT encoder:", rgat_version)

    epsilon = epsilon_start
    global_step = 0

    # ---------------- Training ----------------
    for epoch in range(epochs):
        epoch_success = 0
        epoch_episodes = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", ascii=" .-=#")

        for batch in pbar:
            adj = batch["adj"].to(device)
            rel_feat = batch["rel_feat"].to(device)
            node_feat = batch["node_feat"].to(device)
            nodes = batch["nodes"]
            question = batch["question"][0]

            if adj.dim() == 3:
                adj = adj.squeeze(0)
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

            for qury in batch["query"]:
                start_idx = nodes.index(qury[0][0])
                target_idx = nodes.index(qury[-1][-1])

                ep = q_net.run_episode(
                    start_idx=start_idx,
                    question=question,
                    adj=adj,
                    rgat_nodes=rgat_nodes,
                    target_idx=target_idx,
                    epsilon=epsilon,
                )

                for tr in ep["transitions"]:
                    replay_buffer.push(tr)

                epoch_success += int(ep["success"])
                epoch_episodes += 1

                # ---------------- DQN Update ----------------
                if len(replay_buffer) >= batch_size:
                    batch_tr = replay_buffer.sample(batch_size)
                    loss = 0.0

                    for tr in batch_tr:
                        q_vals = q_net.q_values(tr["q"], tr["h_t"], tr["cand_h"])
                        q_sa = q_vals[tr["action"]]

                        with torch.no_grad():
                            if tr["done"] or tr["cand_h_next"] is None:
                                target = torch.tensor(tr["reward"], device=device)
                            else:
                                q_next = target_net.q_values(
                                    tr["q"],
                                    tr["next_h"],
                                    tr["cand_h_next"],
                                ).max()
                                target = tr["reward"] + gamma * q_next


                        loss += F.mse_loss(q_sa, target)

                    loss = loss / batch_size
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
                    optimizer.step()

                    global_step += 1
                    if global_step % target_update == 0:
                        target_net.load_state_dict(q_net.state_dict())

                epsilon = max(epsilon_end, epsilon * epsilon_decay)

                local_log({
                    "event": "train_step",
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "epsilon": epsilon,
                    "success_rate": epoch_success / max(1, epoch_episodes),
                })

        local_log({
            "event": "epoch_end",
            "epoch": epoch + 1,
            "global_step": global_step,
            "success_rate": epoch_success / max(1, epoch_episodes),
            "episodes": epoch_episodes,
        })

        print(
            f"[Epoch {epoch+1}] success_rate={epoch_success / max(1, epoch_episodes):.4f}"
        )

        torch.save(
            q_net.state_dict(),
            model_save_path + f"/dqn_querypathrl_epoch{epoch+1}.pt",
        )
        if gat_encoder is not None:
            torch.save(
                gat_encoder.state_dict(),
                gat_save_path + f"/dqn_rgat_epoch{epoch+1}.pt",
            )

    print("[INFO] DQN baseline training finished.")


if __name__ == "__main__":
    train(
        test_run=True
    )
