import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                records.append(data)
            except Exception:
                continue
    df = pd.DataFrame(records)
    df = df[df["event"] == "train_step"].reset_index(drop=True)
    return df

def load_epoch_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                records.append(data)
            except Exception:
                continue
    df = pd.DataFrame(records)
    df = df[df["event"] == "epoch_end"].reset_index(drop=True)
    return df
    
def smooth(series, alpha=0.2):
    return series.ewm(alpha=alpha).mean()

def plot_metrics(df, save_path="plots", run_name="rl_run"):
    Path(save_path).mkdir(parents=True, exist_ok=True)

    steps = df["global_step"]

    # --- 1. Reward & Success Rate ---
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(steps, smooth(df["mean_reward"]), color="tab:blue", label="Mean Reward (EMA)")
    ax1.set_xlabel("Global Step")
    ax1.set_ylabel("Mean Reward", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(steps, smooth(df["success_rate"]), color="tab:green", label="Success Rate (EMA)")
    ax2.set_ylabel("Success Rate", color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green")

    plt.title("Reward & Success Rate Over Time")
    fig.tight_layout()
    plt.savefig(f"{save_path}/{run_name}_reward_success.png", dpi=300)
    plt.close(fig)

    # --- 2. Losses ---
    plt.figure(figsize=(10, 5))
    plt.plot(steps, smooth(df["total_loss"]), label="Total Loss", alpha=0.8)
    plt.plot(steps, smooth(df["actor_loss"]), label="Actor Loss", alpha=0.8)
    plt.plot(steps, smooth(df["value_loss"]), label="Value Loss", alpha=0.8)
    plt.xlabel("Global Step")
    plt.ylabel("Loss Value")
    plt.title("Training Losses Over Time")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{save_path}/{run_name}_losses.png", dpi=300)
    plt.close()

    # --- 3. Entropy ---
    plt.figure(figsize=(10, 4))
    plt.plot(steps, smooth(df["entropy"]), color="tab:orange", label="Entropy (EMA)")
    plt.xlabel("Global Step")
    plt.ylabel("Entropy")
    plt.title("Policy Entropy (Exploration vs Exploitation)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{save_path}/{run_name}_entropy.png", dpi=300)
    plt.close()

    # --- 4. Episode length ---
    plt.figure(figsize=(10, 4))
    plt.plot(steps, smooth(df["avg_episode_len"]), color="tab:red", label="Average Episode Length (EMA)")
    plt.xlabel("Global Step")
    plt.ylabel("Episode Length")
    plt.title("Average Episode Length Over Training")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{save_path}/{run_name}_episode_len.png", dpi=300)
    plt.close()

    print(f"[OK] Step charts saved to: {Path(save_path).resolve()}")

def plot_metrics_epoch(df, save_path="plots", run_name="rl_run"):
    Path(save_path).mkdir(parents=True, exist_ok=True)
    steps = df["epoch"]
    
    # --- 1. Mean success rate ---
    plt.figure(figsize=(10, 4))
    plt.plot(steps, smooth(df["success_rate"]), color="tab:red", label="Success Rate")
    plt.xlabel("Epoch")
    plt.ylabel("Success Rate")
    plt.title("Success Rate Over Training")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{save_path}/{run_name}_success_rate.png", dpi=300)
    plt.close()

    # --- 2. Mean reward ---
    plt.figure(figsize=(10, 4))
    plt.plot(steps, smooth(df["mean_reward"]), color="tab:red", label="Mean Reward")
    plt.xlabel("Global Step")
    plt.ylabel("Mean Reward")
    plt.title("Mean Reward Over Training")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{save_path}/{run_name}_mean_reward.png", dpi=300)
    plt.close()

    print(f"[OK] Epoch charts saved to: {Path(save_path).resolve()}")

if __name__ == "__main__":
    # --- Config ---
    log_path = Path(r"logs/run_20251011_144720/run_20251011_144720.jsonl")
    run_name = str(log_path.stem)
    log_dir = str(log_path.parent)

    # --- Load and plot ---
    df = load_jsonl(log_path)
    print(f"Loaded {len(df)} training steps from {log_path}")
    plot_metrics(df, save_path=log_dir, run_name=run_name)

    # --- Load epoch and plot ---
    df_epoch = load_epoch_jsonl(log_path)
    print(f"Loaded {len(df_epoch)} epochs from {log_path}")
    plot_metrics_epoch(df_epoch, save_path=log_dir, run_name=run_name)