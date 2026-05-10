# Reinforcement Learning for Reasoning

A reinforcement learning framework for reasoning tasks using graph attention networks and multiple RL algorithms (PPO, DQN, hierarchical approaches).

## Project Structure

```
├── scripts/                    # All executable scripts
│   ├── training/              # Model training scripts
│   ├── inference/             # Model inference scripts
│   └── utils/                 # Utility & analysis scripts
├── checkpoints_archive/       # Model checkpoints by variant
├── src/                       # Source code & models
│   ├── model/                 # Model architectures
│   └── graph_attention_network/
├── dataloader/                # Data loading utilities
├── dataset/                   # Training datasets
│   ├── train.jsonl
│   ├── test.jsonl
│   └── raw_amr/              # Raw AMR graph data
├── logs/                      # Training logs
├── docs/                      # Documentation
└── requirements.txt           # Dependencies
```

## Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Optional: Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Training

Run training scripts from the `scripts/training/` directory:

```bash
# Base training
python scripts/training/train_rl.py

# Variant-specific training
python scripts/training/train_rl_0.py      # Variant 0
python scripts/training/train_rl_2.py      # Variant 2
python scripts/training/train_rl_3.py      # Variant 3

# Variant 0 sub-experiments
python scripts/training/train_rl_0_1.py    # Through train_rl_0_5.py
```

**Outputs:** Checkpoints are saved to `checkpoints_archive/` with subdirectories for each variant.

### 3. Inference

Run inference scripts from the `scripts/inference/` directory:

```bash
# Base inference
python scripts/inference/inference_rl.py

# Variant-specific inference
python scripts/inference/inference_rl_0.py      # Variant 0
python scripts/inference/inference_rl_2.py      # Variant 2
python scripts/inference/inference_rl_3.py      # Variant 3

# Variant 0 sub-experiments
python scripts/inference/inference_rl_0_1.py    # Through inference_rl_0_5.py
```

### 4. Analysis & Utilities

Use utility scripts from `scripts/utils/`:

```bash
# Hyperparameter tuning
python scripts/utils/hyperparam_tuning.py

# Visualize training logs
python scripts/utils/plot_train_logs.py

# Other utilities
python scripts/utils/ultis.py
```

## Model Checkpoints

Checkpoints are organized by training variant in `checkpoints_archive/`:

| Directory | Description |
|-----------|-------------|
| `checkpoints_main/` | Main/baseline model |
| `checkpoints_baseline/` | Baseline variant |
| `checkpoints_baseline_ppo/` | PPO baseline |
| `checkpoints_baseline_ppo_gae/` | PPO with GAE |
| `checkpoints_dqn/` | DQN algorithm |
| `checkpoints_hier/` | Hierarchical approach |

## Data

- **Training Data:** `dataset/train.jsonl`
- **Test Data:** `dataset/test.jsonl`
- **Raw AMR Graphs:** `dataset/raw_amr/` (LDC2020T02 format)
- **Processed Data:** `dataset/amr_decompose_llm_enhance.jsonl` (with LLM enhancements, contact author for dataset)

## Documentation

- `docs/baseline.md` - Baseline model documentation
- `docs/amr_hrl.md` - Hierarchical RL with AMR graphs
- `docs/emb_hrl.md` - Embedding-based hierarchical RL

## Dependencies

See `requirements.txt` for the complete list.

## Workflow

2. **Train Model:** `python scripts/training/train_rl.py`
3. **Monitor Training:** Check `logs/`
4. **Run Inference:** `python scripts/inference/inference_rl.py`
5. **Analyze Results:** `python scripts/utils/plot_train_logs.py`

## Notes

- Each training variant (0, 2, 3) uses different hyperparameters or algorithms
- Variant 0 has 5 sub-experiments (0_1 through 0_5)
- Training logs are saved to `logs/` for TensorBoard visualization
- Checkpoints contain model weights and training state
