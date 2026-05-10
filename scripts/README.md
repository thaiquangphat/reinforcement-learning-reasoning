# Scripts Directory

Contains all executable Python scripts organized by purpose:

```
scripts/
├── training/          # Model training scripts
│   ├── train_rl.py
│   ├── train_rl_0.py
│   ├── train_rl_0_1.py through train_rl_0_5.py
│   ├── train_rl_2.py
│   ├── train_rl_3.py
│   └── README.md
├── inference/         # Model inference scripts
│   ├── inference_rl.py
│   ├── inference_rl_0.py
│   ├── inference_rl_0_1.py through inference_rl_0_5.py
│   ├── inference_rl_2.py
│   ├── inference_rl_3.py
│   └── README.md
└── utils/             # Utility and helper scripts
    ├── hyperparam_tuning.py
    ├── plot_train_logs.py
    ├── ultis.py
    └── README.md
```

## Quick Start

1. **Training**: `python scripts/training/train_rl.py`
2. **Inference**: `python scripts/inference/inference_rl.py`
3. **Visualization**: `python scripts/utils/plot_train_logs.py`
