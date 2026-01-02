# MAPPO-ABC: Multi-Agent PPO with Artificial Bee Colony Optimization

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)

An enhanced Multi-Agent Proximal Policy Optimization (MAPPO) algorithm integrated with Artificial Bee Colony (ABC) optimization for improved exploration-exploitation balance in multi-agent reinforcement learning tasks.

## Overview

MAPPO-ABC introduces a novel approach to multi-agent reinforcement learning by combining the proven effectiveness of MAPPO with ABC-based policy head selection. The algorithm employs **4 specialized policy heads**, each designed with different exploration-exploitation trade-offs, and uses ABC optimization to dynamically select the most suitable head based on fitness evaluation.

### Key Features

- **Multi-Head Policy Architecture**: 4 distinct policy heads with varying exploration strategies
  - Conservative Head (H0): Low exploration, high exploitation
  - Balanced Head (H1): Moderate exploration-exploitation
  - Aggressive Head (H2): High exploration focus
  - Exploratory Head (H3): Maximum exploration emphasis

- **ABC-Based Head Selection**: Fitness-driven optimization for dynamic policy head selection
- **StarCraft Multi-Agent Challenge v2 (SMACv2)**: Tested on SMACv2 benchmark environments
- **Comprehensive Logging**: Detailed metrics including head selection, fitness values, and performance

## Performance

Our experiments on 5 StarCraft II micromanagement scenarios demonstrate significant improvements over baseline algorithms.

### Win Rate Comparison (Final 100 Episodes)

| Map | MAPPO-ABC | MAPPO | IPPO | QMIX |
|-----|-----------|-------|------|------|
| **Protoss-Advanced** | **4.64** | 5.43 | 3.35 | 7.47 |
| **Protoss-Sharp** | **3.53** | 4.87 | 6.78 | 5.60 |
| **Terran-Base** | **3.83** | 4.11 | 3.61 | 4.33 |
| **Terran-Base-Advanced** | **4.08** | 3.25 | 5.36 | 4.09 |
| **Zerg** | **6.46** | 6.87 | 6.31 | 8.07 |

*Lower return values indicate better performance in these scenarios*

### Average Episode Returns

| Map | MAPPO-ABC | MAPPO | IPPO | QMIX |
|-----|-----------|-------|------|------|
| **Protoss-Advanced** | 4.72 | 4.97 | 3.35 | 6.62 |
| **Protoss-Sharp** | 2.98 | 5.22 | 5.85 | 4.64 |
| **Terran-Base** | 3.43 | 3.62 | 3.42 | 3.34 |
| **Terran-Base-Advanced** | 3.16 | 3.29 | 4.43 | 3.19 |
| **Zerg** | 5.79 | 7.08 | 5.81 | 7.62 |

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryozdemir/MAPPO-ABC.git
cd MAPPO-ABC

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install StarCraft II (required for SMAC)
bash install_sc2.sh
```

### Training

```bash
# Train on a specific map
python scripts/train/train_mappo_abc.py --map protoss_extended --seed 42

# Train with custom configuration
python scripts/train/train_mappo_abc.py --config configs/mappo_abc_custom.yaml
```

### Evaluation

```bash
# Evaluate a trained model
python scripts/eval/evaluate.py --checkpoint models/mappo_abc_protoss.pth --map protoss_extended --episodes 100

# Run evaluation on all maps
bash scripts/eval/eval_all_maps.sh
```

## Project Structure

```
mappo-abc-smac/
├── src/
│   ├── algorithms/
│   │   ├── mappo_abc.py          # Main MAPPO-ABC algorithm
│   │   ├── mappo.py              # Baseline MAPPO
│   │   ├── ippo.py               # Baseline IPPO
│   │   ├── qmix.py               # Baseline QMIX
│   │   ├── vdn.py                # Baseline VDN
│   │   ├── abc_optimizer.py      # ABC optimization logic
│   │   └── buffer.py             # Replay buffer
│   ├── networks/
│   │   ├── policy_network.py     # Multi-head policy network
│   │   ├── value_network.py      # Critic network
│   │   └── heads.py              # Individual policy heads
│   ├── envs/
│   │   └── smac_wrapper.py       # SMAC environment wrapper
│   └── utils/
│       ├── logger.py             # Logging utilities
│       ├── config.py             # Configuration management
│       └── utils.py              # Helper functions
├── configs/
│   ├── mappo_abc_default.yaml    # Default configuration
│   └── maps/                     # Map-specific configs
├── scripts/
│   ├── train/
│   │   ├── train_mappo_abc.py    # MAPPO-ABC training script
│   │   └── train_baselines.py    # Baseline training script
│   └── eval/
│       └── evaluate.py           # Evaluation script
├── models/                        # Saved model checkpoints
├── results/                       # Experiment results
└── requirements.txt
```

## Configuration

Key hyperparameters can be configured in `configs/mappo_abc_default.yaml`:

```yaml
# Policy Network
hidden_dim: 256
n_heads: 4
activation: "relu"

# ABC Optimizer
abc_food_sources: 20
abc_iterations: 50
abc_limit: 10

# Training
lr: 3e-4
gamma: 0.99
gae_lambda: 0.95
clip_param: 0.2
value_loss_coef: 1.0
entropy_coef: 0.01
max_grad_norm: 10.0

# Environment
n_episodes: 5000
episode_length: 400
batch_size: 32
```

## Algorithm Details

### Multi-Head Policy Architecture

Each policy head implements a different exploration strategy:

- **H0 (Conservative)**: ε=0.05, low temperature softmax
- **H1 (Balanced)**: ε=0.15, medium temperature
- **H2 (Aggressive)**: ε=0.25, high temperature  
- **H3 (Exploratory)**: ε=0.40, very high temperature

### ABC Optimization Process

1. **Initialization**: Each head starts with equal fitness
2. **Episode Execution**: All heads generate trajectories
3. **Fitness Evaluation**: Heads evaluated based on episode returns
4. **Selection**: ABC algorithm selects best-performing head
5. **Update**: Selected head's policy is updated via PPO
6. **Iteration**: Process repeats with updated fitness values

The ABC optimizer uses:
- **Employed Bees**: Explore current food sources (policy heads)
- **Onlooker Bees**: Exploit promising food sources
- **Scout Bees**: Explore new solutions when needed

## Monitoring Training

Training progress is logged using TensorBoard:

```bash
tensorboard --logdir results/
```

Tracked metrics include:
- Episode returns
- Win rates
- Policy loss
- Value loss
- Entropy
- Head selection frequency
- Individual head fitness values

## Reproducing Results

To reproduce the results from our experiments:

```bash
# Run MAPPO+ABC experiments
bash scripts/reproduce_all.sh

# Run baseline experiments
bash scripts/train_all_baselines.sh

# Run specific algorithm on specific map
python scripts/train/train_mappo_abc.py --map protoss_extended --seed 42 --episodes 5000

# Train baseline algorithm
python scripts/train/train_baselines.py --algorithm ippo --map protoss_extended --seed 42
```

### Available Baseline Algorithms

The repository includes implementations of four baseline algorithms:

- **IPPO** (Independent PPO): Each agent learns independently using PPO
- **MAPPO** (Multi-Agent PPO): Centralized training with decentralized execution
- **QMIX**: Value-based method with monotonic mixing network
- **VDN** (Value Decomposition Networks): Simple additive value decomposition

Results will be saved in `results/` with detailed logs and checkpoints.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{ozdemir2025mappo_abc,
  title={Artificial Bee Colony as an Exploration Meta-Layer for MAPPO in Cooperative Multi-Agent Systems},
  author={Özdemir, Koray and Şara, Muhammed and Tuncer, Adem and Eken, Süleyman},
  journal={Under Review},
  year={2025},
  note={Source code available at: \url{https://github.com/kryozdemir/MAPPO-ABC}}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [SMAC](https://github.com/oxwhirl/smac) - StarCraft Multi-Agent Challenge
- [PyMARL](https://github.com/oxwhirl/pymarl) - Multi-agent RL framework
- Original MAPPO implementation from [on-policy](https://github.com/marlbenchmark/on-policy)

## Contact

For questions or suggestions, please open an issue or contact [korayozdemir34@gmail.com](mailto:korayozdemir34@gmail.com).

---

**If you find this repository helpful, please consider giving it a star!**
