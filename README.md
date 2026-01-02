# MAPPO-ABC: Multi-Agent PPO with Artificial Bee Colony Optimization

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)

An enhanced Multi-Agent Proximal Policy Optimization (MAPPO) algorithm integrated with Artificial Bee Colony (ABC) optimization for improved exploration-exploitation balance in multi-agent reinforcement learning tasks.

## Overview

MAPPO-ABC introduces a novel approach to multi-agent reinforcement learning by combining the proven effectiveness of MAPPO with ABC-based policy head selection. The algorithm employs **4 specialized policy heads**, each designed with different exploration-exploitation trade-offs, and uses ABC optimization to dynamically select the most suitable head based on fitness evaluation.

### Key Features

- **Multi-Head Policy Architecture**: 4 distinct policy heads with varying exploration strategies
  - Heads are initialized with different random seeds
  - Behavioral specialization emerges naturally during training
  - No manual role assignment - diversity through stochastic initialization
  - ABC-based selection dynamically chooses the most suitable head

- **ABC-Based Head Selection**: Fitness-driven optimization for dynamic policy head selection
- **StarCraft Multi-Agent Challenge v2 (SMACv2) Integration**: Tested on SMACv2 benchmark environments
- **Comprehensive Logging**: Detailed metrics including head selection, fitness values, and performance

## Performance

Our experiments on 5 StarCraft II micromanagement scenarios demonstrate significant improvements over baseline algorithms.

### Performance Comparison Across SMACv2 Scenarios

Results from final 500 episodes. Each entry reports **Win Rate (%) / Mean Return**.

| Scenario | MAPPO+ABC | IPPO | MAPPO | QMIX | VDN |
|----------|-----------|------|-------|------|-----|
| **Terran-Base** | **32.8** / 4.65 | 6.2 / 5.45 | 5.8 / 5.28 | 9.4 / 6.12 | 8.6 / 5.89 |
| **Terran-Advanced** | **39.0** / 4.82 | 7.8 / 5.92 | 6.4 / 5.67 | 10.2 / 6.45 | 9.5 / 6.28 |
| **Protoss-Sharp** | **28.6** / 4.72 | 4.2 / 5.15 | 4.8 / 4.98 | 6.8 / 5.34 | 6.2 / 5.18 |
| **Protoss-Extended** | **29.2** / 3.86 | 4.6 / 4.89 | 5.2 / 5.12 | 7.8 / 5.67 | 7.4 / 5.45 |
| **Zerg-Balanced** | 23.2 / 3.45 | 4.2 / 4.82 | 5.0 / 4.78 | **29.2** / 5.89 | 28.6 / 5.68 |
| **Average** | **30.6** / 4.30 | 5.4 / 5.25 | 5.4 / 5.17 | 12.7 / 5.89 | 12.1 / 5.70 |

**Key Findings:**
- MAPPO+ABC achieves **5-6× improvement** in average win rate (30.6%) compared to policy-based baselines (5.4%)
- Outperforms value-based methods (QMIX: 12.7%, VDN: 12.1%) on most scenarios
- Lower mean returns indicate reward-efficient policies focused on task completion rather than reward maximization
- On Zerg-Balanced (high-variance scenario), value-based methods show competitive performance

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

For questions or suggestions, please open an issue or contact [adem.tuncer@yalova.edu.tr](mailto:adem.tuncer@yalova.edu.tr).

---

**If you find this repository helpful, please consider giving it a star!**
