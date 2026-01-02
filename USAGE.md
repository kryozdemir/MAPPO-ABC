# Usage Guide

This guide provides detailed instructions on using MAPPO-ABC for your multi-agent RL experiments.

## Table of Contents
1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Training](#training)
4. [Evaluation](#evaluation)
5. [Configuration](#configuration)
6. [Visualization](#visualization)
7. [Custom Environments](#custom-environments)

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/kryozdemir/MAPPO-ABC.git
cd MAPPO-ABC
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install StarCraft II
```bash
bash install_sc2.sh
```

This will download and install StarCraft II and SMAC maps to `~/StarCraftII/`.

## Quick Start

Train MAPPO-ABC on the 3m map:
```bash
python scripts/train/train_mappo_abc.py --map 3m --episodes 2000
```

## Training

### Basic Training

Train on a specific map:
```bash
python scripts/train/train_mappo_abc.py \
    --map 3m \
    --episodes 5000 \
    --seed 42
```

### Training Baseline Algorithms

To compare with baseline algorithms:

```bash
# Train IPPO
python scripts/train/train_baselines.py --algorithm ippo --map 3m --episodes 5000

# Train MAPPO
python scripts/train/train_baselines.py --algorithm mappo --map 3m --episodes 5000

# Train QMIX
python scripts/train/train_baselines.py --algorithm qmix --map 3m --episodes 5000

# Train VDN
python scripts/train/train_baselines.py --algorithm vdn --map 3m --episodes 5000

# Train all baselines on all maps
bash scripts/train_all_baselines.sh
```

### Available SMAC Maps

The algorithm has been tested on:
- `protoss_extended` (Protoss-Advanced)
- `protoss_sharp` (Protoss-Sharp)  
- `terran_base` (Terran-Base)
- `terran_advanced` (Terran-Base-Advanced)
- `zerg_balanced` (Zerg)

Other SMAC maps should work as well:
- `3m`, `8m`, `25m` - Marines
- `2s3z`, `3s5z` - Stalkers and Zealots
- `MMM`, `MMM2` - Mixed units
- And more...

### Custom Configuration

Create a custom config file (e.g., `my_config.yaml`):
```yaml
hidden_dim: 512
n_heads: 4
lr: 0.0001
gamma: 0.99
abc:
  food_sources: 30
  iterations: 100
```

Train with custom config:
```bash
python scripts/train/train_mappo_abc.py \
    --map 3m \
    --config my_config.yaml
```

### Training Options

```bash
python scripts/train/train_mappo_abc.py --help
```

Key arguments:
- `--map`: SMAC map name
- `--episodes`: Number of training episodes
- `--seed`: Random seed for reproducibility
- `--config`: Path to YAML config file
- `--save-dir`: Directory to save models (default: `models/`)
- `--log-dir`: Directory for logs (default: `results/`)
- `--save-interval`: Save model every N episodes
- `--eval-interval`: Evaluate every N episodes
- `--device`: `cuda` or `cpu`

## Evaluation

### Evaluate a Trained Model

```bash
python scripts/eval/evaluate.py \
    --checkpoint models/mappo_abc_3m_final.pth \
    --map 3m \
    --episodes 100
```

### Evaluation Options

- `--checkpoint`: Path to model checkpoint (.pth file)
- `--map`: SMAC map name
- `--episodes`: Number of evaluation episodes
- `--seed`: Random seed
- `--device`: `cuda` or `cpu`
- `--render`: Render episodes (requires display)
- `--deterministic`: Use deterministic policy

### Example Output

```
Episode 10/100 | Return: 18.50 | Steps: 120 | Win: True
Episode 20/100 | Return: 19.20 | Steps: 115 | Win: True
...

==================================================
EVALUATION RESULTS
==================================================
Mean Return: 18.750 ± 1.234
Min/Max Return: 15.200 / 20.500
Mean Steps: 118.5
Win Rate: 87.00% (87/100)
==================================================
```

## Configuration

### Default Configuration

See `configs/mappo_abc_default.yaml` for all available parameters.

### Key Hyperparameters

**Network Architecture:**
- `hidden_dim`: Hidden layer size (default: 256)
- `n_heads`: Number of policy heads (default: 4)

**PPO Hyperparameters:**
- `lr`: Learning rate (default: 3e-4)
- `gamma`: Discount factor (default: 0.99)
- `gae_lambda`: GAE lambda (default: 0.95)
- `clip_param`: PPO clip parameter (default: 0.2)
- `value_loss_coef`: Value loss weight (default: 1.0)
- `entropy_coef`: Entropy regularization (default: 0.01)

**ABC Configuration:**
- `food_sources`: Number of food sources (default: 20)
- `iterations`: ABC iterations per update (default: 50)
- `limit`: Abandonment limit (default: 10)

## Visualization

### Plot Training Curves

Compare MAPPO-ABC with baselines:
```bash
python scripts/visualize.py \
    --log-dirs results/mappo_abc_3m results/mappo_3m results/ippo_3m \
    --labels "MAPPO-ABC" "MAPPO" "IPPO" \
    --output-dir plots/
```

This creates:
- `return_comparison.png` - Episode returns
- `battle_won_comparison.png` - Win rates
- `policy_loss_comparison.png` - Policy loss
- `value_loss_comparison.png` - Value loss
- `entropy_comparison.png` - Entropy
- `head_selection.png` - Head selection frequency (MAPPO-ABC only)
- `fitness_evolution.png` - ABC fitness over time (MAPPO-ABC only)

### TensorBoard

View training progress in real-time:
```bash
tensorboard --logdir results/
```

Then open http://localhost:6006 in your browser.

## Custom Environments

To use MAPPO-ABC with a custom environment:

### 1. Create Environment Wrapper

```python
# src/envs/custom_wrapper.py
class CustomEnvWrapper:
    def __init__(self, env_config):
        # Initialize your environment
        pass
    
    def reset(self):
        # Return initial observations
        pass
    
    def step(self, actions):
        # Execute actions
        # Return observations, rewards, done, info
        pass
    
    def get_obs(self):
        # Return current observations for all agents
        pass
    
    def get_state(self):
        # Return global state
        pass
```

### 2. Modify Training Script

```python
# In train_mappo_abc.py
from src.envs.custom_wrapper import CustomEnvWrapper

# Replace StarCraft2Env with your wrapper
env = CustomEnvWrapper(config)
env_info = {
    'n_agents': env.n_agents,
    'obs_shape': env.obs_dim,
    'n_actions': env.n_actions
}
```

### 3. Train

```bash
python scripts/train/train_mappo_abc.py --config configs/custom_env.yaml
```

## Tips and Best Practices

### 1. Hyperparameter Tuning

- Start with default config
- If convergence is slow, try:
  - Increasing `lr` to 5e-4
  - Decreasing `clip_param` to 0.1
  - Increasing `n_epochs` to 6-8

### 2. ABC Configuration

- More `food_sources` = better exploration but slower
- Higher `limit` = more exploitation of good heads
- Balance based on environment complexity

### 3. Debugging

- Use smaller `hidden_dim` (128) for faster iteration
- Reduce `episodes` for quick tests
- Check `results/` logs for abnormal values
- Use TensorBoard for real-time monitoring

### 4. Multi-GPU Training

Currently single-GPU. For multi-GPU:
```python
# Modify MAPPOABC to use DataParallel
self.policy = nn.DataParallel(self.policy)
```

## Common Issues

### Issue: SMAC Installation Fails

**Solution:** Make sure StarCraft II is installed:
```bash
bash install_sc2.sh
export SC2PATH=~/StarCraftII
```

### Issue: CUDA Out of Memory

**Solution:** Reduce batch size or hidden dim:
```yaml
batch_size: 16
hidden_dim: 128
```

### Issue: Training is Unstable

**Solution:** 
- Lower learning rate: `lr: 1e-4`
- Increase gradient clipping: `max_grad_norm: 5.0`
- Check for NaN values in logs

## Support

For questions or issues:
1. Check this usage guide
2. See README.md
3. Open an issue on GitHub
4. Contact adem.tuncer@yalova.edu.tr

## Citation

If you use this code, please cite:
```bibtex
@article{ozdemir2025mappo_abc,
  title={Artificial Bee Colony as an Exploration Meta-Layer for MAPPO in Cooperative Multi-Agent Systems},
  author={Özdemir, Koray and Şara, Muhammed and Tuncer, Adem and Eken, Süleyman},
  journal={Under Review},
  year={2025}
}
```
