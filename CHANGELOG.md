# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-01-02

### Added
- Initial release of MAPPO-ABC
- Multi-head policy architecture with 4 specialized heads
- ABC optimizer for dynamic head selection
- Comprehensive SMAC integration
- Training and evaluation scripts
- TensorBoard and CSV logging
- Visualization tools for results analysis
- Baseline MAPPO implementation for comparison
- Detailed documentation (README, USAGE, API docs)
- Example configurations for different scenarios

### Features
- **Multi-Head Policy**: 4 policy heads with different exploration-exploitation trade-offs
  - H0: Conservative strategy
  - H1: Balanced approach
  - H2: Aggressive exploration
  - H3: Maximum exploration
- **ABC Optimization**: Fitness-based head selection using Artificial Bee Colony algorithm
- **Centralized Critic**: CTDE (Centralized Training with Decentralized Execution) architecture
- **GAE**: Generalized Advantage Estimation for variance reduction
- **PPO**: Proximal Policy Optimization for stable updates

### Performance
- Tested on 5 SMAC scenarios:
  - Protoss-Advanced
  - Protoss-Sharp
  - Terran-Base
  - Terran-Base-Advanced
  - Zerg
- Competitive or superior performance compared to IPPO, QMIX, and vanilla MAPPO

### Documentation
- Comprehensive README with quick start guide
- Detailed USAGE guide with examples
- Code documentation with docstrings
- Installation scripts for StarCraft II

## [Unreleased]

### Planned Features
- Multi-GPU training support
- Recurrent policy networks (GRU/LSTM)
- Prioritized experience replay
- Additional SMAC scenarios
- Integration with other multi-agent environments
- Hyperparameter optimization scripts
- Pre-trained model weights

### Known Issues
- None currently reported

---

For detailed changes between versions, see the commit history on GitHub.
