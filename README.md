# Reinforcement Learning for Adaptive Boundary Constraint-Handling in PSO

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**A Deep Q-Network approach for learning adaptive boundary constraint-handling method selection in Particle Swarm Optimization for constrained optimization problems.**

---

## Overview

When solving constrained optimization problems with metaheuristics, particles frequently violate variable bounds during search. **Boundary Constraint-Handling Methods (BCHMs)** repair these violations by relocating particles back into the feasible space. The choice of BCHM significantly impacts algorithm performance, yet most implementations use fixed methods regardless of problem characteristics or search stage.

This work investigates **adaptive BCHM selection using Deep Q-Networks (DQN)**, where a reinforcement learning agent learns to select the most appropriate repair method based on the current optimization state.

### Key Contributions

- **DQN-based adaptive selection** of BCHMs during PSO optimization
- **Ray-parallelized training** for efficient experience collection (12 workers)
- **Boundary violation buffer** focusing learning on constraint-handling scenarios
- **Train/test split** for proper generalization evaluation across 20 CEC2006 problems
- **Comprehensive baselines**: 26 fixed BCHM combinations and adaptive ABCHS

---

## Problem Formulation

### Constrained Optimization

Minimize: `f(x)` subject to:
- Variable bounds: `l_i ≤ x_i ≤ u_i` for all `i`
- Inequality constraints: `g_j(x) ≤ 0` for `j = 1, ..., m`
- Equality constraints: `h_k(x) = 0` for `k = 1, ..., p`

### Boundary Constraint-Handling

When PSO particle `i` violates bounds after velocity update:
```
x_i(t+1) = x_i(t) + v_i(t+1)
```

A BCHM repairs the position to satisfy `l ≤ x_i(t+1) ≤ u`.

### Reinforcement Learning Formulation

**State Space** (9 features):
- Best fitness value and constraint violation
- Population diversity
- Boundary violation rate
- Progress indicator (generation/max_generations)
- Previous method ID

**Action Space** (5 BCHMs):
1. Random&Deb - Random reinitialization
2. ExpTarget&RandB - Exponential correction toward target
3. ExpBest&Deb - Exponential correction toward global best
4. MidBest&Deb - Midpoint toward global best
5. Boundary&RandB - Saturation at bounds

**Reward Function**:
```python
reward = fitness_improvement + diversity_bonus - stagnation_penalty
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     RL-BCHM System                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────┐ │
│  │   CEC2006    │─────→│   RL-PSO     │◄─────│ DQN Agent│ │
│  │ 20 Problems  │      │ PSO + BCHM   │      │ (PyTorch)│ │
│  └──────────────┘      └──────────────┘      └──────────┘ │
│                                │                           │
│                                ▼                           │
│                     ┌──────────────────┐                   │
│                     │ Ray Workers (12) │                   │
│                     │ Experience Pool  │                   │
│                     └──────────────────┘                   │
└─────────────────────────────────────────────────────────────┘
```

### DQN Network

**Architecture**: Fully connected feedforward network
- Input: 9-dimensional state vector
- Hidden layers: 2 × 64 neurons (ReLU activation)
- Output: Q-values for 5 actions

**Training**:
- Boundary violation buffer (stores experiences with prob_infeas_bounds > 0.15)
- Target network updated every 10 episodes
- ε-greedy exploration with adaptive decay
- Batch size: 128, Learning rate: 0.001, γ = 0.95

---

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/rl-bchm.git
cd rl-bchm

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Requirements**: Python 3.8+, PyTorch 2.0+, Ray, NumPy, SciPy

---

## Usage

### Training DQN Agent

```bash
# Standard training (300 episodes, all problems)
python main.py dqn --train --episodes 300 --workers 12

# Full training with train/test split (5000 episodes)
python main.py dqn --train --episodes 5000 --workers 12 --train-split

# Direct script call with custom parameters
python scripts/train_dqn.py \
  --episodes 5000 \
  --workers 12 \
  --max-evals 100000 \
  --train-split \
  --save-interval 500
```

**Training Output**: `results/experiments/dqn/dqn_<timestamp>/`
- `dqn_final.pth` - Trained model
- `training_history.json` - Episode statistics
- `config.json` - Hyperparameters
- Checkpoints every 500 episodes

**Train/Test Split**:
- Training: 10 problems (G1, G3, G4, G6, G7, G10, G11, G14, G16, G18)
- Testing: 10 problems (G2, G5, G8, G9, G13, G15, G17, G19, G23, G24)

### Evaluating Trained Model

```bash
# Evaluate on all problems (25 runs each)
python main.py dqn --eval \
  --model results/experiments/dqn/dqn_*/dqn_final.pth \
  --runs 25

# Evaluate on specific problems
python main.py dqn --eval \
  --model results/experiments/dqn/dqn_*/dqn_final.pth \
  --problems G02 G04 G11 \
  --runs 30 \
  --workers 12
```

**Evaluation Output**: `results/experiments/dqn/evaluation/eval_<timestamp>/`
- Per-problem results with method selection statistics
- Generation-level metrics and convergence trajectories
- Comparison with ABCHS baseline

### Baseline Experiments

```bash
# Fixed BCHM methods (26 combinations)
python main.py fixed --problems all --runs 25

# ABCHS (Castillo et al., 2014)
python main.py castillo --problems all --runs 25

# ABCHS with custom pool
python main.py custom --problems all --runs 25
```

---

## Benchmark Suite

**CEC2006 Special Session** on Constrained Real-Parameter Optimization

| Problem | Dim | Type | LI | NLI | LE | NLE | Active |
|---------|-----|------|----|----|----|----|--------|
| G01 | 13 | Quadratic | 9 | 0 | 0 | 0 | 6 |
| G02 | 20 | Nonlinear | 0 | 2 | 0 | 0 | 1 |
| G04 | 5 | Quadratic | 0 | 6 | 0 | 0 | 2 |
| G06 | 2 | Cubic | 0 | 2 | 0 | 0 | 2 |
| G07 | 10 | Quadratic | 3 | 5 | 0 | 0 | 6 |
| G08 | 2 | Nonlinear | 0 | 2 | 0 | 0 | 0 |
| G09 | 7 | Polynomial | 0 | 4 | 0 | 0 | 2 |
| G10 | 8 | Linear | 3 | 3 | 0 | 0 | 6 |
| ... | ... | ... | ... | ... | ... | ... | ... |

**20 problems total** (G01-G11, G13-G19, G23-G24)  
Dimensions: 2-24 variables  
Constraints: 1-38 inequalities, 0-11 equalities

---

## Implementation Details

### Boundary Constraint-Handling Methods

**Position Repair** (13 methods):
- **Saturation (Bou)**: Clamp to bounds
- **Random (Ran)**: Random reinitialization
- **Reflection (Ref)**: Mirror at boundary
- **Wrapping (Wra)**: Toroidal wrap-around
- **Centroid (Cen)**: Move toward feasible centroid
- **Midpoint (MidT/MidB)**: Midpoint toward target/best
- **Exponential (ExpT/ExpB)**: Exponential decay correction
- **Vector (VecT/VecB)**: Vector projection
- **Evolutionary (Evo)**: Evolutionary-inspired
- **Dismiss (Dis)**: Complete reinitialization

**Velocity Update** (2 strategies):
- **DeB**: Deterministic back (-0.5 × v)
- **RaB**: Random back (-λ × v, λ ∈ [0,1])

**Total combinations**: 13 × 2 = 26 methods

### PSO Configuration

- Population size: 100 particles
- Max evaluations: 100,000 per run (training), 500,000 (evaluation)
- Constriction coefficient: χ = 0.7298
- Cognitive/social parameters: c1 = c2 = 2.05
- Constraint handling: Deb's feasibility rules

### DQN Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| State dim | 9 | Population-level features |
| Action dim | 5 | BCHM pool size |
| Hidden layers | [64, 64] | 2-layer MLP |
| Learning rate | 0.001 | Adam optimizer |
| Discount factor | 0.95 | Future reward weight |
| Batch size | 128 | Training mini-batch |
| Buffer capacity | 50,000 | Experience replay |
| ε-start | 1.0 | Initial exploration |
| ε-end | 0.2 | Final exploration |
| ε-decay | (0.2)^(1/episodes) | Exponential decay |
| Target update | 10 episodes | Hard update frequency |

---

## Project Structure

```
rl-bchm/
├── main.py                      # Main entry point
├── scripts/
│   ├── train_dqn.py            # DQN training with Ray
│   ├── run_dqn_eval.py         # DQN evaluation
│   ├── run_fixed_methods.py    # Fixed BCHM baselines
│   ├── run_abchs_castillo.py   # ABCHS baseline
│   └── run_abchs_custom.py     # ABCHS custom pool
├── src/
│   ├── benchmarks/
│   │   └── cec2006.py          # CEC2006 problems
│   ├── core/
│   │   ├── pso.py              # PSO algorithm
│   │   ├── boundary_handler.py # BCHM implementations
│   │   ├── constraint_handler.py # Deb's rules
│   │   └── velocity_strategy.py  # Velocity updates
│   ├── rl/
│   │   ├── dqn.py              # DQN agent
│   │   ├── rl_pso.py           # RL-enhanced PSO
│   │   └── reward_function.py  # State/reward computation
│   └── utils/
│       └── generation_tracker.py # Metrics collection
├── analysis/                    # Result analysis scripts
├── tests/                       # Unit tests
└── results/                     # Experimental results
```

---

## Experimental Results

### Method Selection Diversity

Preliminary results show DQN learns problem-specific policies:
- **High-dimensional problems**: Prefers ExpBest&Deb (exploitation)
- **Multimodal landscapes**: Balances Random&Deb and Boundary&RandB
- **Near-convergence**: Switches to Boundary&RandB (fine-tuning)

### Comparison with Baselines

Performance metrics on CEC2006 (feasibility rate, mean error, success rate):
- **Fixed methods**: Problem-dependent, no adaptation
- **ABCHS**: Stage-based adaptation (exploration → exploitation)
- **DQN**: Continuous state-driven adaptation with learned policy

*Detailed results in `results/experiments/` after running experiments.*

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{rl_bchm_2025,
  author = {Mitran, Madalina-Andreea and Ene, Mihai-Lucian},
  title = {Reinforcement Learning for Adaptive Boundary Constraint-Handling in PSO},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/rl-bchm}
}
```

### Related Work

**BCHM Foundation**:
- Juárez-Castillo et al. (2017). "An Improved Centroid-Based Boundary Constraint-Handling Method in Differential Evolution." *IJPRAI*, 31(11).

**ABCHS Baseline**:
- Castillo et al. (2014). "A comparative study of constraint-handling techniques in Particle Swarm Optimization."

**DQN**:
- Mnih et al. (2015). "Human-level control through deep reinforcement learning." *Nature*, 518(7540).

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- CEC2006 benchmark suite by Liang et al. (2006)
- ABCHS implementation based on Castillo et al. (2014)
- PSO with constriction coefficient by Clerc & Kennedy (2002)
- Deb's constraint-handling rules (2000)

---

**Authors**: Madalina-Andreea Mitran, Mihai-Lucian Ene
**Last Updated**: January 2025
**Status**: Research Implementation
