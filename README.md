# Reinforcement Learning-Based Adaptive Boundary Constraint Handling for Particle Swarm Optimization

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Deep Q-Network integration for adaptive boundary constraint-handling method selection in Particle Swarm Optimization.

---

## Overview

Position updates in PSO frequently generate solutions violating bound constraints. This work investigates **DQN-based adaptive selection** of boundary constraint-handling methods, learning policies that select repair strategies based on population state rather than fixed or probability-based rules.

### Key Contributions

1. **Empirical evaluation** of 26 hybrid BCHMs on CEC2006, informing data-driven pool construction
2. **MDP formulation** with population-level state encoding (9 dimensions) reducing decision complexity to per-generation scope
3. **DQN implementation** with epsilon-greedy exploration and violation-filtered experience replay
4. **Experimental validation** showing statistically significant improvements on 5 of 12 test problems

---

## Problem & Approach

**Constrained Optimization**:
```
minimize    f(x)
subject to  g_i(x) ≤ 0, h_j(x) = 0, a_i ≤ x_i ≤ b_i
```

**Boundary Constraint Handling**: When PSO updates violate bounds, repair methods (Boundary, Random, Exponential, Reflection, etc.) combined with velocity strategies (DeB/RaB) restore feasibility. Choice significantly impacts performance.

**RL Formulation**:
- **State** (9D): generation progress, best fitness, diversity, violation rate, stagnation counter
- **Actions** (5): Random&DeB, ExpTarget&RaB, ExpBest&DeB, MidBest&DeB, Boundary&RaB
- **Reward**: fitness improvement + diversity preservation - stagnation penalty

**Network**: 9 → 64 → 64 → 5 feedforward (5,125 parameters)

---

## Installation

```bash
git clone https://github.com/mehmihai/rl-bchm.git
cd rl-bchm
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Requirements**: Python 3.8+, PyTorch, NumPy, SciPy

---

## Usage

**Training**:
```bash
python scripts/train_dqn.py --episodes 5000 --max-evals 100000 --train-split
```

**Evaluation**:
```bash
python scripts/run_dqn_eval.py --model results/experiments/dqn/dqn_*/dqn_final.pth --runs 25
```

**Baselines**:
```bash
python scripts/run_fixed_methods.py --problems all --runs 25
python scripts/run_abchs_castillo.py --problems all --runs 25
```

**Train/Test Split**:
- Training: 8 problems (G01, G03, G04, G06, G07, G10, G11, G14)
- Testing: 12 problems (G02, G05, G08, G09, G13, G15, G16, G17, G18, G19, G23, G24)

---

## Benchmark

**CEC2006**: 21 constrained problems (G01-G24 excluding G20-G22)
- Dimensions: 2-24 variables
- Constraints: 1-38 inequalities, 0-11 equalities
- Budget: 500,000 function evaluations

---

## Results

**Experiment 1 - Fixed Methods**: 13 methods achieved first rank across problems; no universal best.

**Experiment 2 - ABCHS Pools**: Revised pool (data-driven) outperformed original on 16/21 problems with significant gains on G01 (67%), G04 (5 orders of magnitude), G07 (55%), G16 (3 orders of magnitude), G19 (86%).

**Experiment 3 - DQN Selection**:
- **7/12 test problems**: Best mean error
- **5/12 problems**: Statistically significant (p < 0.05) - G01, G13, G14, G15, G19
- **Generalization**: Policy learned on 8 training problems transferred to 12 unseen instances
- **Method selection**: Agent preferentially selected ExpBest&DeB (30-48%) on G13, G14, G19, matching fixed-method performance rankings

---

## Configuration

**PSO**: 100 particles, constriction k=0.729, c₁=c₂=1.49445, Deb's constraint handling

**DQN**: Adam lr=0.001, γ=0.95, batch=128, buffer=50k (τ=0.15 violation filter), ε: 1.0→0.2, target update every 10 episodes

---

## Citation

```bibtex
@inproceedings{mitran2025rl,
  title={Reinforcement Learning-Based Adaptive Boundary Constraint Handling
         for Particle Swarm Optimization},
  author={Mitran, Madalina-Andreea and Ene, Mihai-Lucian},
  booktitle={Proceedings of the International Conference},
  year={2025}
}
```

---

**Repository**: https://github.com/mehmihai/rl-bchm
**Status**: Research Implementation
