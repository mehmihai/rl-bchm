"""
Reinforcement Learning Module for Adaptive BCHM Selection

This module implements Deep Q-Network (DQN) based method selection
for boundary constraint handling in particle swarm optimization.
"""

from .dqn import DQNAgent, DQNNetwork, BoundaryViolationBuffer, extract_state, compute_reward
from .rl_pso import RL_PSO

__all__ = [
    'DQNAgent',
    'DQNNetwork',
    'BoundaryViolationBuffer',
    'RL_PSO',
    'extract_state',
    'compute_reward',
]

__version__ = '1.0.0'
__author__ = 'Madalina Mitran'
