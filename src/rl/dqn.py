"""
Deep Q-Network (DQN) for Adaptive BCHM Selection

Population-level DQN that learns to select the best BCHM for the entire population
at each generation based on optimization state.

Key Features:
- State representation: 9 population-level features (fitness, diversity, boundary violations)
- Action space: 5 BCHMs [Ran&DeB, ExpT&RaB, ExpB&DeB, MidB&DeB, Bou&RaB]
- Prioritized experience replay for handling sparse boundary violations
- Target network for stable Q-value targets
- Îµ-greedy exploration with decay
- Reward: fitness improvement + diversity preservation - stagnation penalty

Architecture follows Mnih et al. (2015) DQN with curriculum learning for constrained optimization.
"""

import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Experience tuple for replay buffer
Experience = namedtuple('Experience',
                        ['state', 'action', 'reward', 'next_state', 'done'])


class DQNNetwork(nn.Module):
    """
    Deep Q-Network for BCHM selection (Population-level)

    Architecture:
    - Input: 9-dimensional state vector (population-level features)
    - Hidden layers: 2 fully connected layers with ReLU (64 neurons each)
    - Output: Q-values for 5 actions (BCHMs)

    Simpler architecture than particle-level version since:
    - State space is smaller (9 features vs 20-30)
    - Action space is smaller (5 methods)
    - Population-level decisions are less frequent
    """

    def __init__(self, state_dim=9, action_dim=5, hidden_dims=[64, 64]):
        """
        Args:
            state_dim: Dimension of state vector (default: 9)
            action_dim: Number of actions - methods in pool (default: 5)
            hidden_dims: List of hidden layer dimensions
        """
        super(DQNNetwork, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Build network layers
        layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # Output layer (Q-values)
        layers.append(nn.Linear(prev_dim, action_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize network weights using He initialization"""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, state):
        """
        Forward pass

        Args:
            state: Tensor of shape (batch_size, state_dim) or (state_dim,)

        Returns:
            Q-values: Tensor of shape (batch_size, action_dim) or (action_dim,)
        """
        return self.network(state)


class BoundaryViolationBuffer:
    """
    Simple Replay Buffer - Stores ONLY boundary violation experiences

    Only experiences with prob_infeas_bounds > 0.15 are stored.
    Uses uniform random sampling (no prioritization needed).

    This focuses learning exclusively on constraint handling scenarios
    and eliminates numerical instability from irrelevant experiences.
    """

    def __init__(self, capacity=10000):
        """
        Args:
            capacity: Maximum buffer size
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.position = 0

    def push(self, state, action, reward, next_state, done, prob_infeas_bounds=0.0):
        """
        Add experience ONLY if it contains boundary violations

        Args:
            prob_infeas_bounds: Boundary violation rate (0-1)
        """
        # Store ONLY if boundary violations present
        if prob_infeas_bounds > 0.15:
            experience = Experience(state, action, reward, next_state, done)
            self.buffer.append(experience)

    def sample(self, batch_size):
        """
        Uniform random sampling from buffer

        Returns:
            experiences, indices, weights (all weights = 1.0 for uniform sampling)
        """
        if len(self.buffer) < batch_size:
            # Not enough experiences yet
            return [], [], []

        # Uniform random sampling
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        experiences = [self.buffer[idx] for idx in indices]

        # Uniform weights (no importance sampling needed)
        weights = np.ones(batch_size)

        return experiences, indices, weights

    def update_priorities(self, indices, td_errors):
        """
        No-op for uniform buffer (kept for API compatibility)

        Args:
            indices: Sampled indices
            td_errors: TD errors (ignored)
        """
        pass  # No priority updates needed for uniform sampling

    def __len__(self):
        return len(self.buffer)

    def get_stats(self):
        """
        Get buffer statistics for logging/monitoring

        Returns:
            dict with buffer statistics
        """
        return {
            'total_size': len(self.buffer),
            'boundary_size': len(self.buffer),
            'normal_size': 0,
            'boundary_pct': 100.0,
            'normal_pct': 0.0,
            'avg_boundary_priority': 1.0,
            'avg_normal_priority': 0.0,
            'capacity_boundary': self.capacity,
            'capacity_normal': 0,
            'training_mode': 'boundary_violations_only',
        }


class DQNAgent:
    """
    DQN Agent for Population-level BCHM selection

    Implements:
    - Q-learning with function approximation
    - Prioritized experience replay
    - Target network for stable learning
    - Îµ-greedy exploration
    """

    def __init__(self,
                 state_dim=9,
                 action_dim=5,
                 learning_rate=0.001,
                 gamma=0.95,
                 epsilon_start=1.0,
                 epsilon_end=0.05,
                 epsilon_decay=0.995,
                 buffer_capacity=10000,
                 batch_size=64,
                 target_update_freq=10,  # Update every N episodes
                 device='cpu'):
        """
        Args:
            state_dim: Dimension of state vector (9)
            action_dim: Number of actions - 5 BCHMs
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Decay rate for epsilon
            buffer_capacity: Size of replay buffer
            batch_size: Mini-batch size for training
            target_update_freq: Episodes between target network updates
            device: 'cpu' or 'cuda'
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = torch.device(device)

        # Q-network and target network
        self.q_network = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_network = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Target network is not trained directly

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Boundary violation buffer (stores ONLY experiences with prob_infeas_bounds > 0.15)
        self.memory = BoundaryViolationBuffer(buffer_capacity)

        # Training statistics
        self.steps = 0
        self.episodes = 0
        self.losses = []

    def select_action(self, state, training=True):
        """
        Select action using Îµ-greedy policy

        Args:
            state: Current state (numpy array, shape (9,))
            training: If True, use Îµ-greedy; if False, use greedy

        Returns:
            action: Selected action index (0-4)
        """
        if training and random.random() < self.epsilon:
            # Explore: random action
            return random.randrange(self.action_dim)
        else:
            # Exploit: action with highest Q-value
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                return q_values.argmax(dim=1).item()

    def store_experience(self, state, action, reward, next_state, done, prob_infeas_bounds=0.0):
        """
        Store experience in prioritized replay buffer

        Args:
            prob_infeas_bounds: Boundary violation rate (for priority boosting)
        """
        self.memory.push(state, action, reward, next_state, done, prob_infeas_bounds)

    def train_step(self):
        """
        Perform one training step (sample batch and update Q-network)

        Returns:
            loss: TD loss for this batch (None if not enough samples)
        """
        if len(self.memory) < self.batch_size:
            return None

        # Sample mini-batch with priorities
        experiences, indices_info, weights = self.memory.sample(self.batch_size)

        # Check if buffer has enough samples
        if len(experiences) == 0:
            return None

        batch = Experience(*zip(*experiences))

        # Convert to tensors
        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.LongTensor(batch.action).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        done_batch = torch.FloatTensor(batch.done).to(self.device)
        weight_batch = torch.FloatTensor(weights).to(self.device)

        # Compute current Q-values
        current_q_values = self.q_network(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze()

        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_state_batch).max(1)[0]
            target_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values

        # DIAGNOSTIC: Check for NaN/Inf BEFORE training
        if torch.isnan(current_q_values).any() or torch.isinf(current_q_values).any():
            print(f"\n[DIAGNOSTIC] NaN/Inf in current Q-values!")
            print(f"  Current Q: [{current_q_values.min().item():.4f}, {current_q_values.max().item():.4f}]")
            print(f"  State: [{state_batch.min().item():.4f}, {state_batch.max().item():.4f}]")
            return None

        if torch.isnan(target_q_values).any() or torch.isinf(target_q_values).any():
            print(f"\n[DIAGNOSTIC] NaN/Inf in target Q-values!")
            print(f"  Rewards: [{reward_batch.min().item():.4f}, {reward_batch.max().item():.4f}]")
            print(f"  Next Q: [{next_q_values.min().item():.4f}, {next_q_values.max().item():.4f}]")
            print(f"  Target Q: [{target_q_values.min().item():.4f}, {target_q_values.max().item():.4f}]")
            return None

        # Compute TD errors for priority update
        td_errors = (target_q_values - current_q_values).detach().cpu().numpy()
        self.memory.update_priorities(indices_info, td_errors)

        # Compute weighted loss (importance sampling)
        loss = (weight_batch * F.mse_loss(current_q_values, target_q_values, reduction='none')).mean()

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n[DIAGNOSTIC] NaN/Inf in loss!")
            print(f"  Loss: {loss.item()}")
            return None

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        # Store loss
        self.steps += 1
        self.losses.append(loss.item())

        return loss.item()

    def update_target_network(self):
        """Copy Q-network weights to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def end_episode(self):
        """Call at the end of each episode"""
        self.episodes += 1

        # Update target network periodically
        if self.episodes % self.target_update_freq == 0:
            self.update_target_network()

        # Decay epsilon
        self.decay_epsilon()

    def save(self, filepath):
        """Save agent state"""
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'episodes': self.episodes,
            'losses': self.losses,
        }
        torch.save(checkpoint, filepath)

    def load(self, filepath):
        """Load agent state"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        self.episodes = checkpoint['episodes']
        self.losses = checkpoint['losses']

    def get_stats(self):
        """Get training statistics"""
        return {
            'steps': self.steps,
            'episodes': self.episodes,
            'epsilon': self.epsilon,
            'avg_loss_last_100': np.mean(self.losses[-100:]) if len(self.losses) >= 100 else None,
            'buffer_size': len(self.memory)
        }


def extract_state(pso_state, generation, max_generations, previous_method_id=0):
    """
    Extract 9-dimensional state vector from PSO state (POPULATION-LEVEL)

    State features (conform plan):
    1. normalized_generation: generation / max_generations
    2. best_fitness_normalized: normalized best fitness
    3. fitness_improvement_rate: trend over last 5 generations
    4. stagnation_counter_normalized: generations without improvement / 50
    5. diversity_normalized: current population diversity
    6. diversity_trend: diversity change over last 5 generations
    7. prob_infeas_bounds: % particles violating bounds
    8. num_infeas_bounds_normalized: absolute count / pop_size
    9. previous_method_id: last selected method (0-4)

    Args:
        pso_state: Dict with PSO state information
        generation: Current generation
        max_generations: Maximum generations
        previous_method_id: Previously selected method (0-4)

    Returns:
        state: numpy array of shape (9,)
    """
    # 1. Normalized generation
    normalized_generation = generation / max_generations

    # 2. Best fitness normalized (assume minimization, use log scale)
    best_fitness = pso_state.get('best_fitness', 0)
    best_fitness_normalized = np.clip(np.log10(abs(best_fitness) + 1) / 10, 0, 1)

    # 3. Fitness improvement rate (from history)
    fitness_history = pso_state.get('fitness_history', [best_fitness])
    if len(fitness_history) >= 6:
        recent = fitness_history[-6:]
        fitness_improvement_rate = (recent[0] - recent[-1]) / (abs(recent[0]) + 1e-10)
    else:
        fitness_improvement_rate = 0.0
    fitness_improvement_rate = np.clip(fitness_improvement_rate, -1, 1)

    # 4. Stagnation counter normalized
    stagnation_counter = pso_state.get('stagnation_counter', 0)
    stagnation_counter_normalized = np.clip(stagnation_counter / 50, 0, 1)

    # 5. Diversity normalized
    diversity = pso_state.get('diversity', 0)
    diversity_normalized = np.clip(diversity, 0, 1)

    # 6. Diversity trend (from history)
    diversity_history = pso_state.get('diversity_history', [diversity])
    if len(diversity_history) >= 6:
        recent_div = diversity_history[-6:]
        diversity_trend = (recent_div[-1] - recent_div[0]) / (abs(recent_div[0]) + 1e-10)
    else:
        diversity_trend = 0.0
    diversity_trend = np.clip(diversity_trend, -1, 1)

    # 7. prob_infeas_bounds
    prob_infeas_bounds = pso_state.get('prob_infeas_bounds', 0)

    # 8. num_infeas_bounds normalized
    num_infeas_bounds = pso_state.get('num_infeas_bounds', 0)
    pop_size = pso_state.get('pop_size', 100)
    num_infeas_bounds_normalized = num_infeas_bounds / pop_size

    # 9. Previous method ID (normalized to [0, 1])
    previous_method_normalized = previous_method_id / 4.0  # 5 methods â†’ 0-4

    # Build state vector
    state = np.array([
        normalized_generation,
        best_fitness_normalized,
        fitness_improvement_rate,
        stagnation_counter_normalized,
        diversity_normalized,
        diversity_trend,
        prob_infeas_bounds,
        num_infeas_bounds_normalized,
        previous_method_normalized
    ], dtype=np.float32)

    return state


def compute_reward(old_state, new_state, w_fitness=1.0, w_diversity=0.2, w_stagnation=0.1):
    """
    Compute reward based on fitness improvement + diversity preservation - stagnation penalty

    Reward = w1 * fitness_improvement
           + w2 * diversity_preservation
           - w3 * stagnation_penalty

    NO feasibility term (as per plan)

    Args:
        old_state: Dict with PSO state before action
        new_state: Dict with PSO state after action
        w_fitness: Weight for fitness improvement (default: 1.0)
        w_diversity: Weight for diversity preservation (default: 0.2)
        w_stagnation: Weight for stagnation penalty (default: 0.1)

    Returns:
        reward: float, normalized reward
    """
    # 1. Fitness improvement
    old_fitness = old_state.get('best_fitness', 0)
    new_fitness = new_state.get('best_fitness', 0)

    if abs(old_fitness) > 1e-10:
        fitness_improvement = (old_fitness - new_fitness) / abs(old_fitness)
    else:
        fitness_improvement = 0.0

    fitness_improvement = np.clip(fitness_improvement, -1, 1)

    # 2. Diversity preservation
    old_diversity = old_state.get('diversity', 0)
    new_diversity = new_state.get('diversity', 0)

    diversity_preservation = new_diversity - old_diversity  # Positive if diversity increased
    diversity_preservation = np.clip(diversity_preservation, -1, 1)

    # 3. Stagnation penalty
    stagnation_counter = new_state.get('stagnation_counter', 0)
    stagnation_threshold = 25

    if stagnation_counter > stagnation_threshold:
        stagnation_penalty = 1.0
    else:
        stagnation_penalty = 0.0

    # Combined reward
    reward = (w_fitness * fitness_improvement +
              w_diversity * diversity_preservation -
              w_stagnation * stagnation_penalty)

    return reward
