"""
RL-PSO with DQN for Population-Level BCHM Selection

DQN makes ONE decision per generation for the ENTIRE population (not per particle).
This simplifies the problem and ensures consistent training signals.
"""

import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.pso import PSO
from src.core.velocity_strategy import VelocityUpdater
from src.core.constraint_handler import DebsRules
from src.rl.dqn import DQNAgent, extract_state, compute_reward

# Method pool: custom5 (same as ABCHS-custom5)
# Method IDs from boundary_handler.py
METHOD_POOL_CUSTOM5 = [
    (3, "DeB"),  # Action 0: Ran&DeB (METHOD_UNIF = 3)
    (8, "RaB"),  # Action 1: ExpT&RaB (METHOD_EXPC_TARGET = 8)
    (9, "DeB"),  # Action 2: ExpB&DeB (METHOD_EXPC_BEST = 9)
    (2, "DeB"),  # Action 3: MidB&DeB (METHOD_MIDPOINT_BEST = 2)
    (0, "RaB")  # Action 4: Bou&RaB (METHOD_SATURATION = 0)
]


class RL_PSO(PSO):
    """
    RL-enhanced PSO with DQN for population-level BCHM selection.

    Key differences from particle-level:
    - DQN selects ONE method per generation (applies to ALL particles)
    - State: population-level aggregated features (9 dims)
    - Reward: based on population-level improvement
    - Simpler, more stable training
    """

    def __init__(self,
                 problem,
                 population_size=100,
                 constriction_k=0.729,
                 c1=1.49445,
                 c2=1.49445,
                 max_evaluations=500000,
                 random_seed=None,
                 dqn_agent=None,
                 training_mode=True,
                 method_pool=None):
        """
        Args:
            problem: Optimization problem
            population_size: Number of particles
            constriction_k: Constriction coefficient
            c1, c2: Acceleration coefficients
            max_evaluations: Budget of function evaluations
            random_seed: Random seed for reproducibility
            dqn_agent: Pre-initialized DQN agent (if None, creates new one)
            training_mode: If True, agent learns; if False, agent only exploits
            method_pool: List of (position_method, velocity_strategy) tuples
                        If None, uses custom5 pool
        """
        # Initialize base PSO (with dummy methods, will be overridden)
        super().__init__(
            problem=problem,
            population_size=population_size,
            constriction_k=constriction_k,
            c1=c1,
            c2=c2,
            position_method="boundary",  # Dummy, will be overridden
            velocity_strategy="deb",  # Dummy, will be overridden
            max_evaluations=max_evaluations,
            random_seed=random_seed
        )

        # Method pool
        if method_pool is None:
            self.method_pool = METHOD_POOL_CUSTOM5
        else:
            self.method_pool = method_pool

        self.num_actions = len(self.method_pool)

        # DQN agent
        state_dim = 9  # Population-level state
        if dqn_agent is None:
            self.dqn_agent = DQNAgent(
                state_dim=state_dim,
                action_dim=self.num_actions,
                learning_rate=0.001,
                gamma=0.95,
                epsilon_start=1.0 if training_mode else 0.0,
                epsilon_end=0.05,
                epsilon_decay=0.995,
                buffer_capacity=10000,
                batch_size=64,
                target_update_freq=10,  # episodes
                device='cpu'
            )
        else:
            self.dqn_agent = dqn_agent

        self.training_mode = training_mode

        # Tracking
        self.action_counts = np.zeros(self.num_actions, dtype=int)
        self.previous_method_id = 0

        # State tracking (for reward calculation)
        self.fitness_history = []
        self.diversity_history = []
        self.stagnation_counter = 0
        self.prev_best_fitness = float('inf')

    def step(self):
        """
        Override PSO step to integrate DQN-based BCHM selection.

        Key difference: DQN selects ONE method at START of generation,
        applies it to ALL particles that violate bounds.
        """
        # Increment generation counter
        self.generation += 1

        # Compute max generations
        max_generations = self.max_evaluations / len(self.swarm)

        # === POPULATION-LEVEL STATE (BEFORE GENERATION) ===
        old_pso_state = self._get_pso_state()
        old_state = extract_state(old_pso_state, self.generation, max_generations, self.previous_method_id)

        # === DQN SELECTS METHOD FOR THIS GENERATION ===
        action = self.dqn_agent.select_action(old_state, training=self.training_mode)
        self.action_counts[action] += 1
        self.previous_method_id = action

        # Get selected methods
        position_method, velocity_strategy = self.method_pool[action]

        # Reset generation tracking for boundary violations
        self.gen_violated_components = 0
        self.gen_total_components = 0

        # === PER-PARTICLE ITERATION ===
        for i, particle in enumerate(self.swarm):
            # Store old position
            old_position = particle.position.copy()

            # STEP 1: Update velocity
            r1 = np.random.rand(self.problem.dimension)
            r2 = np.random.rand(self.problem.dimension)

            cognitive = self.c1 * r1 * (particle.pbest_position - particle.position)
            social = self.c2 * r2 * (self.gbest_position - particle.position)

            particle.velocity = self.k * (particle.velocity + cognitive + social)

            # STEP 2: Update position
            particle.position = particle.position + particle.velocity

            # STEP 3: Check boundary violations
            has_violations, violations = self.boundary_handler.check_bounds(particle.position)

            # Track violations for state features
            self.gen_total_components += self.problem.dimension
            if has_violations:
                num_violated = np.sum(violations != 0)
                self.gen_violated_components += num_violated

            particle.was_repaired = False

            # STEP 4: If violated, apply SELECTED method
            if has_violations:
                particle.was_repaired = True

                # Apply position repair
                particle.position = self.boundary_handler.apply_method(
                    particle.position,
                    position_method,
                    target=old_position,
                    best=self.gbest_position,
                    pbest_population=self._get_pbest_population()
                )

                # Apply velocity update
                particle.velocity = VelocityUpdater.update_velocity(
                    particle.velocity,
                    violations,
                    velocity_strategy
                )

            # STEP 5: Evaluate particle
            self._evaluate_particle(particle)

            # STEP 6: Update personal best
            self._update_personal_best(particle)

            # STEP 7: Update global best (asynchronous)
            if DebsRules.is_better(
                    particle.pbest_fitness, particle.pbest_cv,
                    self.gbest_fitness, self.gbest_cv,
                    minimize=True
            ):
                self.gbest_position = particle.pbest_position.copy()
                self.gbest_fitness = particle.pbest_fitness
                self.gbest_cv = particle.pbest_cv
                self.gbest_is_feasible = particle.pbest_cv <= 0

        # Update NFS
        self._compute_nfs()

        # === POPULATION-LEVEL STATE (AFTER GENERATION) ===
        new_pso_state = self._get_pso_state()
        new_state = extract_state(new_pso_state, self.generation, max_generations, self.previous_method_id)

        # === COMPUTE REWARD ===
        reward = compute_reward(old_pso_state, new_pso_state)

        # === STORE EXPERIENCE AND TRAIN (if training mode) ===
        done = self.nfe >= self.max_evaluations
        prob_infeas_bounds = new_pso_state.get('prob_infeas_bounds', 0)

        if self.training_mode:
            self.dqn_agent.store_experience(
                old_state, action, reward, new_state, done, prob_infeas_bounds
            )

            # Train DQN
            loss = self.dqn_agent.train_step()

        # Update stagnation counter
        if abs(self.gbest_fitness - self.prev_best_fitness) < 1e-10:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
        self.prev_best_fitness = self.gbest_fitness

        # Update history
        self.fitness_history.append(self.gbest_fitness)
        self.diversity_history.append(self._compute_diversity())

    def _get_pso_state(self):
        """
        Get population-level PSO state for DQN

        Returns:
            dict with state information
        """
        # Compute boundary violations
        prob_infeas_bounds = self.gen_violated_components / max(self.gen_total_components, 1)
        num_infeas_bounds = sum(1 for p in self.swarm
                                if not self.boundary_handler.check_bounds(p.position)[0])

        return {
            'best_fitness': self.gbest_fitness,
            'fitness_history': self.fitness_history,
            'diversity': self._compute_diversity(),
            'diversity_history': self.diversity_history,
            'stagnation_counter': self.stagnation_counter,
            'prob_infeas_bounds': prob_infeas_bounds,
            'num_infeas_bounds': num_infeas_bounds,
            'pop_size': len(self.swarm)
        }

    def _compute_diversity(self):
        """Compute population diversity (average distance from centroid)"""
        positions = np.array([p.position for p in self.swarm])
        centroid = np.mean(positions, axis=0)
        distances = np.linalg.norm(positions - centroid, axis=1)

        # Normalize by problem scale
        problem_scale = np.max(self.problem.upper_bounds - self.problem.lower_bounds)
        dimension_scale = np.sqrt(self.problem.dimension)

        return np.mean(distances) / (problem_scale * dimension_scale + 1e-10)

    def is_done(self):
        """Check if optimization is complete"""
        return self.nfe >= self.max_evaluations

    def end_episode(self):
        """Call at the end of each episode (problem run)"""
        if self.training_mode:
            self.dqn_agent.end_episode()

    def get_stats(self):
        """Get optimizer statistics"""
        dqn_stats = self.dqn_agent.get_stats()

        return {
            'generation': self.generation,
            'nfe': self.nfe,
            'gbest_fitness': self.gbest_fitness,
            'gbest_is_feasible': self.gbest_is_feasible,
            'diversity': self._compute_diversity(),
            'stagnation_counter': self.stagnation_counter,
            'action_counts': self.action_counts.tolist(),
            'action_frequencies': (self.action_counts / max(1, np.sum(self.action_counts))).tolist(),
            'dqn_epsilon': dqn_stats['epsilon'],
            'dqn_buffer_size': dqn_stats['buffer_size'],
            'dqn_steps': dqn_stats['steps'],
            'avg_loss': dqn_stats['avg_loss_last_100'],
        }

    def save_agent(self, filepath):
        """Save DQN agent to file"""
        self.dqn_agent.save(filepath)

    def load_agent(self, filepath):
        """Load DQN agent from file"""
        self.dqn_agent.load(filepath)
