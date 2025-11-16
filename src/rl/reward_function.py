"""
Reward Function for RL-BCHM
Simple reward balancing convergence, feasibility, and minimal diversity maintenance
"""

from typing import Tuple

import numpy as np


class RewardFunction:
    """
    Reward function for RL-based BCHM selection

    Components:
    1. Error minimization (weight: -1.0) - primary objective
    2. Feasibility bonus (weight: 0.5) - adaptive based on constraint difficulty
    3. Improvement bonus (weight: 0.3) - discrete reward for progress
    4. Diversity maintenance (weight: 0.1) - minimal, prevents premature convergence
    """

    def __init__(self, problem):
        """
        Initialize reward function

        Args:
            problem: CEC2006Problem instance
        """
        self.problem = problem
        self.known_optimum = problem.known_optimum

        # Compute search space diagonal for diversity normalization
        self.search_space_diagonal = np.linalg.norm(
            problem.upper_bounds - problem.lower_bounds
        )

    def compute_reward(self,
                       current_metrics,
                       previous_metrics,
                       swarm) -> Tuple[float, dict]:
        """
        Compute reward from generation metrics

        Args:
            current_metrics: GenerationMetrics from current generation
            previous_metrics: GenerationMetrics from previous generation
            swarm: List of Particle objects (for prob_infeas_functional)

        Returns:
            (reward, reward_components): reward value and breakdown dict
        """
        # ====================================================================
        # 1. ERROR IMPROVEMENT (Convergence - primary objective)
        # ====================================================================
        error_improvement = previous_metrics.error - current_metrics.error
        error_normalized = current_metrics.error / abs(self.known_optimum + 1e-10)

        # ====================================================================
        # 2. FEASIBILITY BONUS (Adaptive based on constraint difficulty)
        # ====================================================================
        # Calculate prob_infeas_functional (functional constraint violation rate)
        # High value = hard to find feasible space → higher reward
        # Low value = easy to find feasible space → lower reward
        cv_violations = sum(1 for p in swarm if p.cv > 0)
        prob_infeas_functional = cv_violations / len(swarm)

        if current_metrics.gbest_is_feasible:
            # Feasible solution: reward scaled by constraint difficulty
            feasibility_bonus = 1.0 + prob_infeas_functional  # Range: [1.0, 2.0]
        else:
            # Infeasible solution: penalty proportional to CV
            feasibility_bonus = -current_metrics.gbest_cv

        # ====================================================================
        # 3. IMPROVEMENT BONUS (Discrete reward for finding better gbest)
        # ====================================================================
        improvement_bonus = 1.0 if error_improvement > 0 else 0.0

        # ====================================================================
        # 4. DIVERSITY MAINTENANCE (Minimal weight, prevents premature convergence)
        # ====================================================================
        diversity_normalized = current_metrics.diversity / self.search_space_diagonal

        # ====================================================================
        # COMBINE ALL COMPONENTS
        # ====================================================================
        reward = (
                -1.0 * error_normalized +  # Convergence (weight: -1.0)
                0.5 * feasibility_bonus +  # Feasibility (weight: 0.5)
                0.3 * improvement_bonus +  # Progress (weight: 0.3)
                0.1 * diversity_normalized  # Diversity (weight: 0.1)
        )

        # Return reward and breakdown for logging/debugging
        reward_components = {
            'reward_total': float(reward),
            'error_normalized': float(error_normalized),
            'error_improvement': float(error_improvement),
            'feasibility_bonus': float(feasibility_bonus),
            'improvement_bonus': float(improvement_bonus),
            'diversity_normalized': float(diversity_normalized),
            'prob_infeas_functional': float(prob_infeas_functional),
            'term_error': float(-1.0 * error_normalized),
            'term_feasibility': float(0.5 * feasibility_bonus),
            'term_improvement': float(0.3 * improvement_bonus),
            'term_diversity': float(0.1 * diversity_normalized)
        }

        return reward, reward_components


class StateSpaceExtractor:
    """
    Extract RL state features from PSO/ABCHS instance
    """

    def __init__(self, problem):
        """
        Initialize state space extractor

        Args:
            problem: CEC2006Problem instance
        """
        self.problem = problem

        # Compute search space diagonal for normalization
        self.search_space_diagonal = np.linalg.norm(
            problem.upper_bounds - problem.lower_bounds
        )

        # Max generations estimate (for normalization)
        # Typical: 500000 evals / 100 pop_size = 5000 generations
        self.max_generations_estimate = 5000

    def extract_state(self, pso_instance) -> np.ndarray:
        """
        Extract RL state vector from PSO/ABCHS instance

        State components (13 features):
        1. error_normalized: Distance from known optimum
        2. gbest_cv: Constraint violation at gbest
        3. feasibility_rate: NFS / population_size
        4. prob_infeas: Boundary violation probability (this generation)
        5. prob_infeas_functional: Functional constraint violation rate
        6. generation_progress: Normalized generation counter
        7. stagnation: Generations without improvement
        8. diversity: Normalized population diversity
        9. max_distance: Normalized max distance from centroid
        10. mean_velocity: Normalized mean velocity magnitude
        11. fitness_std: Fitness standard deviation (normalized)
        12. cv_std: CV standard deviation (normalized)
        13. stage: Current stage (1 or 2) for ABCHS, 0 for PSO

        Args:
            pso_instance: PSO or ABCHS instance with current state

        Returns:
            state: np.ndarray of shape (13,)
        """
        from src.utils.generation_tracker import GenerationMetrics

        # Get current generation metrics
        metrics = GenerationMetrics(pso_instance, self.problem)

        # Calculate prob_infeas_functional
        cv_violations = sum(1 for p in pso_instance.swarm if p.cv > 0)
        prob_infeas_functional = cv_violations / len(pso_instance.swarm)

        # Calculate generations without improvement
        # (This requires tracking in PSO - simplified version)
        if hasattr(pso_instance, 'last_improvement_generation'):
            gens_without_improvement = (
                    pso_instance.generation - pso_instance.last_improvement_generation
            )
        else:
            gens_without_improvement = 0

        # Normalize stagnation (cap at 100 generations)
        stagnation_normalized = min(gens_without_improvement / 100.0, 1.0)

        # Stage indicator (ABCHS-specific, 0 for base PSO)
        stage = float(pso_instance.stage) if hasattr(pso_instance, 'stage') else 0.0

        # Build state vector
        state = np.array([
            # Convergence quality
            metrics.error / abs(self.problem.known_optimum + 1e-10),  # [0] error_normalized

            # Constraint satisfaction
            metrics.gbest_cv,  # [1] gbest_cv
            metrics.feasibility_rate,  # [2] feasibility_rate (NFS/pop)

            # Violation metrics
            metrics.prob_infeas,  # [3] boundary violations
            prob_infeas_functional,  # [4] functional violations

            # Progress indicators
            pso_instance.generation / self.max_generations_estimate,  # [5] generation_progress
            stagnation_normalized,  # [6] stagnation

            # Diversity metrics
            metrics.diversity / self.search_space_diagonal,  # [7] diversity
            metrics.max_distance / self.search_space_diagonal,  # [8] max_distance
            metrics.mean_velocity_magnitude / self.search_space_diagonal,  # [9] mean_velocity

            # Population statistics
            metrics.std_fitness / (abs(metrics.mean_fitness) + 1e-10),  # [10] fitness_std
            metrics.std_cv / (metrics.mean_cv + 1e-10) if metrics.mean_cv > 0 else 0.0,  # [11] cv_std

            # Stage indicator
            stage,  # [12] stage (ABCHS: 1 or 2, PSO: 0)
        ], dtype=np.float32)

        return state

    @staticmethod
    def get_state_dim() -> int:
        """Return state space dimensionality"""
        return 13

    @staticmethod
    def get_feature_names() -> list:
        """Return feature names for logging/visualization"""
        return [
            'error_normalized',
            'gbest_cv',
            'feasibility_rate',
            'prob_infeas_boundary',
            'prob_infeas_functional',
            'generation_progress',
            'stagnation',
            'diversity',
            'max_distance',
            'mean_velocity',
            'fitness_std',
            'cv_std',
            'stage'
        ]
