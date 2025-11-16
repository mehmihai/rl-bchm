"""
Generation-level tracking for PSO optimization
Collects comprehensive metrics for analysis and RL state space
"""

import json
from pathlib import Path
from typing import Dict, List

import numpy as np


class GenerationMetrics:
    """
    Comprehensive metrics collected at each generation
    Used for: statistical analysis, RL state space, visualization
    """

    def __init__(self, pso_instance, problem_instance):
        """
        Compute metrics from current PSO state

        Args:
            pso_instance: PSO object with current state
            problem_instance: CEC2006Problem with known optimum
        """
        self.generation = pso_instance.generation
        self.nfe = pso_instance.nfe

        # Global best metrics
        self.gbest_position = pso_instance.gbest_position.copy()
        self.gbest_fitness = pso_instance.gbest_fitness
        self.gbest_cv = pso_instance.gbest_cv
        self.gbest_is_feasible = pso_instance.gbest_is_feasible

        # Error from known optimum
        self.error = abs(self.gbest_fitness - problem_instance.known_optimum)

        # Population statistics
        positions = np.array([p.position for p in pso_instance.swarm])
        fitness_values = np.array([p.fitness for p in pso_instance.swarm])
        cv_values = np.array([p.cv for p in pso_instance.swarm])
        velocities = np.array([p.velocity for p in pso_instance.swarm])

        self.mean_fitness = float(np.mean(fitness_values))
        self.std_fitness = float(np.std(fitness_values))
        self.min_fitness = float(np.min(fitness_values))
        self.max_fitness = float(np.max(fitness_values))

        self.mean_cv = float(np.mean(cv_values))
        self.std_cv = float(np.std(cv_values))
        self.max_cv = float(np.max(cv_values))

        # Diversity: average Euclidean distance from centroid
        centroid = np.mean(positions, axis=0)
        distances = np.linalg.norm(positions - centroid, axis=1)
        self.diversity = float(np.mean(distances))
        self.max_distance = float(np.max(distances))

        # Velocity statistics
        velocity_magnitudes = np.linalg.norm(velocities, axis=1)
        self.mean_velocity_magnitude = float(np.mean(velocity_magnitudes))
        self.max_velocity_magnitude = float(np.max(velocity_magnitudes))

        # Feasibility statistics
        self.nfs = pso_instance.nfs  # Number of feasible solutions (pbest-based)
        self.feasibility_rate = self.nfs / len(pso_instance.swarm)

        # Boundary violation probability (this generation)
        if pso_instance.gen_total_components > 0:
            self.prob_infeas = pso_instance.gen_violated_components / pso_instance.gen_total_components
        else:
            self.prob_infeas = 0.0

        # Repair statistics (if available)
        self.repairs_made = sum(1 for p in pso_instance.swarm if p.was_repaired)
        self.repair_rate = self.repairs_made / len(pso_instance.swarm)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'generation': int(self.generation),
            'nfe': int(self.nfe),

            # Global best
            'gbest_fitness': float(self.gbest_fitness),
            'gbest_cv': float(self.gbest_cv),
            'gbest_is_feasible': bool(self.gbest_is_feasible),
            'error': float(self.error),

            # Population statistics
            'mean_fitness': float(self.mean_fitness),
            'std_fitness': float(self.std_fitness),
            'min_fitness': float(self.min_fitness),
            'max_fitness': float(self.max_fitness),

            'mean_cv': float(self.mean_cv),
            'std_cv': float(self.std_cv),
            'max_cv': float(self.max_cv),

            # Diversity
            'diversity': float(self.diversity),
            'max_distance': float(self.max_distance),

            # Velocity
            'mean_velocity_magnitude': float(self.mean_velocity_magnitude),
            'max_velocity_magnitude': float(self.max_velocity_magnitude),

            # Feasibility
            'nfs': int(self.nfs),
            'feasibility_rate': float(self.feasibility_rate),

            # Boundary violations
            'prob_infeas': float(self.prob_infeas),
            'repairs_made': int(self.repairs_made),
            'repair_rate': float(self.repair_rate)
        }


class RunTracker:
    """
    Tracks all generations for a single optimization run
    """

    def __init__(self, run_number: int, problem_id: int, method_name: str):
        """
        Initialize run tracker

        Args:
            run_number: Run ID (1, 2, 3, ...)
            problem_id: CEC2006 problem ID (1-24)
            method_name: Method/algorithm name
        """
        self.run_number = run_number
        self.problem_id = problem_id
        self.method_name = method_name

        # Generation history
        self.generations: List[GenerationMetrics] = []

        # Convergence tracking
        self.last_improvement_generation = 0
        self.best_fitness_history = []

    def log_generation(self, pso_instance, problem_instance):
        """
        Log metrics for current generation

        Args:
            pso_instance: PSO with current state
            problem_instance: CEC2006Problem
        """
        metrics = GenerationMetrics(pso_instance, problem_instance)
        self.generations.append(metrics)

        # Track improvements
        if len(self.best_fitness_history) == 0 or \
                metrics.gbest_fitness < self.best_fitness_history[-1]:
            self.last_improvement_generation = metrics.generation

        self.best_fitness_history.append(metrics.gbest_fitness)

    def get_generations_without_improvement(self) -> int:
        """Get number of generations since last gbest improvement"""
        if len(self.generations) == 0:
            return 0
        current_gen = self.generations[-1].generation
        return current_gen - self.last_improvement_generation

    def to_dict(self) -> Dict:
        """Convert run data to dictionary"""
        return {
            'run_number': int(self.run_number),
            'problem_id': int(self.problem_id),
            'method_name': str(self.method_name),

            # Final results
            'final_generation': int(self.generations[-1].generation) if self.generations else 0,
            'final_nfe': int(self.generations[-1].nfe) if self.generations else 0,
            'final_fitness': float(self.generations[-1].gbest_fitness) if self.generations else float('inf'),
            'final_error': float(self.generations[-1].error) if self.generations else float('inf'),
            'final_cv': float(self.generations[-1].gbest_cv) if self.generations else float('inf'),
            'is_feasible': bool(self.generations[-1].gbest_is_feasible) if self.generations else False,

            # Convergence
            'generations_without_improvement': int(self.get_generations_without_improvement()),

            # Generation-by-generation history
            'generation_history': [g.to_dict() for g in self.generations]
        }

    def save_json(self, output_path: Path):
        """
        Save run data to JSON file

        Args:
            output_path: Path to output JSON file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Run {self.run_number} saved to {output_path}")


class ExperimentTracker:
    """
    Tracks multiple runs for an experiment (method × problem)
    """

    def __init__(self, method_name: str, problem_id: int):
        """
        Initialize experiment tracker

        Args:
            method_name: Method/algorithm name
            problem_id: CEC2006 problem ID
        """
        self.method_name = method_name
        self.problem_id = problem_id
        self.runs: List[RunTracker] = []

    def create_run(self, run_number: int) -> RunTracker:
        """
        Create new run tracker

        Args:
            run_number: Run ID (1, 2, 3, ...)

        Returns:
            RunTracker instance
        """
        run_tracker = RunTracker(run_number, self.problem_id, self.method_name)
        self.runs.append(run_tracker)
        return run_tracker

    def get_statistics(self) -> Dict:
        """
        Compute aggregate statistics across all runs

        Returns:
            Dictionary with mean, std, min, max, success rate
        """
        if not self.runs:
            return {}

        final_fitness = [r.generations[-1].gbest_fitness for r in self.runs if r.generations]
        final_errors = [r.generations[-1].error for r in self.runs if r.generations]
        final_cvs = [r.generations[-1].gbest_cv for r in self.runs if r.generations]
        is_feasible = [r.generations[-1].gbest_is_feasible for r in self.runs if r.generations]

        return {
            'method_name': self.method_name,
            'problem_id': self.problem_id,
            'num_runs': len(self.runs),

            # Fitness statistics
            'mean_fitness': float(np.mean(final_fitness)),
            'std_fitness': float(np.std(final_fitness)),
            'min_fitness': float(np.min(final_fitness)),
            'max_fitness': float(np.max(final_fitness)),

            # Error statistics
            'mean_error': float(np.mean(final_errors)),
            'std_error': float(np.std(final_errors)),
            'min_error': float(np.min(final_errors)),
            'max_error': float(np.max(final_errors)),

            # Constraint violation statistics
            'mean_cv': float(np.mean(final_cvs)),
            'std_cv': float(np.std(final_cvs)),
            'max_cv': float(np.max(final_cvs)),

            # Success rate
            'success_rate': float(np.mean(is_feasible)),
            'feasible_runs': int(np.sum(is_feasible)),
        }

    def save_summary(self, output_path: Path):
        """
        Save experiment summary (all runs + statistics)

        Args:
            output_path: Path to output JSON file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'experiment': {
                'method_name': self.method_name,
                'problem_id': self.problem_id,
                'num_runs': len(self.runs)
            },
            'statistics': self.get_statistics(),
            'runs': [r.to_dict() for r in self.runs]
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Experiment summary saved to {output_path}")
        print(f"  Method: {self.method_name}")
        print(f"  Problem: G{self.problem_id:02d}")
        print(f"  Runs: {len(self.runs)}")
        print(f"  Mean error: {self.get_statistics()['mean_error']:.6e}")
        print(f"  Success rate: {self.get_statistics()['success_rate']:.2%}")
