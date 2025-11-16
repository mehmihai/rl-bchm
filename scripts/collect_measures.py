"""
Collect comprehensive metrics for all methods (ABCHS + fixed BCHMs)
For feature importance and correlation analysis
"""

import json
from pathlib import Path

import numpy as np

from src.benchmarks.cec2006 import get_problem
from src.core.abchs import ABCHS
from src.core.boundary_handler import (
    METHOD_SATURATION, METHOD_MIDPOINT_TARGET, METHOD_MIDPOINT_BEST,
    METHOD_UNIF, METHOD_MIRROR, METHOD_TOROIDAL,
    METHOD_EXPC_TARGET, METHOD_EXPC_BEST,
    METHOD_VECTOR_TARGET, METHOD_VECTOR_BEST,
    METHOD_DISMISS, CUSTOM_CENTROID, CUSTOM_EVOLUTIONARY
)
from src.core.pso import PSO
from src.core.velocity_strategy import VELOCITY_DEB, VELOCITY_RAB
from src.utils.generation_tracker import GenerationMetrics

# Define all methods: 13 position × 2 velocity = 26 fixed + 1 ABCHS = 27 total
ALL_METHODS = [
    # ABCHS (adaptive)
    {"name": "ABCHS", "type": "adaptive"},

    # Fixed methods with DeB
    {"name": "Bou&DeB", "position": METHOD_SATURATION, "velocity": VELOCITY_DEB},
    {"name": "MidT&DeB", "position": METHOD_MIDPOINT_TARGET, "velocity": VELOCITY_DEB},
    {"name": "MidB&DeB", "position": METHOD_MIDPOINT_BEST, "velocity": VELOCITY_DEB},
    {"name": "Ran&DeB", "position": METHOD_UNIF, "velocity": VELOCITY_DEB},
    {"name": "Ref&DeB", "position": METHOD_MIRROR, "velocity": VELOCITY_DEB},
    {"name": "Wra&DeB", "position": METHOD_TOROIDAL, "velocity": VELOCITY_DEB},
    {"name": "ExpT&DeB", "position": METHOD_EXPC_TARGET, "velocity": VELOCITY_DEB},
    {"name": "ExpB&DeB", "position": METHOD_EXPC_BEST, "velocity": VELOCITY_DEB},
    {"name": "VecT&DeB", "position": METHOD_VECTOR_TARGET, "velocity": VELOCITY_DEB},
    {"name": "VecB&DeB", "position": METHOD_VECTOR_BEST, "velocity": VELOCITY_DEB},
    {"name": "Dis&DeB", "position": METHOD_DISMISS, "velocity": VELOCITY_DEB},
    {"name": "Cen&DeB", "position": CUSTOM_CENTROID, "velocity": VELOCITY_DEB},
    {"name": "Evo&DeB", "position": CUSTOM_EVOLUTIONARY, "velocity": VELOCITY_DEB},

    # Fixed methods with RaB
    {"name": "Bou&RaB", "position": METHOD_SATURATION, "velocity": VELOCITY_RAB},
    {"name": "MidT&RaB", "position": METHOD_MIDPOINT_TARGET, "velocity": VELOCITY_RAB},
    {"name": "MidB&RaB", "position": METHOD_MIDPOINT_BEST, "velocity": VELOCITY_RAB},
    {"name": "Ran&RaB", "position": METHOD_UNIF, "velocity": VELOCITY_RAB},
    {"name": "Ref&RaB", "position": METHOD_MIRROR, "velocity": VELOCITY_RAB},
    {"name": "Wra&RaB", "position": METHOD_TOROIDAL, "velocity": VELOCITY_RAB},
    {"name": "ExpT&RaB", "position": METHOD_EXPC_TARGET, "velocity": VELOCITY_RAB},
    {"name": "ExpB&RaB", "position": METHOD_EXPC_BEST, "velocity": VELOCITY_RAB},
    {"name": "VecT&RaB", "position": METHOD_VECTOR_TARGET, "velocity": VELOCITY_RAB},
    {"name": "VecB&RaB", "position": METHOD_VECTOR_BEST, "velocity": VELOCITY_RAB},
    {"name": "Dis&RaB", "position": METHOD_DISMISS, "velocity": VELOCITY_RAB},
    {"name": "Cen&RaB", "position": CUSTOM_CENTROID, "velocity": VELOCITY_RAB},
    {"name": "Evo&RaB", "position": CUSTOM_EVOLUTIONARY, "velocity": VELOCITY_RAB},
]


class MeasuresCollector:
    """Collect comprehensive measures for analysis"""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_single_method(self,
                          method_config: dict,
                          problem_id: int,
                          run_id: int,
                          population_size: int = 100,
                          max_evaluations: int = 500000):
        """
        Run single method and collect essential generation-by-generation metrics

        Returns:
            dict with metrics history
        """
        problem = get_problem(problem_id)
        random_seed = run_id * 1000 + problem_id

        # Create algorithm
        if method_config.get("type") == "adaptive":
            algorithm = ABCHS(
                problem=problem,
                population_size=population_size,
                max_evaluations=max_evaluations,
                random_seed=random_seed
            )
        else:
            algorithm = PSO(
                problem=problem,
                population_size=population_size,
                max_evaluations=max_evaluations,
                position_method=method_config["position"],
                velocity_strategy=method_config["velocity"],
                random_seed=random_seed
            )

        # Storage for metrics
        generation_data = []

        # Track improvement
        last_improvement_generation = 0
        prev_gbest_fitness = np.inf
        prev_gbest_error = np.inf

        # Track first feasible generation
        first_feasible_generation = None

        # Initial state
        metrics = GenerationMetrics(algorithm, problem)

        # Compute population errors
        pop_errors = [abs(p.fitness - problem.known_optimum) for p in algorithm.swarm]

        # Track first feasible if applicable
        if first_feasible_generation is None and metrics.nfs > 0:
            first_feasible_generation = 0

        # Compute new measures
        gbest_to_optimum_distance = float(np.linalg.norm(algorithm.gbest_position - problem.optimum_position))
        particles_improved_rate = float(algorithm.gen_particles_improved) / algorithm.population_size

        initial_data = {
            "generation": 0,
            "nfe": algorithm.nfe,
            "gbest_error": float(metrics.error),
            "gbest_cv": float(metrics.gbest_cv),
            "prob_infeas_bounds": float(metrics.prob_infeas),
            "diversity": float(metrics.diversity),
            "mean_error": float(np.mean(pop_errors)),
            "std_error": float(np.std(pop_errors)),
            "nfs": int(metrics.nfs),
            "feasibility_rate": float(metrics.feasibility_rate),
            "improvement": False,
            "gens_since_improvement": 0,
            "updates_this_gen": 0,
            "first_feasible_generation": first_feasible_generation,
            "improvement_rate": 0.0,
            "particles_improved_rate": particles_improved_rate,
            "gbest_to_optimum_distance": gbest_to_optimum_distance,
        }

        # ABCHS-specific measures
        if hasattr(algorithm, 'stage'):
            initial_data["selected_method_id"] = algorithm.last_selected_method_id
            initial_data["method_probabilities"] = algorithm.probabilities.tolist()
            initial_data["cumulative_rsB"] = algorithm.rsB.tolist()
            initial_data["cumulative_rsW"] = algorithm.rsW.tolist()

        generation_data.append(initial_data)

        # Run optimization
        while algorithm.nfe < algorithm.max_evaluations:
            # Track gbest before step
            prev_gbest_fitness = algorithm.gbest_fitness
            prev_gbest_error = abs(prev_gbest_fitness - problem.known_optimum)

            algorithm.step()

            # Collect metrics
            metrics = GenerationMetrics(algorithm, problem)

            # Compute population errors
            pop_errors = [abs(p.fitness - problem.known_optimum) for p in algorithm.swarm]

            # Check improvement
            improvement = algorithm.gbest_fitness < prev_gbest_fitness
            if improvement:
                last_improvement_generation = algorithm.generation

            gens_since_improvement = algorithm.generation - last_improvement_generation

            # Track first feasible if applicable
            if first_feasible_generation is None and metrics.nfs > 0:
                first_feasible_generation = algorithm.generation

            # Compute improvement rate
            current_error = float(metrics.error)
            if prev_gbest_error > 0:
                improvement_rate = (prev_gbest_error - current_error) / prev_gbest_error
            else:
                improvement_rate = 0.0

            # Compute new measures
            gbest_to_optimum_distance = float(np.linalg.norm(algorithm.gbest_position - problem.optimum_position))
            particles_improved_rate = float(algorithm.gen_particles_improved) / algorithm.population_size

            # Get number of times gbest was updated this generation
            # For asynchronous PSO, gbest can update multiple times per generation
            updates_this_gen = algorithm.gen_gbest_updates

            gen_data = {
                "generation": int(algorithm.generation),
                "nfe": int(algorithm.nfe),
                "gbest_error": current_error,
                "gbest_cv": float(metrics.gbest_cv),
                "prob_infeas_bounds": float(metrics.prob_infeas),
                "diversity": float(metrics.diversity),
                "mean_error": float(np.mean(pop_errors)),
                "std_error": float(np.std(pop_errors)),
                "nfs": int(metrics.nfs),
                "feasibility_rate": float(metrics.feasibility_rate),
                "improvement": bool(improvement),
                "gens_since_improvement": int(gens_since_improvement),
                "updates_this_gen": int(updates_this_gen),
                "first_feasible_generation": first_feasible_generation,
                "improvement_rate": float(improvement_rate),
                "particles_improved_rate": particles_improved_rate,
                "gbest_to_optimum_distance": gbest_to_optimum_distance,
            }

            # ABCHS-specific measures
            if hasattr(algorithm, 'stage'):
                gen_data["stage"] = int(algorithm.stage)
                gen_data["selected_method_id"] = algorithm.last_selected_method_id
                gen_data["method_probabilities"] = algorithm.probabilities.tolist()
                gen_data["cumulative_rsB"] = algorithm.rsB.tolist()
                gen_data["cumulative_rsW"] = algorithm.rsW.tolist()

            generation_data.append(gen_data)

        # Final results
        final_metrics = generation_data[-1]

        result = {
            "method_name": method_config["name"],
            "problem_id": problem_id,
            "run_id": run_id,
            "config": {
                "population_size": population_size,
                "max_evaluations": max_evaluations,
                "random_seed": random_seed,
            },

            # Final performance
            "final_fitness": float(algorithm.gbest_fitness),
            "final_error": final_metrics["gbest_error"],
            "final_cv": final_metrics["gbest_cv"],
            "is_feasible": final_metrics["gbest_cv"] <= 0,

            # Generation history
            "generation_history": generation_data,
        }

        return result

    def save_result(self, result: dict, output_subdir: str = None):
        """Save result to JSON"""
        if output_subdir:
            save_dir = self.output_dir / output_subdir
        else:
            save_dir = self.output_dir

        save_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{result['method_name']}_G{result['problem_id']:02d}_run{result['run_id']}.json"
        filepath = save_dir / filename

        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"Saved: {filepath}")
        return filepath


def run_full_collection(problems: list = [2, 4],
                        num_runs: int = 25,
                        methods: list = None,
                        output_dir: str = None):
    """
    Full collection: All methods on specified problems

    Args:
        problems: List of problem IDs
        num_runs: Number of runs per method-problem
        methods: List of method configs (None = all 27 methods)
        output_dir: Output directory (required, already timestamped)
    """
    if methods is None:
        methods = ALL_METHODS

    if output_dir is None:
        raise ValueError("output_dir must be specified!")

    print("=" * 80)
    print("FULL MEASURES COLLECTION")
    print("=" * 80)
    print(f"Methods: {len(methods)}")
    print(f"Problems: {problems}")
    print(f"Runs per method-problem: {num_runs}")
    print(f"Total experiments: {len(methods) * len(problems) * num_runs}")
    print(f"Output: {output_dir}")
    print("=" * 80)

    collector = MeasuresCollector(output_dir=output_dir)

    total = len(methods) * len(problems) * num_runs
    completed = 0

    for method_config in methods:
        for problem_id in problems:
            for run_id in range(num_runs):
                print(
                    f"\n[{completed + 1}/{total}] Running: {method_config['name']} - G{problem_id:02d} - Run {run_id}")

                result = collector.run_single_method(
                    method_config=method_config,
                    problem_id=problem_id,
                    run_id=run_id,
                    population_size=100,
                    max_evaluations=500000
                )

                collector.save_result(result)  # Save directly to output_dir

                print(f"  Final error: {result['final_error']:.6e}, Feasible: {result['is_feasible']}")

                completed += 1

    print("\n" + "=" * 80)
    print("FULL COLLECTION COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    import sys

    print("=" * 80)
    print("MEASURES COLLECTION - 27 Methods (ABCHS + Fixed BCHMs)")
    print("=" * 80)

    if len(sys.argv) > 1:
        # Parse arguments: python collect_method_metrics.py G02 G04 --runs 25
        problems = []
        num_runs = 25

        for arg in sys.argv[1:]:
            if arg.startswith('G'):
                problems.append(int(arg[1:]))
            elif arg == '--runs' and len(sys.argv) > sys.argv.index(arg) + 1:
                num_runs = int(sys.argv[sys.argv.index(arg) + 1])

        if not problems:
            problems = [2, 4]

        print(f"\nConfiguration:")
        print(f"  Problems: {['G{:02d}'.format(p) for p in problems]}")
        print(f"  Runs per method-problem: {num_runs}")
        print(f"  Methods: {len(ALL_METHODS)}")
        print(f"  Total experiments: {len(ALL_METHODS) * len(problems) * num_runs}")
        print()

        run_full_collection(problems=problems, num_runs=num_runs)
    else:
        print("\nUsage: python scripts/collect_method_metrics.py G02 G04 --runs 25")
        print("\nThis will run all 27 methods:")
        print("  - 1 ABCHS (adaptive)")
        print("  - 26 Fixed BCHMs (13 position × 2 velocity)")
        print("\nMeasures collected per generation:")
        print("  - gbest_error: |gbest_fitness - known_optimum|")
        print("  - gbest_cv: constraint violation")
        print("  - prob_infeas_bounds: violated_components / total_components")
        print("  - diversity: population diversity")
        print("  - mean_error, std_error: population statistics")
        print("  - nfs: number of feasible solutions (pbest CV <= 0)")
        print("  - feasibility_rate: NFS / population_size")
        print("  - improvement: gbest improved this generation")
        print("  - gens_since_improvement: generations since last improvement")
        print("  - updates_this_gen: times gbest updated this generation")
        print("\nFor testing, use: python tests/test_measures_collection.py")
