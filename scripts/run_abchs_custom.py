"""
Run ABCHS Custom
ABCHS with custom method pool (user-defined)
"""

import numpy as np

from scripts.collect_measures import MeasuresCollector
from src.benchmarks.cec2006 import get_problem
from src.core.abchs import ABCHS
from src.core.boundary_handler import (
    METHOD_MIDPOINT_BEST,
    METHOD_UNIF, METHOD_MIRROR, METHOD_EXPC_TARGET, METHOD_EXPC_BEST
)
from src.core.velocity_strategy import VELOCITY_DEB, VELOCITY_RAB
from src.utils.generation_tracker import GenerationMetrics


# Global function for multiprocessing (must be picklable)
def _run_single_problem_custom(args):
    """Run ABCHS custom on a single problem - for parallel execution"""
    problem_id, method_pool, pool_name, num_runs, output_subdir = args

    print(f"\n{'=' * 80}")
    print(f"WORKER: Processing G{problem_id:02d} with {num_runs} runs")
    print(f"{'=' * 80}\n")

    for run_id in range(num_runs):
        print(f"  G{problem_id:02d} - Run {run_id}/{num_runs}")

        problem = get_problem(problem_id)
        random_seed = run_id * 1000 + problem_id

        # Create ABCHS with custom pool
        algorithm = ABCHSCustom(
            problem=problem,
            population_size=100,
            max_evaluations=500000,
            method_pool=method_pool,
            random_seed=random_seed
        )

        # Track improvement
        last_improvement_generation = 0
        prev_gbest_fitness = np.inf
        prev_gbest_error = np.inf

        # Track first feasible generation
        first_feasible_generation = None

        # Storage
        generation_data = []

        # Initial
        metrics = GenerationMetrics(algorithm, problem)
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
            "stage": int(algorithm.stage),
            "first_feasible_generation": first_feasible_generation,
            "improvement_rate": 0.0,
            "particles_improved_rate": particles_improved_rate,
            "gbest_to_optimum_distance": gbest_to_optimum_distance,
            "selected_method_id": algorithm.last_selected_method_id,
            "method_probabilities": algorithm.probabilities.tolist(),
            "cumulative_rsB": algorithm.rsB.tolist(),
            "cumulative_rsW": algorithm.rsW.tolist(),
        }
        generation_data.append(initial_data)

        # Run
        while algorithm.nfe < algorithm.max_evaluations:
            prev_gbest_fitness = algorithm.gbest_fitness
            prev_gbest_error = abs(prev_gbest_fitness - problem.known_optimum)

            algorithm.step()

            metrics = GenerationMetrics(algorithm, problem)
            pop_errors = [abs(p.fitness - problem.known_optimum) for p in algorithm.swarm]

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
                "stage": int(algorithm.stage),
                "first_feasible_generation": first_feasible_generation,
                "improvement_rate": float(improvement_rate),
                "particles_improved_rate": particles_improved_rate,
                "gbest_to_optimum_distance": gbest_to_optimum_distance,
                "selected_method_id": algorithm.last_selected_method_id,
                "method_probabilities": algorithm.probabilities.tolist(),
                "cumulative_rsB": algorithm.rsB.tolist(),
                "cumulative_rsW": algorithm.rsW.tolist(),
            }
            generation_data.append(gen_data)

        # Save
        final_metrics = generation_data[-1]
        result = {
            "method_name": f"ABCHS-{pool_name}",
            "problem_id": problem_id,
            "run_id": run_id,
            "config": {
                "population_size": 100,
                "max_evaluations": 500000,
                "random_seed": random_seed,
                "pool_name": pool_name,
                "pool_size": len(method_pool),
            },
            "final_error": final_metrics["gbest_error"],
            "final_cv": final_metrics["gbest_cv"],
            "is_feasible": final_metrics["gbest_cv"] <= 0,
            "generation_history": generation_data,
        }

        collector = MeasuresCollector(output_dir=f"results/experiments/abchs_custom")
        collector.save_result(result, output_subdir=output_subdir)

    print(f"  ✓ G{problem_id:02d} completed ({num_runs} runs)")
    return problem_id


class ABCHSCustom(ABCHS):
    """
    ABCHS with custom method pool
    User can specify which methods to include
    """

    def __init__(self, problem, population_size=100, max_evaluations=500000,
                 method_pool=None, random_seed=None):
        """
        Initialize ABCHS with custom method pool

        Args:
            problem: CEC2006Problem
            population_size: Swarm size
            max_evaluations: Max function evaluations
            method_pool: List of (position_method, velocity_strategy) tuples
            random_seed: Random seed
        """
        # Initialize base ABCHS
        super().__init__(
            problem=problem,
            population_size=population_size,
            max_evaluations=max_evaluations,
            random_seed=random_seed
        )

        # Override method pool if provided
        if method_pool is not None:
            self.method_pool = method_pool
            # Reset probabilities for new pool size
            n_methods = len(self.method_pool)
            self.probabilities = np.full(n_methods, 1.0 / n_methods)
            self.rsB = np.zeros(n_methods)
            self.rsW = np.zeros(n_methods)


def run_abchs_custom(problems: list, method_pool: list, pool_name: str, num_runs: int = 25, parallel: bool = False):
    """
    Run ABCHS with custom method pool

    Args:
        problems: List of problem IDs
        method_pool: List of (position_method, velocity_strategy) tuples
        pool_name: Name for this pool configuration
        num_runs: Number of runs per problem
        parallel: If True, parallelize across problems
    """
    print("=" * 80)
    print("ABCHS CUSTOM - User-Defined Method Pool")
    print("=" * 80)
    print(f"\nPool name: {pool_name}")
    print(f"Methods in pool: {len(method_pool)}")
    print(f"\nProblems: {['G{:02d}'.format(p) for p in problems]}")
    print(f"Runs per problem: {num_runs}")
    print(f"Total experiments: {len(problems) * num_runs}")
    print(f"Parallel: {parallel}")
    print()

    print("Method pool:")
    for i, (pos, vel) in enumerate(method_pool, 1):
        from src.core.boundary_handler import get_method_name
        print(f"  {i}. {get_method_name(pos)} & {vel}")

    print("\n" + "=" * 80)

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_subdir = f"abchs_custom_{pool_name}_{timestamp}"

    if parallel:
        from multiprocessing import Pool, cpu_count

        n_cores = min(cpu_count(), len(problems))
        print(f"\nParallelizing across {len(problems)} problems using {n_cores} cores")
        print(f"Output directory: results/experiments/abchs_custom/{output_subdir}")
        print()

        # Prepare arguments for parallel execution
        args_list = [(p, method_pool, pool_name, num_runs, output_subdir) for p in problems]

        with Pool(n_cores) as pool:
            pool.map(_run_single_problem_custom, args_list)

        print("\n" + "=" * 80)
        print("ABCHS CUSTOM COLLECTION COMPLETE!")
        print("=" * 80)
        return

    # Sequential execution (original code)
    # Custom collector for ABCHS custom
    collector = MeasuresCollector(output_dir=f"results/experiments/abchs_custom")

    total = len(problems) * num_runs
    completed = 0

    for problem_id in problems:
        for run_id in range(num_runs):
            print(f"\n[{completed + 1}/{total}] Running: ABCHS-{pool_name} - G{problem_id:02d} - Run {run_id}")

            problem = get_problem(problem_id)
            random_seed = run_id * 1000 + problem_id

            # Create ABCHS with custom pool
            algorithm = ABCHSCustom(
                problem=problem,
                population_size=100,
                max_evaluations=500000,
                method_pool=method_pool,
                random_seed=random_seed
            )

            # Track improvement
            last_improvement_generation = 0
            prev_gbest_fitness = np.inf
            prev_gbest_error = np.inf

            # Track first feasible generation
            first_feasible_generation = None

            # Storage
            generation_data = []

            # Initial
            metrics = GenerationMetrics(algorithm, problem)
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
                "stage": int(algorithm.stage),
                "first_feasible_generation": first_feasible_generation,
                "improvement_rate": 0.0,
                "particles_improved_rate": particles_improved_rate,
                "gbest_to_optimum_distance": gbest_to_optimum_distance,
                "selected_method_id": algorithm.last_selected_method_id,
                "method_probabilities": algorithm.probabilities.tolist(),
                "cumulative_rsB": algorithm.rsB.tolist(),
                "cumulative_rsW": algorithm.rsW.tolist(),
            }
            generation_data.append(initial_data)

            # Run
            while algorithm.nfe < algorithm.max_evaluations:
                prev_gbest_fitness = algorithm.gbest_fitness
                prev_gbest_error = abs(prev_gbest_fitness - problem.known_optimum)

                algorithm.step()

                metrics = GenerationMetrics(algorithm, problem)
                pop_errors = [abs(p.fitness - problem.known_optimum) for p in algorithm.swarm]

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
                    "stage": int(algorithm.stage),
                    "first_feasible_generation": first_feasible_generation,
                    "improvement_rate": float(improvement_rate),
                    "particles_improved_rate": particles_improved_rate,
                    "gbest_to_optimum_distance": gbest_to_optimum_distance,
                    "selected_method_id": algorithm.last_selected_method_id,
                    "method_probabilities": algorithm.probabilities.tolist(),
                    "cumulative_rsB": algorithm.rsB.tolist(),
                    "cumulative_rsW": algorithm.rsW.tolist(),
                }
                generation_data.append(gen_data)

            # Save
            final_metrics = generation_data[-1]
            result = {
                "method_name": f"ABCHS-{pool_name}",
                "problem_id": problem_id,
                "run_id": run_id,
                "config": {
                    "population_size": 100,
                    "max_evaluations": 500000,
                    "random_seed": random_seed,
                    "pool_name": pool_name,
                    "pool_size": len(method_pool),
                },
                "final_error": final_metrics["gbest_error"],
                "final_cv": final_metrics["gbest_cv"],
                "is_feasible": final_metrics["gbest_cv"] <= 0,
                "generation_history": generation_data,
            }

            collector.save_result(result, output_subdir=output_subdir)
            print(f"  Final error: {result['final_error']:.6e}, Feasible: {result['is_feasible']}")

            completed += 1

    print("\n" + "=" * 80)
    print("ABCHS CUSTOM COLLECTION COMPLETE!")
    print("=" * 80)


# Pool configuration
POOL_CONFIGS = {
    # Custom pool: Ran&DeB, ExpT&RaB, ExpB&DeB, MidB&DeB, Ref&RaB
    "custom5": [
        (METHOD_UNIF, VELOCITY_DEB),  # 1. Ran&DeB
        (METHOD_EXPC_TARGET, VELOCITY_RAB),  # 2. ExpT&RaB
        (METHOD_EXPC_BEST, VELOCITY_DEB),  # 3. ExpB&DeB
        (METHOD_MIDPOINT_BEST, VELOCITY_DEB),  # 4. MidB&DeB
        (METHOD_MIRROR, VELOCITY_RAB),  # 5. Ref&RaB
    ],
}

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("=" * 80)
        print("RUN ABCHS CUSTOM")
        print("=" * 80)
        print("\nUsage: python scripts/run_abchs_custom.py <pool_config> G02 G04 --runs 25")
        print("\nAvailable pool:")
        print("  custom5: [Ran&DeB, ExpT&RaB, ExpB&DeB, MidB&DeB, Ref&RaB]")
        sys.exit(0)

    # Parse arguments
    pool_name = sys.argv[1] if len(sys.argv) > 1 else "pool1"
    problems = []
    num_runs = 25

    for arg in sys.argv[2:]:
        if arg.startswith('G'):
            problems.append(int(arg[1:]))
        elif arg == '--runs' and len(sys.argv) > sys.argv.index(arg) + 1:
            num_runs = int(sys.argv[sys.argv.index(arg) + 1])

    if not problems:
        problems = [2, 4]

    if pool_name not in POOL_CONFIGS:
        print(f"Error: Pool '{pool_name}' not found!")
        print(f"Available pools: {list(POOL_CONFIGS.keys())}")
        sys.exit(1)

    run_abchs_custom(
        problems=problems,
        method_pool=POOL_CONFIGS[pool_name],
        pool_name=pool_name,
        num_runs=num_runs
    )
