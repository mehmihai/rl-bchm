"""
Run DQN Evaluation
Evaluate trained DQN model on CEC2006 benchmark with 25 independent runs per problem
"""

import json
import logging
import multiprocessing
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from src.benchmarks.cec2006 import get_problem
from src.rl.dqn import DQNAgent
from src.rl.rl_pso import RL_PSO, METHOD_POOL_CUSTOM5
from src.utils.generation_tracker import GenerationMetrics

# Force spawn method for macOS compatibility
try:
    multiprocessing.set_start_method('spawn')
except RuntimeError:
    pass  # Already set

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def load_baseline_results():
    """Load baseline results from ABCHS custom for comparison"""
    baselines = {}
    baseline_dir = Path("results/experiments/abchs_custom5")

    if baseline_dir.exists():
        logging.info(f"Loading baseline results from {baseline_dir}")
        for problem_file in baseline_dir.glob("G*_summary.json"):
            try:
                with open(problem_file, 'r') as f:
                    data = json.load(f)
                    problem_id = problem_file.stem.split('_')[0]
                    baselines[problem_id] = {
                        'mean_error': data.get('mean_error', None),
                        'best_error': data.get('best_error', None),
                        'feasibility_rate': data.get('feasibility_rate', None)
                    }
            except Exception as e:
                logging.warning(f"Could not load baseline for {problem_file}: {e}")
    else:
        logging.warning(f"Baseline directory not found: {baseline_dir}")

    return baselines


def _run_single_evaluation(args):
    """
    Run single evaluation (for parallel execution)

    Args:
        args: Tuple of (problem_id, run_id, model_path, random_seed, max_evaluations)

    Returns:
        dict: Run results
    """
    problem_id, run_id, model_path, random_seed, max_evaluations = args

    try:
        problem = get_problem(problem_id)

        # Load DQN agent (each worker needs its own copy)
        dqn_agent = DQNAgent(
            state_dim=9,
            action_dim=5,
            device='cpu'
        )
        dqn_agent.load(model_path)
        dqn_agent.epsilon = 0.0

        # Create RL-PSO optimizer
        optimizer = RL_PSO(
            problem=problem,
            population_size=100,
            max_evaluations=max_evaluations,
            random_seed=random_seed,
            dqn_agent=dqn_agent,
            training_mode=False,
            method_pool=METHOD_POOL_CUSTOM5
        )

        # Track metrics
        generation_data = []
        last_improvement_generation = 0
        prev_gbest_fitness = np.inf
        prev_gbest_error = np.inf
        first_feasible_generation = None

        # Initial metrics
        metrics = GenerationMetrics(optimizer, problem)
        pop_errors = [abs(p.fitness - problem.known_optimum) for p in optimizer.swarm]

        if first_feasible_generation is None and metrics.nfs > 0:
            first_feasible_generation = 0

        gbest_to_optimum_distance = float(np.linalg.norm(optimizer.gbest_position - problem.optimum_position))

        initial_data = {
            "generation": 0,
            "nfe": int(optimizer.nfe),
            "gbest_error": float(metrics.error),
            "gbest_cv": float(metrics.gbest_cv),
            "prob_infeas_bounds": float(metrics.prob_infeas),
            "diversity": float(metrics.diversity),
            "mean_error": float(np.mean(pop_errors)),
            "std_error": float(np.std(pop_errors)),
            "nfs": int(metrics.nfs),
            "feasibility_rate": float(metrics.feasibility_rate),
            "improvement": bool(False),
            "gens_since_improvement": int(0),
            "first_feasible_generation": int(
                first_feasible_generation) if first_feasible_generation is not None else None,
            "gbest_to_optimum_distance": float(gbest_to_optimum_distance),
            "selected_method_id": int(optimizer.previous_method_id)
        }
        generation_data.append(initial_data)

        # Run optimization
        while not optimizer.is_done():
            prev_gbest_fitness = optimizer.gbest_fitness
            prev_gbest_error = abs(prev_gbest_fitness - problem.known_optimum)

            optimizer.step()

            metrics = GenerationMetrics(optimizer, problem)
            pop_errors = [abs(p.fitness - problem.known_optimum) for p in optimizer.swarm]

            improvement = optimizer.gbest_fitness < prev_gbest_fitness
            if improvement:
                last_improvement_generation = optimizer.generation

            gens_since_improvement = optimizer.generation - last_improvement_generation

            if first_feasible_generation is None and metrics.nfs > 0:
                first_feasible_generation = optimizer.generation

            current_error = float(metrics.error)
            if prev_gbest_error > 0:
                improvement_rate = (prev_gbest_error - current_error) / prev_gbest_error
            else:
                improvement_rate = 0.0

            gbest_to_optimum_distance = float(np.linalg.norm(optimizer.gbest_position - problem.optimum_position))

            gen_data = {
                "generation": int(optimizer.generation),
                "nfe": int(optimizer.nfe),
                "gbest_error": float(current_error),
                "gbest_cv": float(metrics.gbest_cv),
                "prob_infeas_bounds": float(metrics.prob_infeas),
                "diversity": float(metrics.diversity),
                "mean_error": float(np.mean(pop_errors)),
                "std_error": float(np.std(pop_errors)),
                "nfs": int(metrics.nfs),
                "feasibility_rate": float(metrics.feasibility_rate),
                "improvement": bool(improvement),
                "gens_since_improvement": int(gens_since_improvement),
                "first_feasible_generation": int(
                    first_feasible_generation) if first_feasible_generation is not None else None,
                "improvement_rate": float(improvement_rate),
                "gbest_to_optimum_distance": float(gbest_to_optimum_distance),
                "selected_method_id": int(optimizer.previous_method_id)
            }
            generation_data.append(gen_data)

        # Final stats
        final_stats = optimizer.get_stats()
        final_cv = problem.compute_cv(optimizer.gbest_position)

        # Get method selection statistics
        action_counts = final_stats['action_counts']
        action_frequencies = final_stats['action_frequencies']

        # Method names for logging
        method_names = ['Random&Deb', 'ExpTarget&RandB', 'ExpBest&Deb', 'MidBest&Deb', 'Boundary&RandB']

        # Find most used method
        most_used_idx = np.argmax(action_counts)
        most_used_method = method_names[most_used_idx]
        most_used_freq = action_frequencies[most_used_idx]

        return {
            "success": True,
            "run_id": int(run_id),
            "problem": f"G{problem_id:02d}",
            "random_seed": int(random_seed),
            "final_fitness": float(final_stats['gbest_fitness']),
            "final_error": float(abs(final_stats['gbest_fitness'] - problem.known_optimum)),
            "is_feasible": bool(final_stats['gbest_is_feasible']),
            "cv": float(final_cv),
            "nfe": int(final_stats['nfe']),
            "generations": int(final_stats['generation']),
            "first_feasible_generation": int(
                first_feasible_generation) if first_feasible_generation is not None else None,
            "generation_data": generation_data,
            "action_counts": [int(x) for x in action_counts],
            "action_frequencies": [float(x) for x in action_frequencies],
            "most_used_method": str(most_used_method),
            "most_used_frequency": float(most_used_freq)
        }

    except Exception as e:
        return {
            "success": False,
            "run_id": run_id,
            "problem": f"G{problem_id:02d}",
            "error": str(e)
        }


def run_dqn_eval(model_path, problems, num_runs=25, max_evaluations=500000, num_workers=12):
    """
    Evaluate trained DQN model on CEC2006 problems

    Args:
        model_path: Path to trained DQN model (.pth file)
        problems: List of problem IDs to evaluate
        num_runs: Number of independent runs per problem (default: 25)
        max_evaluations: Max evaluations per run (default: 500k)
        num_workers: Number of parallel workers (default: 12)
    """
    print("=" * 80)
    print("DQN EVALUATION - CEC2006 BENCHMARK (PARALLEL)")
    print("=" * 80)
    print(f"\nModel: {model_path}")
    print(f"Problems: {problems}")
    print(f"Runs per problem: {num_runs}")
    print(f"Workers: {num_workers}")
    print(f"Max evaluations: {max_evaluations:,}")
    print()

    logging.info("=" * 80)
    logging.info("DQN EVALUATION STARTED (PARALLEL)")
    logging.info(f"Model: {model_path}")
    logging.info(f"Problems: {len(problems)} problems")
    logging.info(f"Runs per problem: {num_runs}")
    logging.info(f"Workers: {num_workers}")
    logging.info("=" * 80)

    # Load baseline results for comparison
    logging.info("Loading baseline results...")
    baselines = load_baseline_results()
    if baselines:
        logging.info(f"✅ Loaded baselines for {len(baselines)} problems")
    else:
        logging.warning("⚠️  No baseline results found for comparison")

    # Load trained DQN agent
    print("Loading trained DQN model...")
    logging.info("Loading trained DQN model...")
    dqn_agent = DQNAgent(
        state_dim=9,
        action_dim=5,
        device='cpu'
    )
    dqn_agent.load(model_path)
    dqn_agent.epsilon = 0.0  # Pure exploitation mode (no exploration)
    print(f"✅ Model loaded successfully (epsilon={dqn_agent.epsilon})\n")
    logging.info(f"✅ DQN model loaded (epsilon={dqn_agent.epsilon})")

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results/experiments/dqn/evaluation") / f"eval_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}\n")
    logging.info(f"Output directory: {output_dir}")

    # Setup file handler for detailed logging
    log_file = output_dir / "evaluation.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logging.getLogger().addHandler(file_handler)
    logging.info(f"Logging to file: {log_file}")

    # Evaluate each problem
    all_results = {}

    for problem_id in problems:
        print(f"\n{'=' * 80}")
        print(f"Evaluating G{problem_id:02d} ({num_runs} runs)")
        print(f"{'=' * 80}\n")

        logging.info("=" * 80)
        logging.info(f"PROBLEM G{problem_id:02d} - Starting {num_runs} runs with {num_workers} workers")
        logging.info("=" * 80)

        # Prepare arguments for parallel execution
        run_args = [
            (problem_id, run_id, model_path, run_id * 1000 + problem_id, max_evaluations)
            for run_id in range(num_runs)
        ]

        # Run in parallel
        print(f"  Launching {num_workers} workers for {num_runs} runs...")
        logging.info(f"Launching {num_workers} workers for {num_runs} runs...")
        logging.info(f"Multiprocessing start method: {multiprocessing.get_start_method()}")

        import time
        start_time = time.time()

        try:
            with Pool(processes=num_workers) as pool:
                logging.info(f"Pool created with {num_workers} workers")
                print(f"  Workers initializing (loading PyTorch + DQN model)...")

                # Use map_async to get progress
                async_result = pool.map_async(_run_single_evaluation, run_args)

                # Wait with progress indication
                completed = 0
                while not async_result.ready():
                    async_result.wait(timeout=5)
                    if async_result._number_left is not None:
                        new_completed = num_runs - async_result._number_left
                        if new_completed > completed:
                            completed = new_completed
                            print(f"  Progress: {completed}/{num_runs} runs completed...")
                            logging.info(f"Progress: {completed}/{num_runs} runs completed")

                problem_results = async_result.get()
                logging.info("Pool.map completed")
        except Exception as e:
            logging.error(f"Parallel execution failed: {e}")
            logging.info("Falling back to sequential execution...")
            problem_results = [_run_single_evaluation(args) for args in run_args]

        elapsed = time.time() - start_time
        logging.info(f"Completed {num_runs} runs in {elapsed:.2f}s ({elapsed / num_runs:.2f}s per run)")

        # Process results
        successful_results = []
        for result in problem_results:
            if not result['success']:
                print(f"  ⚠️  Run {result['run_id'] + 1} FAILED: {result.get('error', 'Unknown error')}")
                logging.error(
                    f"G{problem_id:02d} Run {result['run_id'] + 1} - FAILED: {result.get('error', 'Unknown')}")
                continue

            successful_results.append(result)

            # Print progress
            if result['is_feasible']:
                print(
                    f"  G{problem_id:02d} - Run {result['run_id'] + 1}/{num_runs} → ✅ Feas: {result['final_fitness']:.6e} | Method: {result['most_used_method']} ({result['most_used_frequency']:.0%})")
                logging.info(
                    f"G{problem_id:02d} Run {result['run_id'] + 1} - ✅ FEASIBLE - Fitness: {result['final_fitness']:.6e}, Error: {result['final_error']:.6e}")
                logging.info(
                    f"  Method usage: {result['most_used_method']} ({result['most_used_frequency']:.0%}), All: {result['action_frequencies']}")
            else:
                print(
                    f"  G{problem_id:02d} - Run {result['run_id'] + 1}/{num_runs} → ❌ Infeas: CV={result['cv']:.6e} | Method: {result['most_used_method']} ({result['most_used_frequency']:.0%})")
                logging.info(f"G{problem_id:02d} Run {result['run_id'] + 1} - ❌ INFEASIBLE - CV: {result['cv']:.6e}")
                logging.info(
                    f"  Method usage: {result['most_used_method']} ({result['most_used_frequency']:.0%}), All: {result['action_frequencies']}")

        problem_results = successful_results

        # Compute statistics for this problem
        fitnesses = [r['final_fitness'] for r in problem_results]
        errors = [r['final_error'] for r in problem_results]
        feasible_runs = [r for r in problem_results if r['is_feasible']]

        # Aggregate method usage across all runs
        method_names = ['Random&Deb', 'ExpTarget&RandB', 'ExpBest&Deb', 'MidBest&Deb', 'Boundary&RandB']
        total_action_counts = np.zeros(5)
        for r in problem_results:
            total_action_counts += np.array(r['action_counts'])

        total_actions = np.sum(total_action_counts)
        avg_action_frequencies = total_action_counts / total_actions if total_actions > 0 else total_action_counts

        # Find most preferred method across all runs
        most_preferred_idx = np.argmax(avg_action_frequencies)
        most_preferred_method = method_names[most_preferred_idx]
        most_preferred_freq = avg_action_frequencies[most_preferred_idx]

        problem_stats = {
            "problem": f"G{problem_id:02d}",
            "num_runs": num_runs,
            "best_fitness": float(np.min(fitnesses)),
            "worst_fitness": float(np.max(fitnesses)),
            "mean_fitness": float(np.mean(fitnesses)),
            "median_fitness": float(np.median(fitnesses)),
            "std_fitness": float(np.std(fitnesses)),
            "best_error": float(np.min(errors)),
            "mean_error": float(np.mean(errors)),
            "median_error": float(np.median(errors)),
            "feasibility_rate": len(feasible_runs) / num_runs,
            "num_feasible": len(feasible_runs),
            "best_feasible_fitness": float(
                np.min([r['final_fitness'] for r in feasible_runs])) if feasible_runs else None,
            "mean_feasible_fitness": float(
                np.mean([r['final_fitness'] for r in feasible_runs])) if feasible_runs else None,
            "best_feasible_error": float(np.min([r['final_error'] for r in feasible_runs])) if feasible_runs else None,
            "mean_feasible_error": float(np.mean([r['final_error'] for r in feasible_runs])) if feasible_runs else None,
            "method_usage": {
                "method_names": method_names,
                "action_counts": [int(x) for x in total_action_counts],
                "action_frequencies": [float(x) for x in avg_action_frequencies],
                "most_preferred_method": most_preferred_method,
                "most_preferred_frequency": float(most_preferred_freq)
            },
            "runs": problem_results
        }

        all_results[f"G{problem_id:02d}"] = problem_stats

        # Print summary for this problem
        print(f"\nG{problem_id:02d} Summary:")
        print(f"  Feasibility: {problem_stats['feasibility_rate']:.0%} ({len(feasible_runs)}/{num_runs})")

        logging.info("-" * 80)
        logging.info(f"G{problem_id:02d} SUMMARY")
        logging.info(f"  Feasibility: {problem_stats['feasibility_rate']:.0%} ({len(feasible_runs)}/{num_runs})")

        if feasible_runs:
            print(f"  Best error: {problem_stats['best_feasible_error']:.6e}")
            print(f"  Mean error: {problem_stats['mean_feasible_error']:.6e}")
            print(
                f"  🎯 Preferred method: {problem_stats['method_usage']['most_preferred_method']} ({problem_stats['method_usage']['most_preferred_frequency']:.0%})")

            logging.info(f"  Best error: {problem_stats['best_feasible_error']:.6e}")
            logging.info(f"  Mean error: {problem_stats['mean_feasible_error']:.6e}")
            logging.info(f"  Median error: {problem_stats['median_error']:.6e}")
            logging.info(f"  Std error: {problem_stats['std_fitness']:.6e}")

            # Log method usage distribution
            logging.info(f"\n  METHOD USAGE DISTRIBUTION (across all {num_runs} runs):")
            for i, (name, freq) in enumerate(zip(problem_stats['method_usage']['method_names'],
                                                 problem_stats['method_usage']['action_frequencies'])):
                count = problem_stats['method_usage']['action_counts'][i]
                logging.info(f"    {name:20s}: {freq:6.1%} ({count:4d} selections)")
            logging.info(
                f"  → Most preferred: {problem_stats['method_usage']['most_preferred_method']} ({problem_stats['method_usage']['most_preferred_frequency']:.0%})")
        else:
            print(f"  ❌ No feasible solutions found")
            logging.warning(f"  ❌ No feasible solutions found for G{problem_id:02d}")

            # Still log method usage for failed runs
            logging.info(f"\n  METHOD USAGE DISTRIBUTION (across all {num_runs} runs):")
            for i, (name, freq) in enumerate(zip(problem_stats['method_usage']['method_names'],
                                                 problem_stats['method_usage']['action_frequencies'])):
                count = problem_stats['method_usage']['action_counts'][i]
                logging.info(f"    {name:20s}: {freq:6.1%} ({count:4d} selections)")

        # Compare with baseline
        baseline_key = f"G{problem_id:02d}"
        if baseline_key in baselines:
            baseline = baselines[baseline_key]
            logging.info(f"\n  COMPARISON WITH ABCHS CUSTOM5 BASELINE:")

            if baseline['mean_error'] is not None and problem_stats['mean_feasible_error'] is not None:
                improvement = ((baseline['mean_error'] - problem_stats['mean_feasible_error']) / baseline[
                    'mean_error']) * 100
                logging.info(f"    Baseline mean error: {baseline['mean_error']:.6e}")
                logging.info(f"    DQN mean error:      {problem_stats['mean_feasible_error']:.6e}")
                if improvement > 0:
                    logging.info(f"    → DQN is {improvement:.1f}% BETTER ✅")
                    print(f"  📊 vs ABCHS: {improvement:+.1f}% better")
                else:
                    logging.info(f"    → DQN is {abs(improvement):.1f}% WORSE ❌")
                    print(f"  📊 vs ABCHS: {improvement:+.1f}% worse")

            if baseline['feasibility_rate'] is not None:
                feas_diff = (problem_stats['feasibility_rate'] - baseline['feasibility_rate']) * 100
                logging.info(f"    Baseline feasibility: {baseline['feasibility_rate']:.0%}")
                logging.info(f"    DQN feasibility:      {problem_stats['feasibility_rate']:.0%}")
                logging.info(f"    → Difference: {feas_diff:+.0f}%")
        else:
            logging.warning(f"  No baseline data available for comparison")

        logging.info("-" * 80)

        # Save individual problem results
        problem_output = output_dir / f"G{problem_id:02d}_results.json"
        with open(problem_output, 'w') as f:
            json.dump(problem_stats, f, indent=2)

    # Save summary
    summary = {
        "model_path": str(model_path),
        "timestamp": timestamp,
        "num_runs_per_problem": num_runs,
        "max_evaluations": max_evaluations,
        "problems": problems,
        "results": all_results
    }

    summary_path = output_dir / "evaluation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Print final summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"\n{'Problem':<10} {'Feas%':<10} {'Best Error':<15} {'Mean Error':<15}")
    print("-" * 80)

    for problem_id in problems:
        key = f"G{problem_id:02d}"
        stats = all_results[key]

        if stats['feasibility_rate'] > 0:
            print(f"{key:<10} "
                  f"{stats['feasibility_rate']:<10.0%} "
                  f"{stats['best_feasible_error']:<15.6e} "
                  f"{stats['mean_feasible_error']:<15.6e}")
        else:
            print(f"{key:<10} {stats['feasibility_rate']:<10.0%} {'N/A':<15} {'N/A':<15}")

    # Overall statistics
    feas_rates = [all_results[f"G{pid:02d}"]['feasibility_rate'] for pid in problems]
    avg_feas_rate = np.mean(feas_rates)
    perfect_feas = sum(1 for r in feas_rates if r == 1.0)

    print("\n" + "=" * 80)
    print(f"Overall feasibility rate: {avg_feas_rate:.1%}")
    print(f"Problems with 100% feasibility: {perfect_feas}/{len(problems)}")
    print(f"\nResults saved to: {output_dir}")
    print("=" * 80)

    # Final logging summary
    logging.info("\n" + "=" * 80)
    logging.info("FINAL SUMMARY")
    logging.info("=" * 80)
    logging.info(f"Total problems evaluated: {len(problems)}")
    logging.info(f"Overall feasibility rate: {avg_feas_rate:.1%}")
    logging.info(f"Problems with 100% feasibility: {perfect_feas}/{len(problems)}")

    # Count wins vs baseline
    if baselines:
        wins = 0
        losses = 0
        ties = 0
        for pid in problems:
            key = f"G{pid:02d}"
            if key in baselines and baselines[key]['mean_error'] is not None:
                dqn_err = all_results[key]['mean_feasible_error']
                baseline_err = baselines[key]['mean_error']
                if dqn_err is not None:
                    if dqn_err < baseline_err * 0.99:  # 1% threshold
                        wins += 1
                    elif dqn_err > baseline_err * 1.01:
                        losses += 1
                    else:
                        ties += 1

        logging.info(f"\nComparison with ABCHS Custom5 baseline:")
        logging.info(f"  Wins:   {wins}/{len(problems)} (DQN better)")
        logging.info(f"  Losses: {losses}/{len(problems)} (ABCHS better)")
        logging.info(f"  Ties:   {ties}/{len(problems)} (similar)")

    logging.info(f"\nAll results saved to: {output_dir}")
    logging.info("=" * 80)

    return all_results
