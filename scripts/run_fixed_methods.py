"""
Run Fixed BCHM Methods
26 fixed methods: 13 position × 2 velocity strategies (DeB, RaB)
"""

from scripts.collect_measures import ALL_METHODS


# Global function for multiprocessing (must be picklable)
def _run_single_problem_fixed(args):
    """Run fixed methods on a single problem - for parallel execution"""
    problem_id, num_runs, output_dir = args
    from scripts.collect_measures import run_full_collection, ALL_METHODS

    fixed_methods = [m for m in ALL_METHODS if m.get("type") != "adaptive"]

    run_full_collection(
        problems=[problem_id],
        num_runs=num_runs,
        methods=fixed_methods,
        output_dir=output_dir
    )


def run_fixed_methods(problems: list, num_runs: int = 25, parallel: bool = False):
    """
    Run all 26 fixed BCHM methods

    Args:
        problems: List of problem IDs
        num_runs: Number of runs per method-problem
        parallel: If True, parallelize across problems
    """
    # Extract only fixed methods (exclude ABCHS)
    fixed_methods = [m for m in ALL_METHODS if m.get("type") != "adaptive"]

    print("=" * 80)
    print("FIXED BCHM METHODS - PSO with Fixed Method Selection")
    print("=" * 80)
    print(f"\nMethods: {len(fixed_methods)} fixed")
    print(f"Problems: {['G{:02d}'.format(p) for p in problems]}")
    print(f"Runs per method-problem: {num_runs}")
    print(f"Total experiments: {len(fixed_methods) * len(problems) * num_runs}")
    print(f"Parallel: {parallel}")
    print()

    # Show methods
    print("Fixed methods:")
    for i, m in enumerate(fixed_methods, 1):
        print(f"  {i:2d}. {m['name']}")

    print("\n" + "=" * 80)

    # Collect measures using shared code
    from scripts.collect_measures import run_full_collection

    if parallel:
        from multiprocessing import Pool, cpu_count
        from datetime import datetime
        from pathlib import Path

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"results/experiments/fixed_methods/{timestamp}").resolve()
        output_dir = str(output_dir)  # Convert back to string for JSON serialization

        n_cores = min(cpu_count(), len(problems))
        print(f"\nParallelizing across {len(problems)} problems using {n_cores} cores")
        print(f"Output directory: {output_dir}")

        # Prepare arguments for parallel execution
        args_list = [(p, num_runs, output_dir) for p in problems]

        with Pool(n_cores) as pool:
            pool.map(_run_single_problem_fixed, args_list)
    else:
        from datetime import datetime
        from pathlib import Path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"results/experiments/fixed_methods/{timestamp}").resolve()
        output_dir = str(output_dir)

        run_full_collection(
            problems=problems,
            num_runs=num_runs,
            methods=fixed_methods,
            output_dir=output_dir
        )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("=" * 80)
        print("RUN FIXED BCHM METHODS")
        print("=" * 80)
        print("\nUsage: python scripts/run_fixed_methods.py G02 G04 --runs 25")
        print("\nThis will run 26 fixed methods:")
        print("  13 position methods × 2 velocity strategies (DeB, RaB)")
        print("\nPosition methods:")
        print("  - Boundary, Midpoint_Target, Midpoint_Best")
        print("  - Random, Reflection, Wrapping")
        print("  - ExpC_Target, ExpC_Best")
        print("  - Vector_Target, Vector_Best")
        print("  - Dismiss, Centroid, Evolutionary")
        print("\nVelocity strategies:")
        print("  - DeB: Deterministic Back (V = -0.5 * V)")
        print("  - RaB: Random Back (V = -λ * V, λ ~ U[0,1])")
        sys.exit(0)

    # Parse arguments
    problems = []
    num_runs = 25

    for arg in sys.argv[1:]:
        if arg.startswith('G'):
            problems.append(int(arg[1:]))
        elif arg == '--runs' and len(sys.argv) > sys.argv.index(arg) + 1:
            num_runs = int(sys.argv[sys.argv.index(arg) + 1])

    if not problems:
        problems = [2, 4]

    run_fixed_methods(problems=problems, num_runs=num_runs)
