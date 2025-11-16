"""
Run ABCHS Castillo (Original)
ABCHS with original method pool: Ran&RaB, Cen&DeB, Ref&DeB, Wra&RaB
"""

from scripts.collect_measures import ALL_METHODS


# Global function for multiprocessing (must be picklable)
def _run_single_problem_abchs(args):
    """Run ABCHS on a single problem - for parallel execution"""
    problem_id, num_runs, output_dir = args
    from scripts.collect_measures import run_full_collection, ALL_METHODS

    abchs_methods = [m for m in ALL_METHODS if m.get("type") == "adaptive"]

    run_full_collection(
        problems=[problem_id],
        num_runs=num_runs,
        methods=abchs_methods,
        output_dir=output_dir
    )


def run_abchs_castillo(problems: list, num_runs: int = 25, parallel: bool = False):
    """
    Run ABCHS Castillo (original paper implementation)

    Method pool: [Ran&RaB, Cen&DeB, Ref&DeB, Wra&RaB]

    Args:
        problems: List of problem IDs
        num_runs: Number of runs per problem
        parallel: If True, parallelize across problems
    """
    # Extract ABCHS method
    abchs_methods = [m for m in ALL_METHODS if m.get("type") == "adaptive"]

    print("=" * 80)
    print("ABCHS CASTILLO - Adaptive Boundary Constraint-Handling Scheme (Original)")
    print("=" * 80)
    print(f"\nMethod: ABCHS (adaptive)")
    print(f"Method pool: [Ran&RaB, Cen&DeB, Ref&DeB, Wra&RaB]")
    print(f"\nProblems: {['G{:02d}'.format(p) for p in problems]}")
    print(f"Runs per problem: {num_runs}")
    print(f"Total experiments: {len(problems) * num_runs}")
    print(f"Parallel: {parallel}")
    print()

    print("Algorithm:")
    print("  Stage 1 (NFS = 0): Ran&RaB (exploration)")
    print("  Stage 2 (NFS > 0): Adaptive selection via roulette wheel")
    print("    - Update probabilities every LP generations")
    print("    - LP = round(0.5 * D) + 2")
    print()

    print("=" * 80)

    # Collect measures
    from scripts.collect_measures import run_full_collection

    if parallel:
        from multiprocessing import Pool, cpu_count
        from datetime import datetime
        from pathlib import Path

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"results/experiments/abchs_castillo/{timestamp}").resolve()
        output_dir = str(output_dir)  # Convert back to string for JSON serialization

        n_cores = min(cpu_count(), len(problems))
        print(f"\nParallelizing across {len(problems)} problems using {n_cores} cores")
        print(f"Output directory: {output_dir}")

        # Prepare arguments for parallel execution
        args_list = [(p, num_runs, output_dir) for p in problems]

        with Pool(n_cores) as pool:
            pool.map(_run_single_problem_abchs, args_list)
    else:
        from datetime import datetime
        from pathlib import Path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"results/experiments/abchs_castillo/{timestamp}").resolve()
        output_dir = str(output_dir)

        run_full_collection(
            problems=problems,
            num_runs=num_runs,
            methods=abchs_methods,
            output_dir=output_dir
        )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("=" * 80)
        print("RUN ABCHS CASTILLO (ORIGINAL)")
        print("=" * 80)
        print("\nUsage: python scripts/run_abchs_castillo.py G02 G04 --runs 25")
        print("\nABCHS (Adaptive Boundary Constraint-Handling Scheme)")
        print("Original implementation from Castillo et al. paper")
        print("\nMethod pool (4 methods):")
        print("  1. Ran&RaB  - Random + Random Back")
        print("  2. Cen&DeB  - Centroid + Deterministic Back")
        print("  3. Ref&DeB  - Reflection + Deterministic Back")
        print("  4. Wra&RaB  - Wrapping + Random Back")
        print("\nTwo-stage operation:")
        print("  Stage 1: Use Ran&RaB when no feasible solutions (NFS = 0)")
        print("  Stage 2: Adaptive selection when NFS > 0")
        print("    - Roulette wheel based on performance")
        print("    - Update probabilities every LP generations")
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

    run_abchs_castillo(problems=problems, num_runs=num_runs)
