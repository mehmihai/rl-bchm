"""
Test: Measures Collection
Verify collection of essential per-generation measures

MEASURES EXPLANATION:
===================

1. NFS (Number of Feasible Solutions):
   - Calculat din PBEST-uri, nu din poziții curente
   - NFS = count(pbest_cv <= 0)
   - Exemple:
     * 50 particule, 40 au pbest_cv <= 0 → NFS = 40
     * 100 particule, toate au pbest_cv > 0 → NFS = 0

2. Feasibility Rate:
   - feasibility_rate = NFS / population_size
   - Exemple:
     * NFS=40, pop=50 → feasibility_rate = 0.8 (80%)
     * NFS=0, pop=100 → feasibility_rate = 0.0 (0%)

3. Success Rate (calculat la final peste runs):
   - success_rate = count(runs with final_cv <= 0) / total_runs
   - Exemple:
     * 25 runs, 22 au final_cv <= 0 → success_rate = 0.88 (88%)
     * 25 runs, toate au final_cv > 0 → success_rate = 0.0 (0%)

4. Prob_infeas_bounds (boundary violations):
   - prob_infeas_bounds = componente_depășite / total_componente
   - Calculat per generație
   - Exemple:
     * 50 particule × 20 dimensiuni = 1000 componente
     * 150 componente au depășit bounds → prob_infeas = 0.15 (15%)
"""

from scripts.collect_measures import ALL_METHODS


def test_single_method_measures():
    """Test collection with Random&RaB on G02"""
    print("=" * 80)
    print("TEST: Measures Collection - Random&RaB on G02")
    print("=" * 80)

    # Save to dedicated test folder
    collector = MetricsCollector(output_dir="results/test_measures")

    # Find Random&RaB method
    ran_rab = next(m for m in ALL_METHODS if m["name"] == "Ran&RaB")

    print(f"\nMethod: {ran_rab['name']}")
    print(f"Problem: G02")
    print(f"Max evals: 50000 (quick test)")

    # Run
    result = collector.run_single_method(
        method_config=ran_rab,
        problem_id=2,
        run_id=1,
        population_size=50,
        max_evaluations=50000
    )

    # Save to test folder
    filepath = collector.save_result(result)

    # Summary
    print("\n" + "-" * 80)
    print("RESULTS:")
    print("-" * 80)
    print(f"Generations: {result['generation_history'][-1]['generation']}")
    print(f"Final NFE: {result['generation_history'][-1]['nfe']}")
    print(f"Final error: {result['final_error']:.6e}")
    print(f"Final CV: {result['final_cv']:.6e}")
    print(f"Feasible: {result['is_feasible']}")

    print(f"\nCollected {len(result['generation_history'])} generation snapshots")
    print(f"Saved to: {filepath}")

    print(f"\n" + "=" * 80)
    print("MEASURES PER GENERATION:")
    print("=" * 80)
    gen1 = result['generation_history'][1]

    measures = [
        ('gbest_error', 'Error from known optimum'),
        ('gbest_cv', 'Constraint violation at gbest'),
        ('prob_infeas_bounds', 'Boundary violation rate (violated_components/total_components)'),
        ('diversity', 'Population diversity'),
        ('mean_error', 'Mean error in population'),
        ('std_error', 'Std error in population'),
        ('nfs', 'Number of feasible solutions (pbest CV <= 0)'),
        ('feasibility_rate', 'NFS / population_size'),
        ('improvement', 'Gbest improved this generation'),
        ('gens_since_improvement', 'Generations since last improvement'),
        ('updates_this_gen', 'Number of times gbest updated this generation'),
    ]

    for key, description in measures:
        value = gen1[key]
        print(f"  {key:25s}: {value:12} - {description}")

    # Verify NFS calculation
    print("\n" + "-" * 80)
    print("NFS CALCULATION VERIFICATION:")
    print("-" * 80)
    print(f"Generation 1:")
    print(f"  NFS = {gen1['nfs']}")
    print(f"  Population size = 50")
    print(f"  Feasibility rate = {gen1['feasibility_rate']:.3f}")
    print(f"  Expected: {gen1['nfs'] / 50:.3f}")
    assert abs(gen1['feasibility_rate'] - gen1['nfs'] / 50) < 1e-6, "Feasibility rate mismatch!"
    print(f"  ✓ Verified: feasibility_rate = NFS / population_size")

    # Show evolution
    print("\n" + "-" * 80)
    print("EVOLUTION (sample generations):")
    print("-" * 80)
    print(f"{'Gen':>5} {'Error':>12} {'NFS':>5} {'Feas%':>7} {'Updates':>8} {'Stagnation':>11}")
    print("-" * 80)
    for i in [0, 10, 50, 100, 500, 999]:
        if i < len(result['generation_history']):
            gen = result['generation_history'][i]
            print(f"{gen['generation']:>5} "
                  f"{gen['gbest_error']:>12.6f} "
                  f"{gen['nfs']:>5} "
                  f"{gen['feasibility_rate'] * 100:>6.1f}% "
                  f"{gen['updates_this_gen']:>8} "
                  f"{gen['gens_since_improvement']:>11}")

    print("\n" + "=" * 80)
    print("TEST PASSED!")
    print("=" * 80)

    return result


def test_multiple_runs_success_rate():
    """Test success rate calculation across multiple runs"""
    print("\n" + "=" * 80)
    print("TEST: Success Rate Calculation - 3 runs")
    print("=" * 80)

    collector = MetricsCollector(output_dir="results/test_measures")
    ran_rab = next(m for m in ALL_METHODS if m["name"] == "Ran&RaB")

    results = []
    for run_id in range(3):
        print(f"\nRunning {run_id + 1}/3...")
        result = collector.run_single_method(
            method_config=ran_rab,
            problem_id=2,
            run_id=run_id,
            population_size=30,
            max_evaluations=10000
        )
        results.append(result)
        filepath = collector.save_result(result)
        print(f"  Final error: {result['final_error']:.6e}, Feasible: {result['is_feasible']}")

    # Calculate success rate
    feasible_count = sum(1 for r in results if r['is_feasible'])
    success_rate = feasible_count / len(results)

    print("\n" + "-" * 80)
    print("SUCCESS RATE CALCULATION:")
    print("-" * 80)
    print(f"Total runs: {len(results)}")
    print(f"Feasible runs (final_cv <= 0): {feasible_count}")
    print(f"Success rate: {success_rate:.2%}")
    print(f"\nFormula: success_rate = feasible_runs / total_runs")
    print(f"         {success_rate:.3f} = {feasible_count} / {len(results)}")

    print("\n" + "=" * 80)
    print("TEST PASSED!")
    print("=" * 80)


if __name__ == "__main__":
    test_single_method_measures()
    test_multiple_runs_success_rate()
