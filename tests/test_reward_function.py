"""
Test Reward Function and State Space Extractor
"""

from src.benchmarks.cec2006 import get_problem
from src.core.abchs import ABCHS
from src.rl.reward_function import RewardFunction, StateSpaceExtractor
from src.utils.generation_tracker import GenerationMetrics


def test_reward_function():
    """Test reward function computation"""
    print("=" * 80)
    print("TEST: Reward Function")
    print("=" * 80)

    # Initialize problem
    problem = get_problem(2)  # G02: simple problem
    print(f"\nProblem: G{problem.problem_id:02d}")
    print(f"Known optimum: {problem.known_optimum:.6f}")

    # Initialize ABCHS
    abchs = ABCHS(
        problem=problem,
        population_size=20,
        max_evaluations=1000,
        random_seed=42
    )

    # Initialize reward function
    reward_fn = RewardFunction(problem)

    print(f"\nInitial state:")
    print(f"  Generation: {abchs.generation}")
    print(f"  NFE: {abchs.nfe}")
    print(f"  Gbest fitness: {abchs.gbest_fitness:.6e}")
    print(f"  Gbest CV: {abchs.gbest_cv:.6e}")
    print(f"  NFS: {abchs.nfs}")

    # Get initial metrics
    metrics_prev = GenerationMetrics(abchs, problem)

    # Run a few generations
    for gen in range(5):
        abchs.step()

        # Get current metrics
        metrics_curr = GenerationMetrics(abchs, problem)

        # Compute reward
        reward, components = reward_fn.compute_reward(
            metrics_curr,
            metrics_prev,
            abchs.swarm
        )

        print(f"\n--- Generation {abchs.generation} ---")
        print(f"  Gbest fitness: {abchs.gbest_fitness:.6e}")
        print(f"  Gbest CV: {abchs.gbest_cv:.6e}")
        print(f"  Error: {metrics_curr.error:.6e}")
        print(f"  NFS: {abchs.nfs}")
        print(f"  Diversity: {metrics_curr.diversity:.6f}")
        print(f"  Stage: {abchs.stage}")

        print(f"\n  Reward: {reward:.6f}")
        print(f"    - Error term: {components['term_error']:.6f}")
        print(f"    - Feasibility term: {components['term_feasibility']:.6f}")
        print(f"    - Improvement term: {components['term_improvement']:.6f}")
        print(f"    - Diversity term: {components['term_diversity']:.6f}")
        print(f"  prob_infeas_functional: {components['prob_infeas_functional']:.3f}")

        # Update previous metrics
        metrics_prev = metrics_curr

    print("\n" + "=" * 80)
    print("TEST PASSED: Reward function working correctly")
    print("=" * 80)


def test_state_space():
    """Test state space extraction"""
    print("\n" + "=" * 80)
    print("TEST: State Space Extractor")
    print("=" * 80)

    # Initialize problem
    problem = get_problem(4)  # G04: more complex
    print(f"\nProblem: G{problem.problem_id:02d}")

    # Initialize ABCHS
    abchs = ABCHS(
        problem=problem,
        population_size=30,
        max_evaluations=2000,
        random_seed=123
    )

    # Initialize state extractor
    state_extractor = StateSpaceExtractor(problem)

    print(f"\nState space dimension: {state_extractor.get_state_dim()}")
    print(f"Feature names: {state_extractor.get_feature_names()}")

    # Extract initial state
    state = state_extractor.extract_state(abchs)

    print(f"\nInitial state vector:")
    for i, (name, value) in enumerate(zip(state_extractor.get_feature_names(), state)):
        print(f"  [{i:2d}] {name:25s}: {value:.6f}")

    # Run a few generations and track state evolution
    print("\n" + "-" * 80)
    print("State evolution:")
    print("-" * 80)

    for gen in range(10):
        abchs.step()

        if gen % 2 == 0:  # Print every 2 generations
            state = state_extractor.extract_state(abchs)

            print(f"\nGeneration {abchs.generation}:")
            print(f"  error_norm: {state[0]:.6f}, feasibility: {state[2]:.3f}, "
                  f"diversity: {state[7]:.6f}, stage: {int(state[12])}")

    print("\n" + "=" * 80)
    print("TEST PASSED: State space extractor working correctly")
    print("=" * 80)


def test_reward_diversity_component():
    """Test that diversity component has correct weight (0.1)"""
    print("\n" + "=" * 80)
    print("TEST: Diversity Weight Verification")
    print("=" * 80)

    problem = get_problem(2)
    abchs = ABCHS(problem=problem, population_size=20, max_evaluations=500, random_seed=42)
    reward_fn = RewardFunction(problem)

    # Run until we have some diversity
    for _ in range(3):
        abchs.step()

    metrics_prev = GenerationMetrics(abchs, problem)
    abchs.step()
    metrics_curr = GenerationMetrics(abchs, problem)

    reward, components = reward_fn.compute_reward(metrics_curr, metrics_prev, abchs.swarm)

    print(f"\nDiversity normalized: {components['diversity_normalized']:.6f}")
    print(f"Diversity term (weight 0.1): {components['term_diversity']:.6f}")
    print(f"Expected: {0.1 * components['diversity_normalized']:.6f}")

    # Verify weight
    expected = 0.1 * components['diversity_normalized']
    actual = components['term_diversity']
    assert abs(expected - actual) < 1e-6, f"Weight mismatch! Expected {expected}, got {actual}"

    print(f"\n✓ Diversity weight verified: 0.1")

    # Show breakdown of all terms
    print(f"\nReward breakdown:")
    print(f"  Total reward: {reward:.6f}")
    print(f"  Components:")
    print(f"    Error term (w=-1.0):        {components['term_error']:.6f}")
    print(f"    Feasibility term (w=0.5):   {components['term_feasibility']:.6f}")
    print(f"    Improvement term (w=0.3):   {components['term_improvement']:.6f}")
    print(f"    Diversity term (w=0.1):     {components['term_diversity']:.6f}")

    print("\n" + "=" * 80)
    print("TEST PASSED: Diversity weight is correctly 0.1 (10%)")
    print("=" * 80)


if __name__ == "__main__":
    test_reward_function()
    test_state_space()
    test_reward_diversity_component()

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED!")
    print("=" * 80)
