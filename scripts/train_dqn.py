"""
DQN Training Script for Adaptive BCHM Selection

Features:
- Ray parallelization for efficient training (12 workers on M1 Mac)
- Train/test split for proper generalization evaluation
- Adaptive epsilon decay based on episode count
- Boundary violation buffer (stores experiences with prob_infeas_bounds > 0.15)
- Real-time method bias detection
"""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import ray

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmarks.cec2006 import get_problem
from src.rl.rl_pso import RL_PSO, METHOD_POOL_CUSTOM5
from src.rl.dqn import DQNAgent, extract_state, compute_reward

# Train/Test Split for CEC2006 problems
TRAINING_PROBLEMS = [4, 6, 14, 1, 16, 7, 10, 11, 18, 3]  # 10 problems for training
EVALUATION_PROBLEMS = [2, 5, 8, 9, 13, 15, 17, 19, 23, 24]  # 10 problems for evaluation

# All available CEC2006 problems (exclude G12, G20, G21, G22)
ALL_PROBLEMS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                13, 14, 15, 16, 17, 18, 19, 23, 24]


@ray.remote
def collect_episode_experiences(episode_num, problem_id, network_state_dict,
                                epsilon, max_evaluations, random_seed):
    """
    Ray remote function: Collect experiences from ONE episode

    Runs on a separate worker process. Does NOT train the DQN agent.

    Args:
        episode_num: Episode number
        problem_id: CEC2006 problem ID
        network_state_dict: DQN network weights
        epsilon: Current epsilon value
        max_evaluations: Evaluation budget
        random_seed: Random seed for reproducibility

    Returns:
        dict: Episode results with experiences
    """
    try:
        import logging
        logger = logging.getLogger(f'Worker-{episode_num}')

        # Set random seed for this worker
        if random_seed is not None:
            np.random.seed(random_seed)
            logger.info(f"Episode {episode_num}: Starting on problem G{problem_id:02d} with seed {random_seed}")

        # Create problem
        problem = get_problem(problem_id)
        logger.info(f"Episode {episode_num}: Problem G{problem_id:02d} created (dim={problem.dimension})")

        # Create DQN agent in EVALUATION mode (no training)
        dqn_agent = DQNAgent(
            state_dim=9,
            action_dim=5,
            learning_rate=0.001,
            gamma=0.95,
            epsilon_start=epsilon * np.random.uniform(0.95, 1.05),  # ±5% variation
            epsilon_end=epsilon,
            epsilon_decay=1.0,  # No decay in worker
            buffer_capacity=1000,  # Small buffer (not used)
            batch_size=64,
            device='cpu'
        )

        # Load master network weights
        dqn_agent.q_network.load_state_dict(network_state_dict)
        dqn_agent.target_network.load_state_dict(network_state_dict)

        # Create RL-PSO optimizer
        optimizer = RL_PSO(
            problem=problem,
            population_size=100,
            max_evaluations=max_evaluations,
            random_seed=None,  # Each episode gets different seed
            dqn_agent=dqn_agent,
            training_mode=False,  # CRITICAL: No training in worker
            method_pool=METHOD_POOL_CUSTOM5
        )

        # Collect experiences
        experiences = []

        while not optimizer.is_done():
            # Get state before step
            old_pso_state = optimizer._get_pso_state()
            max_generations = optimizer.max_evaluations / len(optimizer.swarm)

            old_state = extract_state(
                old_pso_state,
                optimizer.generation,
                max_generations,
                optimizer.previous_method_id
            )

            # PSO step (agent selects action but doesn't train)
            optimizer.step()

            # Get state after step
            new_pso_state = optimizer._get_pso_state()
            new_state = extract_state(
                new_pso_state,
                optimizer.generation,
                max_generations,
                optimizer.previous_method_id
            )

            # Compute reward
            reward = compute_reward(old_pso_state, new_pso_state)

            # Store experience
            done = optimizer.is_done()
            prob_infeas_bounds = new_pso_state.get('prob_infeas_bounds', 0.0)

            experiences.append({
                'state': old_state.tolist(),  # Convert to list for serialization
                'action': optimizer.previous_method_id,
                'reward': float(reward),
                'next_state': new_state.tolist(),
                'done': bool(done),
                'prob_infeas_bounds': float(prob_infeas_bounds)
            })

        # Get final stats
        stats = optimizer.get_stats()

        # Calculate action counts for method distribution tracking
        action_counts = [0, 0, 0, 0, 0]
        for exp in experiences:
            action = exp['action']
            action_counts[action] += 1

        logger.info(f"Episode {episode_num}: Completed - Fitness={stats['gbest_fitness']:.6e}, "
                    f"Feasible={stats['gbest_is_feasible']}, Experiences={len(experiences)}, "
                    f"Generations={stats['generation']}, NFE={stats['nfe']}")

        return {
            'success': True,
            'episode': episode_num,
            'problem': problem_id,
            'experiences': experiences,
            'num_experiences': len(experiences),
            'final_fitness': stats['gbest_fitness'],
            'is_feasible': stats['gbest_is_feasible'],
            'generation': stats['generation'],
            'nfe': stats['nfe'],
            'action_counts': action_counts  # Track method usage
        }

    except Exception as e:
        # Handle errors gracefully
        import logging
        logger = logging.getLogger(f'Worker-{episode_num}')
        logger.error(f"Episode {episode_num}: FAILED with error: {str(e)}")
        return {
            'success': False,
            'episode': episode_num,
            'problem': problem_id,
            'error': str(e)
        }


def train_dqn(num_episodes=300, num_workers=12, max_evaluations=100000,
              save_interval=50, output_dir='results/experiments/dqn',
              random_seed=42, use_train_split=False):
    """
    Train DQN with Ray parallelization

    Args:
        num_episodes: Total training episodes (default: 300)
        num_workers: Number of parallel workers (default: 12 for M1 Pro)
        max_evaluations: Budget per episode (default: 100k)
        save_interval: Save checkpoint every N episodes (default: 50)
        output_dir: Output directory
        random_seed: Global random seed (default: 42)
        use_train_split: If True, use only TRAINING_PROBLEMS; otherwise use ALL_PROBLEMS
    """
    # Set global random seed
    np.random.seed(random_seed)
    logging.info(f"Global random seed set to {random_seed}")

    # Select problem set
    problem_set = TRAINING_PROBLEMS if use_train_split else ALL_PROBLEMS
    problem_set_name = "TRAINING (10 problems)" if use_train_split else "ALL (20 problems)"

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / f"dqn_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory created: {run_dir}")

    print("=" * 80)
    print("DQN TRAINING WITH RAY PARALLELIZATION")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Episodes: {num_episodes}")
    print(f"  Workers: {num_workers}")
    print(f"  Max Evaluations: {max_evaluations}")
    print(f"  Problem Set: {problem_set_name}")
    if use_train_split:
        print(f"  Training Problems: {TRAINING_PROBLEMS}")
    print(f"  Random Seed: {random_seed}")
    print(f"  Output: {run_dir}")
    print()

    # Initialize Ray
    print("Initializing Ray...")
    logging.info(f"Initializing Ray with {num_workers} workers...")
    ray.init(num_cpus=num_workers, ignore_reinit_error=True)
    print(f"Ray initialized with {num_workers} CPUs")
    logging.info(f"Ray initialized successfully with {num_workers} CPUs")
    print()

    # Calculate epsilon decay based on num_episodes
    # Formula: epsilon_end = epsilon_start * decay^episodes
    # Solving for decay: decay = (epsilon_end / epsilon_start)^(1/episodes)
    epsilon_start = 1.0
    epsilon_end = 0.2
    epsilon_decay = (epsilon_end / epsilon_start) ** (1.0 / num_episodes)

    print(f"Epsilon Decay Strategy:")
    print(f"  Start: {epsilon_start:.3f}")
    print(f"  End: {epsilon_end:.3f}")
    print(f"  Decay factor: {epsilon_decay:.6f}")
    print(f"  At episode {num_episodes // 4}: ~{epsilon_start * (epsilon_decay ** (num_episodes // 4)):.3f}")
    print(f"  At episode {num_episodes // 2}: ~{epsilon_start * (epsilon_decay ** (num_episodes // 2)):.3f}")
    print(f"  At episode {3 * num_episodes // 4}: ~{epsilon_start * (epsilon_decay ** (3 * num_episodes // 4)):.3f}")
    print()

    # Initialize master DQN agent
    master_dqn = DQNAgent(
        state_dim=9,
        action_dim=5,
        learning_rate=0.001,
        gamma=0.95,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        buffer_capacity=50000,
        batch_size=128,
        target_update_freq=10,
        device='cpu'
    )

    logging.info(f"DQN Agent initialized with BoundaryViolationBuffer: "
                 f"capacity={master_dqn.memory.capacity} (stores ONLY prob_infeas_bounds > 0.15)")

    training_history = []
    start_time = time.time()

    # Track method selection distribution across all episodes
    method_names = ['Random&Deb', 'ExpTarget&RandB', 'ExpBest&Deb', 'MidBest&Deb', 'Boundary&RandB']
    global_action_counts = np.zeros(5, dtype=int)

    logging.info("Training started")

    # Training loop in batches
    for batch_start in range(0, num_episodes, num_workers):
        batch_end = min(batch_start + num_workers, num_episodes)
        batch_size = batch_end - batch_start

        print(f"\n{'=' * 80}")
        print(f"Batch: Episodes {batch_start + 1}-{batch_end} | Epsilon: {master_dqn.epsilon:.3f}")
        print(f"{'=' * 80}")
        logging.info(
            f"Starting batch {batch_start + 1}-{batch_end} (size={batch_size}, epsilon={master_dqn.epsilon:.3f})")

        # Get current network state (shared across workers)
        network_state = master_dqn.q_network.state_dict()
        current_epsilon = master_dqn.epsilon

        # Launch parallel episodes with Ray
        print(f"Launching {batch_size} workers...")
        logging.info(f"Launching {batch_size} Ray workers in parallel...")

        # Create futures for parallel execution
        futures = [
            collect_episode_experiences.remote(
                batch_start + i,
                np.random.choice(problem_set),  # Use selected problem set
                network_state,
                current_epsilon,
                max_evaluations,
                random_seed + batch_start + i if random_seed else None
            )
            for i in range(batch_size)
        ]

        # Wait for all workers to complete
        batch_start_time = time.time()
        results = ray.get(futures)
        batch_elapsed = time.time() - batch_start_time
        logging.info(f"Batch completed in {batch_elapsed:.2f}s ({batch_elapsed / batch_size:.2f}s per episode)")

        # Process results and train
        print(f"\nProcessing {len(results)} episodes...")
        logging.info(f"Processing {len(results)} episodes and training DQN...")

        for result in results:
            if not result['success']:
                print(f"  ⚠️  Episode {result['episode'] + 1} FAILED: {result['error']}")
                continue

            # Add experiences to master buffer
            for exp in result['experiences']:
                # Convert back to numpy arrays
                state = np.array(exp['state'], dtype=np.float32)
                next_state = np.array(exp['next_state'], dtype=np.float32)

                master_dqn.store_experience(
                    state,
                    exp['action'],
                    exp['reward'],
                    next_state,
                    exp['done'],
                    exp['prob_infeas_bounds']
                )

            # Train DQN on new experiences
            num_training_steps = max(10, result['num_experiences'] // 10)
            losses = []

            train_start = time.time()
            for _ in range(num_training_steps):
                loss = master_dqn.train_step()
                if loss is not None:
                    losses.append(loss)

            avg_loss = np.mean(losses) if losses else 0.0
            train_elapsed = time.time() - train_start
            logging.info(
                f"Episode {result['episode'] + 1}: Trained for {num_training_steps} steps in {train_elapsed:.2f}s, avg_loss={avg_loss:.4f}")

            # Decay epsilon
            master_dqn.decay_epsilon()
            master_dqn.episodes += 1

            # Update target network
            if master_dqn.episodes % master_dqn.target_update_freq == 0:
                master_dqn.update_target_network()
                logging.info(
                    f"Episode {result['episode'] + 1}: Target network updated (freq={master_dqn.target_update_freq})")

            # Get buffer stats
            buffer_stats = master_dqn.memory.get_stats()

            # Track action distribution from this episode
            if 'action_counts' in result:
                action_counts = np.array(result['action_counts'])
                global_action_counts += action_counts

            # Calculate ETA
            elapsed = time.time() - start_time
            episodes_done = master_dqn.episodes
            eps_per_sec = episodes_done / elapsed if elapsed > 0 else 0
            eta_seconds = (num_episodes - episodes_done) / eps_per_sec if eps_per_sec > 0 else 0
            eta_hours = eta_seconds / 3600

            print(f"  Ep {result['episode'] + 1:4d}/{num_episodes} | "
                  f"G{result['problem']:02d} | "
                  f"Fit: {result['final_fitness']:12.6e} | "
                  f"Feas: {result['is_feasible']} | "
                  f"Exp: {result['num_experiences']:4d} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"ε: {master_dqn.epsilon:.3f} | "
                  f"Buf: {buffer_stats['total_size']:5d} | "
                  f"ETA: {eta_hours:.1f}h")

            # Every 50 episodes, check for method bias
            if (result['episode'] + 1) % 50 == 0:
                total_actions = global_action_counts.sum()
                if total_actions > 0:
                    action_freqs = global_action_counts / total_actions
                    max_freq = action_freqs.max()
                    max_idx = action_freqs.argmax()
                    dominant_method = method_names[max_idx]

                    # Warning if one method is used > 70%
                    if max_freq > 0.70:
                        print(f"\n  ⚠️  METHOD BIAS DETECTED (Ep {result['episode'] + 1}):")
                        print(f"      {dominant_method} used {max_freq * 100:.1f}% of the time!")
                        print(f"      Distribution: ", end="")
                        for i, (name, freq) in enumerate(zip(method_names, action_freqs)):
                            print(f"{name}:{freq * 100:.0f}% ", end="")
                        print("\n")
                    else:
                        # Log distribution normally
                        logging.info(f"  Method distribution (Ep {result['episode'] + 1}): " +
                                     " | ".join([f"{name}:{freq * 100:.1f}%" for name, freq in
                                                 zip(method_names, action_freqs)]))

            # Save history
            training_history.append({
                'episode': int(result['episode'] + 1),
                'problem': f"G{result['problem']:02d}",
                'final_fitness': float(result['final_fitness']),
                'is_feasible': bool(result['is_feasible']),
                'generations': int(result['generation']),
                'nfe': int(result['nfe']),
                'num_experiences': int(result['num_experiences']),
                'epsilon': float(master_dqn.epsilon),
                'buffer_size': int(buffer_stats['total_size']),
                'avg_loss': float(avg_loss)
            })

        # Save checkpoint
        if batch_end % save_interval == 0 or batch_end == num_episodes:
            checkpoint_path = run_dir / f"dqn_checkpoint_ep{batch_end}.pth"
            master_dqn.save(checkpoint_path)
            print(f"\n  → Checkpoint saved: {checkpoint_path}")
            logging.info(f"Checkpoint saved at episode {batch_end}: {checkpoint_path}")

            # Save history
            history_path = run_dir / "training_history.json"
            with open(history_path, 'w') as f:
                json.dump(training_history, f, indent=2)
            logging.info(f"Training history saved: {history_path}")

    # Save final model
    final_model_path = run_dir / "dqn_final.pth"
    master_dqn.save(final_model_path)
    logging.info(f"Final model saved: {final_model_path}")

    # Save final history
    history_path = run_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    logging.info(f"Final training history saved: {history_path}")

    # Save config
    final_stats = master_dqn.memory.get_stats()
    total_time = time.time() - start_time

    config = {
        'num_episodes': num_episodes,
        'num_workers': num_workers,
        'max_evaluations': max_evaluations,
        'random_seed': random_seed,
        'use_train_split': use_train_split,
        'problem_set': problem_set.copy() if use_train_split else 'ALL',
        'epsilon_start': epsilon_start,
        'epsilon_end': epsilon_end,
        'epsilon_decay': epsilon_decay,
        'final_epsilon': master_dqn.epsilon,
        'final_buffer_size': final_stats['total_size'],
        'total_time_hours': total_time / 3600,
        'timestamp': timestamp,
        'platform': 'M1 Mac (Apple Silicon)'
    }

    config_path = run_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logging.info(f"Configuration saved: {config_path}")

    # Shutdown Ray
    logging.info("Shutting down Ray...")
    ray.shutdown()
    logging.info("Ray shutdown complete")

    # Print summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Total episodes: {num_episodes}")
    print(f"Total time: {total_time / 3600:.2f} hours")
    print(f"Final epsilon: {master_dqn.epsilon:.3f}")
    print(f"Final buffer size: {final_stats['total_size']}")
    print(f"Output: {run_dir}")
    print("=" * 80)

    return training_history


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Train DQN with Ray parallelization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (10 episodes, 4 workers)
  python scripts/train_dqn.py --episodes 10 --workers 4 --max-evals 10000

  # Standard training (300 episodes, 12 workers)
  python scripts/train_dqn.py --episodes 300 --workers 12 --max-evals 100000

  # Full training with train/test split (5000 episodes)
  python scripts/train_dqn.py --episodes 5000 --workers 12 --train-split

  # Long training with checkpoints every 500 episodes
  python scripts/train_dqn.py --episodes 5000 --workers 12 --save-interval 500
        """
    )

    parser.add_argument('--episodes', type=int, default=300,
                        help='Number of training episodes (default: 300)')
    parser.add_argument('--workers', type=int, default=12,
                        help='Number of Ray workers (default: 12 for M1 Pro)')
    parser.add_argument('--max-evals', type=int, default=100000,
                        help='Max evaluations per episode (default: 100000)')
    parser.add_argument('--save-interval', type=int, default=50,
                        help='Save checkpoint every N episodes (default: 50)')
    parser.add_argument('--output-dir', type=str,
                        default='results/experiments/dqn',
                        help='Output directory (default: results/experiments/dqn)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--train-split', action='store_true',
                        help='Use train/test split (train on 10 problems, evaluate on other 10)')

    args = parser.parse_args()

    train_dqn(
        num_episodes=args.episodes,
        num_workers=args.workers,
        max_evaluations=args.max_evals,
        save_interval=args.save_interval,
        output_dir=args.output_dir,
        random_seed=args.seed,
        use_train_split=args.train_split
    )


if __name__ == '__main__':
    main()
