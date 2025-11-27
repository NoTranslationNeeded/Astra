"""
Standalone evaluation script for poker AI
Loads a trained model and evaluates against opponents
"""
import argparse
import os
import ray
from ray.rllib.algorithms.ppo import PPO
from evaluator import PokerEvaluator
import mlflow

def evaluate_checkpoint(checkpoint_path, num_games=100):
    """
    Evaluate a trained model from checkpoint
    
    Args:
        checkpoint_path: Path to Ray checkpoint directory
        num_games: Number of games to play per opponent
    """
    print("=" * 80)
    print("🎯 Poker AI Evaluation")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Games per opponent: {num_games}")
    print("=" * 80)
    
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    
    try:
        # Load the trained algorithm
        print("\n📂 Loading checkpoint...")
        algo = PPO.from_checkpoint(checkpoint_path)
        
        # Get the trained policy
        policy = algo.get_policy("player_0")
        print("✅ Checkpoint loaded successfully\n")
        
        # Initialize evaluator
        evaluator = PokerEvaluator()
        
        # Evaluate against Random Bot
        print("🎲 Evaluating vs Random Bot")
        print("-" * 80)
        metrics_random = evaluator.evaluate_agent(
            policy,
            opponent_name="random",
            num_games=num_games,
            greedy=True
        )
        evaluator.log_evaluation_results(metrics_random)
        
        # Evaluate against Rule-Based Bot
        print("\n🧠 Evaluating vs Rule-Based Bot")
        print("-" * 80)
        metrics_rule = evaluator.evaluate_agent(
            policy,
            opponent_name="rule_based",
            num_games=num_games,
            greedy=True
        )
        evaluator.log_evaluation_results(metrics_rule)
        
        # Summary
        print("\n" + "=" * 80)
        print("📊 EVALUATION SUMMARY")
        print("=" * 80)
        print(f"vs Random Bot:      {metrics_random['win_rate']:.1%} win rate")
        print(f"vs Rule-Based Bot:  {metrics_rule['win_rate']:.1%} win rate")
        print("=" * 80)
        
        # Save results to file
        results_file = "evaluation_results.txt"
        with open(results_file, "w") as f:
            f.write(f"Checkpoint: {checkpoint_path}\n")
            f.write(f"Games per opponent: {num_games}\n\n")
            f.write(f"vs Random Bot:\n")
            f.write(f"  Win Rate: {metrics_random['win_rate']:.1%}\n")
            f.write(f"  Avg Reward: {metrics_random['avg_reward']:.2f}\n")
            f.write(f"  Avg Game Length: {metrics_random['avg_game_length']:.1f}\n\n")
            f.write(f"vs Rule-Based Bot:\n")
            f.write(f"  Win Rate: {metrics_rule['win_rate']:.1%}\n")
            f.write(f"  Avg Reward: {metrics_rule['avg_reward']:.2f}\n")
            f.write(f"  Avg Game Length: {metrics_rule['avg_game_length']:.1f}\n")
        
        print(f"\n💾 Results saved to: {results_file}\n")
        
        return metrics_random, metrics_rule
        
    finally:
        ray.shutdown()


def find_latest_checkpoint(base_dir="./ray_results"):
    """Find the most recent checkpoint in ray_results"""
    checkpoints = []
    
    for root, dirs, files in os.walk(base_dir):
        for dir_name in dirs:
            if dir_name.startswith("checkpoint_"):
                checkpoint_path = os.path.join(root, dir_name)
                # Get checkpoint number
                try:
                    checkpoint_num = int(dir_name.split("_")[1])
                    checkpoints.append((checkpoint_num, checkpoint_path))
                except:
                    pass
    
    if not checkpoints:
        return None
    
    # Return path with highest checkpoint number
    checkpoints.sort(reverse=True)
    return checkpoints[0][1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained poker AI")
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to checkpoint directory (default: find latest)"
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=100,
        help="Number of games per opponent (default: 100)"
    )
    
    args = parser.parse_args()
    
    # Determine checkpoint path
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        print("🔍 Searching for latest checkpoint...")
        checkpoint_path = find_latest_checkpoint()
        if checkpoint_path is None:
            print("❌ No checkpoints found in ./ray_results")
            print("   Please specify --checkpoint manually")
            exit(1)
        print(f"✅ Found: {checkpoint_path}\n")
    
    # Run evaluation
    evaluate_checkpoint(checkpoint_path, args.num_games)
