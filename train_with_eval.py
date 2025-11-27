import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.tune import Callback
import os
import mlflow
from evaluator import PokerEvaluator

# Import our PettingZoo environment
from poker_pettingzoo_v2 import PokerParallelEnv

def env_creator(config):
    """Create PettingZoo environment wrapped for RLlib"""
    env = PokerParallelEnv()
    return env

# Register environment
register_env("poker_pettingzoo", lambda config: ParallelPettingZooEnv(env_creator(config)))


class EvaluationCallback(Callback):
    """Custom callback for periodic evaluation"""
    
    def __init__(self, eval_interval=10):
        super().__init__()
        self.eval_interval = eval_interval
        self.evaluator = PokerEvaluator()
        self.best_win_rate_random = 0.0
        self.best_win_rate_rule = 0.0
    
    def on_trial_result(self, iteration, trials, trial, result, **info):
        """Called after each training iteration"""
        
        # Only evaluate every N iterations
        if iteration % self.eval_interval != 0:
            return
        
        # Get the trained policy
        trainer = trial.runner
        if trainer is None:
            return
        
        try:
            # Get policy for player_0 (we'll evaluate this one)
            policy = trainer.get_policy("player_0")
            
            print(f"\n{'🎯 EVALUATION at Iteration {iteration} ':#^80}\n")
            
            # Evaluate against Random Bot
            print("Evaluating vs Random Bot...")
            metrics_random = self.evaluator.evaluate_agent(
                policy,
                opponent_name="random",
                num_games=50,
                greedy=True
            )
            self.evaluator.log_evaluation_results(metrics_random, iteration)
            
            # Evaluate against Rule-Based Bot
            print("Evaluating vs Rule-Based Bot...")
            metrics_rule = self.evaluator.evaluate_agent(
                policy,
                opponent_name="rule_based",
                num_games=50,
                greedy=True
            )
            self.evaluator.log_evaluation_results(metrics_rule, iteration)
            
            # Check if this is the best model
            if metrics_random["win_rate"] > self.best_win_rate_random:
                self.best_win_rate_random = metrics_random["win_rate"]
                print(f"🏆 NEW BEST vs Random: {self.best_win_rate_random:.1%}")
            
            if metrics_rule["win_rate"] > self.best_win_rate_rule:
                self.best_win_rate_rule = metrics_rule["win_rate"]
                print(f"🏆 NEW BEST vs Rule-Based: {self.best_win_rate_rule:.1%}")
                
                # Save best model
                checkpoint_dir = trainer.save()
                mlflow.log_param("best_checkpoint", checkpoint_dir)
                mlflow.set_tag("best_win_rate_rule", f"{self.best_win_rate_rule:.1%}")
                print(f"💾 Best model saved: {checkpoint_dir}")
            
        except Exception as e:
            print(f"⚠️  Evaluation failed: {e}")


def train_with_evaluation():
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    
    from gymnasium import spaces
    import numpy as np
    
    # Define spaces
    obs_space = spaces.Box(low=0, high=1, shape=(55,), dtype=np.float32)
    act_space = spaces.Discrete(5)
    
    # MLflow setup
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("poker-ai-with-evaluation")

    # Configure PPO with Independent Policies
    config = (
        PPOConfig()
        .environment("poker_pettingzoo")
        .framework("torch")
        .env_runners(num_env_runners=2)
        .training(
            model={
                "fcnet_hiddens": [256, 256],
                "use_lstm": True,
                "lstm_cell_size": 256,
                "max_seq_len": 20,
            },
            gamma=0.99,
            lr=0.0003,
            train_batch_size=8000,
            entropy_coeff=0.01,
        )
        .multi_agent(
            policies={
                "player_0": (None, obs_space, act_space, {"lr": 0.0003}),
                "player_1": (None, obs_space, act_space, {"lr": 0.0003}),
            },
            policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id,
            policies_to_train=["player_0", "player_1"],
        )
    )
    
    # Training configuration
    stop = {
        "training_iteration": 100,
    }
    
    print("=" * 80)
    print("🚀 Training with Evaluation System")
    print("")
    print("Configuration:")
    print("  ✅ Environment: PettingZoo ParallelEnv")
    print("  ✅ Algorithm: PPO with LSTM (256 neurons)")
    print("  ✅ Policies: INDEPENDENT (separate brains)")
    print("  ✅ Evaluation: Every 10 iterations")
    print("    - vs Random Bot (50 games)")
    print("    - vs Rule-Based Bot (50 games)")
    print("")
    print("Tracking:")
    print(f"  📊 TensorBoard: {os.path.abspath('./ray_results')}")
    print("     → tensorboard --logdir=./ray_results")
    print("  📈 MLflow: {os.path.abspath('./mlruns')}")
    print("     → mlflow ui")
    print("=" * 80)
    
    # Run training with evaluation callback
    results = tune.run(
        "PPO",
        name="poker_eval_v1",
        config=config.to_dict(),
        stop=stop,
        checkpoint_freq=10,
        storage_path=os.path.abspath("./ray_results"),
        verbose=1,
        callbacks=[
            MLflowLoggerCallback(
                tracking_uri="file:./mlruns",
                experiment_name="poker-ai-with-evaluation",
                save_artifact=True,
            ),
            EvaluationCallback(eval_interval=10),
        ],
    )
    
    print("=" * 80)
    print("✅ Training complete!")
    print("")
    print("View results:")
    print("  TensorBoard: tensorboard --logdir=./ray_results")
    print("  MLflow UI:   mlflow ui")
    print("=" * 80)
    ray.shutdown()

if __name__ == "__main__":
    train_with_evaluation()
