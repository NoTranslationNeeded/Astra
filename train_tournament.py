"""
Tournament Poker Training Script
Train AI on tournament-style poker with escalating blinds and randomized starting stacks
"""
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env
from ray.air.integrations.mlflow import MLflowLoggerCallback
import os
import mlflow
from gymnasium import spaces
import numpy as np

# Import our custom environment
from tournament_pettingzoo import TournamentPokerParallelEnv

def env_creator(config):
    """Create tournament poker environment wrapped for RLlib"""
    env = TournamentPokerParallelEnv(
        starting_chips=100,
        randomize_stacks=True
    )
    return env

# Register environment
register_env("tournament_poker", lambda config: ParallelPettingZooEnv(env_creator(config)))


def train_tournament():
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    
    # Define spaces - 60 dimensions with blind level
    obs_space = spaces.Box(low=0, high=1, shape=(60,), dtype=np.float32)
    act_space = spaces.Discrete(5)
    
    # MLflow setup
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("tournament-poker-ai")

    # Configure PPO with Independent Policies
    config = (
        PPOConfig()
        .environment("tournament_poker")
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
        "training_iteration": 200,
    }
    
    print("=" * 80)
    print("🚀 Training Tournament Poker AI")
    print("=" * 80)
    print("")
    print("Configuration:")
    print("  ✅ Environment: Tournament Poker (Multi-Hand)")
    print("  ✅ Blind Escalation: Every 5 hands")
    print("  ✅ Blind Levels: 1/2 → 2/4 → 3/6 → 5/10 → 10/20 → ...")
    print("  ✅ Starting Stacks: Randomized (80-120 chips)")
    print("  ✅ Algorithm: PPO with LSTM (256 neurons)")
    print("  ✅ Policies: INDEPENDENT (self-play)")
    print("")
    print("Domain Randomization:")
    print("  📊 Each episode: Random stack sizes (40-60 BB)")
    print("  🎲 Smooth generalization across stack depths")
    print("")
    print("Tracking:")
    print(f"  📊 TensorBoard: {os.path.abspath('./ray_results')}")
    print("     → tensorboard --logdir=./ray_results")
    print(f"  📈 MLflow: {os.path.abspath('./mlruns')}")
    print("     → mlflow ui")
    print("=" * 80)
    
    # Run training
    results = tune.run(
        "PPO",
        name="tournament_poker_v1",
        config=config.to_dict(),
        stop=stop,
        checkpoint_freq=10,
        storage_path=os.path.abspath("./ray_results"),
        verbose=1,
        callbacks=[
            MLflowLoggerCallback(
                tracking_uri="file:./mlruns",
                experiment_name="tournament-poker-ai",
                save_artifact=True,
            ),
        ],
    )
    
    print("")
    print("=" * 80)
    print("✅ Training complete!")
    print("")
    print("View results:")
    print("  TensorBoard: tensorboard --logdir=./ray_results")
    print("  MLflow UI:   mlflow ui")
    print("")
    print("Watch games:")
    print("  python watch_tournament.py --checkpoint <path_to_checkpoint>")
    print("=" * 80)
    ray.shutdown()

if __name__ == "__main__":
    train_tournament()
