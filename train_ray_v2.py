import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from pettingzoo.utils import parallel_to_aec
import os

# Import our custom environment
from poker_pettingzoo import PokerPettingZooEnv

def env_creator(config):
    # Create PettingZoo environment
    env = PokerPettingZooEnv()
    return env

def train_ray():
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    
    from gymnasium import spaces
    import numpy as np
    
    # Define spaces
    obs_space = spaces.Box(low=0, high=1, shape=(55,), dtype=np.float32)
    act_space = spaces.Discrete(5) # 5 actions in NLH

    # Configure the algorithm with Independent Policies
    config = (
        PPOConfig()
        .environment(
            env=PettingZooEnv,
            env_config={"env_creator": env_creator}
        )
        .framework("torch")
        .env_runners(num_env_runners=2) # Use 2 parallel workers
        .training(
            model={
                "fcnet_hiddens": [256, 256],
                "use_lstm": True,
                "lstm_cell_size": 256,
                "max_seq_len": 20,
            },
            gamma=0.99,
            lr=0.0003,  # Increased from 0.0001
            train_batch_size=8000,  # Increased from 4000
            entropy_coeff=0.01,  # Added for more exploration
        )
        .multi_agent(
            policies={
                # Independent policies - each player has their own brain
                "player_0": (None, obs_space, act_space, {"lr": 0.0003}),
                "player_1": (None, obs_space, act_space, {"lr": 0.0003}),
            },
            policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id,
            policies_to_train=["player_0", "player_1"],  # Train both
        )
    )
    
    # Run training
    stop = {
        "training_iteration": 100,
    }
    
    print("=" * 80)
    print("Starting Ray/RLlib training with Independent Policies")
    print("Configuration:")
    print("  - Environment: PettingZoo wrapper for RLCard")
    print("  - Algorithm: PPO with LSTM")
    print("  - Policies: Independent (player_0 and player_1)")
    print(f"  - TensorBoard logs: {os.path.abspath('./ray_results')}")
    print("To view TensorBoard:")
    print("  tensorboard --logdir=./ray_results")
    print("  Then open: http://localhost:6006")
    print("=" * 80)
    
    results = tune.run(
        "PPO",
        name="poker_independent_v1",
        config=config.to_dict(),
        stop=stop,
        checkpoint_freq=10,
        storage_path=os.path.abspath("./ray_results"),
        verbose=1,
    )
    
    print("Training complete!")
    ray.shutdown()

if __name__ == "__main__":
    train_ray()
