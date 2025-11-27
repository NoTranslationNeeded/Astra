import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from poker_rllib import PokerMultiAgentEnv
import os

# Register the environment
def env_creator(config):
    return PokerMultiAgentEnv(config)

register_env("poker_env", env_creator)

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
        .environment("poker_env")
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
            policies_to_train=["player_0", "player_1"],  # Train both independently
        )
    )
    
    # Run training
    stop = {
        "training_iteration": 100,
    }
    
    print("=" * 80)
    print("🚀 Starting Ray/RLlib Training with Independent Policies")
    print("")
    print("Configuration:")
    print("  ✅ Algorithm: PPO with LSTM (256 neurons)")
    print("  ✅ Policies: INDEPENDENT (player_0 and player_1 have separate brains)")
    print("  ✅ Learning Rate: 0.0003 (3x faster)")
    print("  ✅ Entropy Coeff: 0.01 (more exploration)")
    print("  ✅ Batch Size: 8000 (2x larger)")
    print("")
    print(f"TensorBoard logs: {os.path.abspath('./ray_results')}")
    print("")
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
    
    print("=" * 80)
    print("✅ Training complete!")
    print("=" * 80)
    ray.shutdown()

if __name__ == "__main__":
    train_ray()
