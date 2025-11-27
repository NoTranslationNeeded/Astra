import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env
import os

# Import our PettingZoo environment
from poker_pettingzoo_v2 import PokerParallelEnv

def env_creator(config):
    """Create PettingZoo environment wrapped for RLlib"""
    env = PokerParallelEnv()
    return env

# Register environment
register_env("poker_pettingzoo", lambda config: ParallelPettingZooEnv(env_creator(config)))

def train_pettingzoo():
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    
    from gymnasium import spaces
    import numpy as np
    
    # Define spaces
    obs_space = spaces.Box(low=0, high=1, shape=(55,), dtype=np.float32)
    act_space = spaces.Discrete(5)

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
    print("🎯 PettingZoo + Independent Policies Training")
    print("")
    print("Configuration:")
    print("  ✅ Environment: PettingZoo ParallelEnv wrapper")
    print("  ✅ Algorithm: PPO with LSTM (256 neurons)")
    print("  ✅ Policies: INDEPENDENT (separate brains)")
    print("  ✅ Learning Rate: 0.0003")
    print("  ✅ Entropy Coeff: 0.01")
    print("  ✅ Batch Size: 8000")
    print("")
    print(f"TensorBoard: {os.path.abspath('./ray_results')}")
    print("  tensorboard --logdir=./ray_results")
    print("  http://localhost:6006")
    print("=" * 80)
    
    results = tune.run(
        "PPO",
        name="poker_pettingzoo_v1",
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
    train_pettingzoo()
