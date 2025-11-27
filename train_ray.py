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
    # On Windows, local_mode=True might be needed if there are worker issues, 
    # but let's try standard mode first.
    ray.init()
    
    from gymnasium import spaces
    import numpy as np
    
    # Define spaces
    obs_space = spaces.Box(low=0, high=1, shape=(55,), dtype=np.float32)
    act_space = spaces.Discrete(5) # 5 actions in NLH

    # Configure the algorithm
    config = (
        PPOConfig()
        .environment("poker_env")
        .framework("torch")
        .env_runners(num_env_runners=2) # Use 2 parallel workers (adjust based on CPU)
        .training(
            model={
                "fcnet_hiddens": [256, 256],
                "use_lstm": True,
                "lstm_cell_size": 256,
                "max_seq_len": 20,
            },
            gamma=0.99,
            lr=0.0001,
            train_batch_size=4000,
        )
        .multi_agent(
            policies={
                "player_0": (None, obs_space, act_space, {}),
                "player_1": (None, obs_space, act_space, {}),
            },
            policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id,
        )
    )
    
    # Run training
    stop = {
        "training_iteration": 100, # Run for 100 iterations (each iteration is many episodes)
        # "episode_reward_mean": 100, # Stop if reward is high enough
    }
    
    
    print("=" * 80)
    print("Starting Ray/RLlib training with TensorBoard logging...")
    print(f"TensorBoard logs will be saved to: {os.path.abspath('./ray_results')}")
    print("To view TensorBoard, run in another terminal:")
    print("  tensorboard --logdir=./ray_results")
    print("  Then open: http://localhost:6006")
    print("=" * 80)
    
    results = tune.run(
        "PPO",
        config=config.to_dict(),
        stop=stop,
        checkpoint_freq=10,
        storage_path=os.path.abspath("./ray_results"),
        verbose=1,  # Show progress
    )
    
    print("Training complete!")
    ray.shutdown()

if __name__ == "__main__":
    train_ray()
