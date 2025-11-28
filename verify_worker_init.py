import ray
from tournament_pettingzoo import TournamentPokerParallelEnv
import time

@ray.remote
def test_worker():
    print("Initializing Env...")
    try:
        # Test TournamentPokerParallelEnv directly
        env = TournamentPokerParallelEnv(
            starting_chips=100,
            randomize_stacks=True
        )
        print("Env Initialized. Resetting...")
        obs, infos = env.reset()
        print("Reset Complete. Stepping...")
        # Step with random action for player_0
        action = {list(obs.keys())[0]: 1} # Check/Call
        obs, rewards, terms, truncs, infos = env.step(action)
        print("Step Complete.")
        return True
    except Exception as e:
        print(f"Worker Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    ray.init()
    print("Ray Initialized. Launching worker...")
    future = test_worker.remote()
    result = ray.get(future)
    print(f"Worker Result: {result}")
    ray.shutdown()
