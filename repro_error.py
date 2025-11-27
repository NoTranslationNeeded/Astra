import os
# Force python implementation
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import ray
from ray import tune
import gymnasium as gym

print("Ray version:", ray.__version__)
try:
    import google.protobuf
    print("Protobuf version:", google.protobuf.__version__)
except ImportError:
    print("Protobuf not found")

print("Initializing Ray...")
ray.init(num_cpus=2)

print("Defining simple env...")
def simple_env(config):
    return gym.make("CartPole-v1")

tune.register_env("simple_env", simple_env)

print("Running simple tune...")
tune.run(
    "PPO",
    config={
        "env": "simple_env",
        "num_workers": 1,
        "framework": "torch",
    },
    stop={"training_iteration": 1}
)
print("Done!")
