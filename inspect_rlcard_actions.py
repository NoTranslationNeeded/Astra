import rlcard
import numpy as np

env = rlcard.make('no-limit-holdem')
print(f"Num Actions: {env.num_actions}")

state, player_id = env.reset()
print(f"Initial State Legal Actions: {state['legal_actions']}")

# Try to perform actions and see what happens
# We need to see the mapping.
# rlcard usually provides 'raw_legal_actions' in state
if 'raw_legal_actions' in state:
    print(f"Raw Legal Actions: {state['raw_legal_actions']}")

# Let's try to map indices to strings if possible
# env.actions is often a list of strings in some envs
if hasattr(env, 'actions'):
    print(f"Env Actions: {env.actions}")

# If not, let's look at the game object
if hasattr(env, 'game'):
    print(f"Game Actions: {env.game.actions}")
    
# Let's try to step with action 2, 3, 4 and see the bet size change?
# But we can't easily see the bet size change without parsing state.

