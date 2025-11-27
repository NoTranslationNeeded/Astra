import gymnasium as gym
from gymnasium import spaces
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from poker_env import PokerEnv

class PokerMultiAgentEnv(MultiAgentEnv):
    def __init__(self, config=None):
        super().__init__()
        self.env = PokerEnv()
        
        self._agent_ids = {"player_0", "player_1"}
        
        # Define spaces for each agent
        self.observation_space = spaces.Dict({
            "player_0": spaces.Box(low=0, high=1, shape=(55,), dtype=np.float32),
            "player_1": spaces.Box(low=0, high=1, shape=(55,), dtype=np.float32)
        })
        
        self.action_space = spaces.Dict({
            "player_0": spaces.Discrete(self.env.action_space_size),
            "player_1": spaces.Discrete(self.env.action_space_size)
        })
        
    def reset(self, *, seed=None, options=None):
        obs = self.env.reset()
        # PokerEnv.reset() returns the observation for the current player (usually P0 or P1 depending on dealer)
        # But RLlib expects observations for all agents who need to act?
        # Or just the one who acts?
        # In Turn-Based games like Poker, usually only one agent acts.
        # RLlib handles this by only returning obs for the agent who needs to act.
        
        current_player_id = f"player_{self.env.current_player}"
        return {current_player_id: obs}, {}

    def step(self, action_dict):
        # action_dict contains actions for agents who acted this turn
        # In Poker, only one agent acts at a time.
        
        current_player_id = f"player_{self.env.current_player}"
        
        if current_player_id in action_dict:
            action = action_dict[current_player_id]
            
            # Step the environment
            next_obs, reward, done, info = self.env.step(action)
            
            # Now, who acts next?
            next_player_id = f"player_{self.env.current_player}"
            
            rewards = {}
            terminateds = {}
            truncateds = {}
            infos = {}
            
            # Assign reward to the player who just acted
            # Note: PokerEnv.step returns reward for the ACTING player.
            rewards[current_player_id] = reward
            
            if done:
                # Game Over
                terminateds["player_0"] = True
                terminateds["player_1"] = True
                terminateds["__all__"] = True
                
                # If game is over, we might need to distribute final rewards?
                # PokerEnv.step returns reward for the acting player.
                # But the OTHER player might also have a payoff (e.g. if P0 folds, P1 wins).
                # My PokerEnv currently only returns reward for the acting player.
                # I need to check if PokerEnv gives payoffs for both.
                # PokerEnv.step: "Return reward for the player who just acted"
                # If P0 folds, P0 gets negative reward. P1 gets positive.
                # But P1 didn't act, so step() didn't return P1's reward.
                # In RLlib, we can return rewards for agents who didn't act.
                
                # Let's peek at env.env.get_payoffs() again to be sure.
                payoffs = self.env.env.get_payoffs()
                rewards["player_0"] = payoffs[0]
                rewards["player_1"] = payoffs[1]
                
                return {}, rewards, terminateds, truncateds, infos
            else:
                terminateds["__all__"] = False
                # Return observation for the NEXT player
                return {next_player_id: next_obs}, rewards, terminateds, truncateds, infos
                
        return {}, {}, {}, {}, {}

