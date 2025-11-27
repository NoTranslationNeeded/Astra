from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from gymnasium import spaces
import numpy as np
from poker_env import PokerEnv

class PokerPettingZooEnv(AECEnv):
    metadata = {"render_modes": ["human"], "name": "poker_v0"}

    def __init__(self):
        super().__init__()
        
        self.env = PokerEnv()
        
        # Define agents
        self.possible_agents = ["player_0", "player_1"]
        self.agents = self.possible_agents[:]
        
        # Define spaces for each agent
        self.observation_spaces = {
            agent: spaces.Box(low=0, high=1, shape=(55,), dtype=np.float32)
            for agent in self.possible_agents
        }
        
        self.action_spaces = {
            agent: spaces.Discrete(self.env.action_space_size)
            for agent in self.possible_agents
        }
        
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = None
        
    def reset(self, seed=None, options=None):
        # Reset the environment
        obs = self.env.reset()
        
        self.agents = self.possible_agents[:]
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = f"player_{self.env.current_player}"
        
        # Initialize tracking
        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        
        # Store observation for current agent
        self._observations = {self.agent_selection: obs}
        
    def step(self, action):
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            return self._was_dead_step(action)
            
        current_agent = self.agent_selection
        
        # Execute action in environment
        next_obs, reward, done, info = self.env.step(action)
        
        # Update rewards
        self.rewards[current_agent] = reward
        self._cumulative_rewards[current_agent] += reward
        
        # Check if game over
        if done:
            # Game finished
            for agent in self.agents:
                self.terminations[agent] = True
                
            # Get final payoffs for both players
            payoffs = self.env.env.get_payoffs()
            self.rewards["player_0"] = payoffs[0]
            self.rewards["player_1"] = payoffs[1]
            self._cumulative_rewards["player_0"] += payoffs[0]
            self._cumulative_rewards["player_1"] += payoffs[1]
        else:
            # Continue game - next agent's turn
            next_agent = f"player_{self.env.current_player}"
            self._observations[next_agent] = next_obs
            self.agent_selection = next_agent
            
    def observe(self, agent):
        return self._observations.get(agent, np.zeros(55, dtype=np.float32))
    
    def observation_space(self, agent):
        return self.observation_spaces[agent]
    
    def action_space(self, agent):
        return self.action_spaces[agent]


def env_creator(config=None):
    """Factory function for Ray/RLlib"""
    env = PokerPettingZooEnv()
    # PettingZoo to Parallel wrapper for RLlib
    from pettingzoo.utils import parallel_to_aec_wrapper
    return env
