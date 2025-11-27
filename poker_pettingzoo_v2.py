from pettingzoo.utils.env import ParallelEnv
from gymnasium import spaces
import numpy as np
from poker_env import PokerEnv

class PokerParallelEnv(ParallelEnv):
    metadata = {
        "render_modes": ["human"],
        "name": "poker_parallel_v1"
    }

    def __init__(self, render_mode=None):
        super().__init__()
        
        self.env = PokerEnv()
        self.render_mode = render_mode
        
        # Define agents
        self.possible_agents = ["player_0", "player_1"]
        self.agents = self.possible_agents[:]
        
        # Define spaces
        obs_space = spaces.Box(low=0, high=1, shape=(55,), dtype=np.float32)
        act_space = spaces.Discrete(self.env.action_space_size)
        
        self.observation_spaces = {agent: obs_space for agent in self.possible_agents}
        self.action_spaces = {agent: act_space for agent in self.possible_agents}
        
    def reset(self, seed=None, options=None):
        """Reset the environment"""
        if seed is not None:
            np.random.seed(seed)
            
        # Reset underlying env
        obs = self.env.reset()
        
        # Reset agents
        self.agents = self.possible_agents[:]
        
        # Return observations - only current player observes
        current_player = f"player_{self.env.current_player}"
        observations = {current_player: obs}
        
        # Return infos
        infos = {agent: {} for agent in self.agents}
        
        return observations, infos
    
    def step(self, actions):
        """
        Execute actions for agents who are acting this turn
        
        In turn-based games like poker, only one agent acts at a time.
        RLlib will call this with actions dict containing only the acting agent.
        """
        # Get current acting agent
        current_player = f"player_{self.env.current_player}"
        
        # Execute action if provided
        if current_player in actions:
            action = actions[current_player]
            next_obs, reward, done, info = self.env.step(action)
            
            # Prepare returns
            observations = {}
            rewards = {}
            terminations = {}
            truncations = {}
            infos = {}
            
            if done:
                # Game over - get final payoffs
                payoffs = self.env.env.get_payoffs()
                
                # Assign rewards
                rewards["player_0"] = float(payoffs[0])
                rewards["player_1"] = float(payoffs[1])
                
                # Mark both as terminated
                terminations["player_0"] = True
                terminations["player_1"] = True
                truncations["player_0"] = False
                truncations["player_1"] = False
                
                # Empty observations (game over)
                observations = {}
                
                # Infos
                infos["player_0"] = {}
                infos["player_1"] = {}
                
                # Clear agents
                self.agents = []
            else:
                # Game continues - next player's turn
                next_player = f"player_{self.env.current_player}"
                
                # Only next player observes
                observations = {next_player: next_obs}
                
                # Reward for acting player (intermediate reward, usually 0)
                rewards = {current_player: 0.0}
                
                # No terminations yet
                terminations = {agent: False for agent in self.possible_agents}
                truncations = {agent: False for agent in self.possible_agents}
                
                # Infos
                infos = {agent: {} for agent in self.agents}
                
            return observations, rewards, terminations, truncations, infos
        else:
            # No action for current player (should not happen)
            return {}, {}, {}, {}, {}
    
    def observation_space(self, agent):
        return self.observation_spaces[agent]
    
    def action_space(self, agent):
        return self.action_spaces[agent]
    
    def close(self):
        """Cleanup"""
        pass
