"""
PettingZoo ParallelEnv wrapper for Tournament Poker Environment
Enables RLlib to train on tournament-style poker
"""
from pettingzoo.utils.env import ParallelEnv
from gymnasium import spaces
import numpy as np
from tournament_poker_env import TournamentPokerEnv

class TournamentPokerParallelEnv(ParallelEnv):
    metadata = {
        "render_modes": ["human"],
        "name": "tournament_poker_v1"
    }

    def __init__(self, starting_chips=100, randomize_stacks=True, render_mode=None,
                 reward_type='icm_survival', reward_config=None):
        super().__init__()
        
        self.env = TournamentPokerEnv(
            starting_chips=starting_chips,
            small_blind=1,
            big_blind=2,
            randomize_stacks=randomize_stacks,
            reward_type=reward_type,
            reward_config=reward_config
        )
        self.render_mode = render_mode
        
        # Define agents
        self.possible_agents = ["player_0", "player_1"]
        self.agents = self.possible_agents[:]
        
        # Define spaces - 60 dimensions with blind level info
        obs_space = spaces.Box(low=0, high=1, shape=(60,), dtype=np.float32)
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
                # Tournament over - get final payoffs
                # Apply BB normalization: (chip_diff / BB) / 250.0
                final_chips = info['final_chips']
                starting_chips = info['starting_chips']
                current_bb = self.env.big_blind
                
                chip_diff_0 = final_chips[0] - starting_chips[0]
                chip_diff_1 = final_chips[1] - starting_chips[1]
                
                bb_diff_0 = chip_diff_0 / current_bb
                bb_diff_1 = chip_diff_1 / current_bb
                
                # Assign rewards (normalized)
                rewards["player_0"] = float(bb_diff_0 / 250.0)
                rewards["player_1"] = float(bb_diff_1 / 250.0)
                
                # Mark both as terminated
                terminations["player_0"] = True
                terminations["player_1"] = True
                truncations["player_0"] = False
                truncations["player_1"] = False
                
                # Empty observations (game over)
                observations = {}
                
                # Infos
                infos["player_0"] = info
                infos["player_1"] = info
                
                # Clear agents
                self.agents = []
            else:
                # Game continues
                next_agent = f"player_{self.env.current_player}"
                observations = {next_agent: next_obs}
                
                # Reward is for the agent who acted
                rewards = {current_player: float(reward)}
                
                terminations = {agent: False for agent in self.possible_agents}
                truncations = {agent: False for agent in self.possible_agents}
                
                # Infos
                infos = {agent: info for agent in self.agents}
                
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


if __name__ == "__main__":
    # Test the PettingZoo wrapper
    print("=" * 80)
    print("Testing Tournament Poker PettingZoo Wrapper")
    print("=" * 80)
    
    env = TournamentPokerParallelEnv(randomize_stacks=True)
    
    for episode in range(2):
        print(f"\n🎮 Episode {episode + 1}")
        observations, infos = env.reset()
        print(f"Starting chips: {env.env.chips}")
        print(f"Observation shape: {list(observations.values())[0].shape}")
        
        done = False
        step_count = 0
        max_steps = 500
        
        while not done and step_count < max_steps:
            # Get current agent
            current_agent = list(observations.keys())[0]
            
            # Random action
            legal_actions = env.env.get_legal_actions()
            action = np.random.choice(legal_actions)
            
            # Step
            observations, rewards, terminations, truncations, infos = env.step({current_agent: action})
            
            # Check if done
            done = any(terminations.values())
            step_count += 1
            
            if done:
                print(f"\n✅ Episode complete after {step_count} steps")
                info = infos[list(infos.keys())[0]]
                print(f"Hands played: {info['hands_played']}")
                print(f"Winner: Player {info['tournament_winner']}")
                print(f"Final chips: {info['final_chips']}")
        
        if not done:
            print(f"\n⚠️ Max steps reached ({max_steps})")
    
    print("\n" + "=" * 80)
    print("✅ Wrapper test complete!")
