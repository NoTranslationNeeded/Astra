"""
Evaluation system for poker AI
"""
import numpy as np
from poker_env import PokerEnv
from opponents import RandomBot, RuleBasedBot
import mlflow

class PokerEvaluator:
    """Evaluates AI against various opponents"""
    
    def __init__(self, env=None):
        self.env = env or PokerEnv()
        self.opponents = {
            "random": RandomBot(),
            "rule_based": RuleBasedBot(),
        }
    
    def evaluate_agent(self, agent, opponent_name="random", num_games=100, greedy=True):
        """
        Evaluate agent against an opponent
        
        Args:
            agent: The AI agent to evaluate (RLlib policy or custom)
            opponent_name: "random" or "rule_based"
            num_games: Number of games to play
            greedy: If True, use greedy policy (no exploration)
        
        Returns:
            dict with metrics: win_rate, avg_reward, avg_game_length
        """
        opponent = self.opponents[opponent_name]
        
        wins = 0
        total_reward = 0.0
        total_length = 0
        
        for game in range(num_games):
            # Reset environment
            obs = self.env.reset()
            done = False
            game_length = 0
            episode_reward = 0.0
            
            # Randomly assign AI to Player 0 or Player 1
            ai_player = np.random.randint(2)
            
            while not done:
                current_player = self.env.current_player
                legal_actions = self.env.get_legal_actions()
                
                if current_player == ai_player:
                    # AI's turn
                    action = self._get_ai_action(agent, obs, legal_actions, greedy)
                else:
                    # Opponent's turn
                    action = opponent.get_action(obs, legal_actions)
                
                # Step environment
                obs, reward, done, info = self.env.step(action)
                game_length += 1
                
                # Track AI's reward
                if current_player == ai_player:
                    episode_reward += reward
            
            # Game ended - check winner
            payoffs = self.env.env.get_payoffs()
            ai_payoff = payoffs[ai_player]
            
            if ai_payoff > 0:
                wins += 1
            
            total_reward += ai_payoff
            total_length += game_length
        
        # Calculate metrics
        win_rate = wins / num_games
        avg_reward = total_reward / num_games
        avg_game_length = total_length / num_games
        
        metrics = {
            "win_rate": win_rate,
            "avg_reward": avg_reward,
            "avg_game_length": avg_game_length,
            "num_games": num_games,
            "opponent": opponent_name,
        }
        
        return metrics
    
    def _get_ai_action(self, agent, observation, legal_actions, greedy):
        """Get action from AI agent"""
        if greedy:
            # Greedy mode: choose best action (no exploration)
            # For RLlib policy, we'll need to compute action
            # For now, assume agent has a compute_action method
            if hasattr(agent, 'compute_single_action'):
                action, _, _ = agent.compute_single_action(
                    observation, 
                    explore=False
                )
                return action
            elif callable(agent):
                return agent(observation)
            else:
                # Fallback: random
                return np.random.choice(legal_actions)
        else:
            # Exploration mode
            if hasattr(agent, 'compute_single_action'):
                action, _, _ = agent.compute_single_action(
                    observation,
                    explore=True
                )
                return action
            else:
                return np.random.choice(legal_actions)
    
    def log_evaluation_results(self, metrics, iteration=None):
        """Log evaluation results to MLflow"""
        prefix = f"eval/{metrics['opponent']}"
        
        mlflow.log_metric(f"{prefix}/win_rate", metrics["win_rate"], step=iteration)
        mlflow.log_metric(f"{prefix}/avg_reward", metrics["avg_reward"], step=iteration)
        mlflow.log_metric(f"{prefix}/avg_game_length", metrics["avg_game_length"], step=iteration)
        
        print(f"\n{'='*60}")
        print(f"Evaluation vs {metrics['opponent']} ({metrics['num_games']} games)")
        print(f"  Win Rate: {metrics['win_rate']:.1%}")
        print(f"  Avg Reward: {metrics['avg_reward']:.2f}")
        print(f"  Avg Game Length: {metrics['avg_game_length']:.1f} turns")
        print(f"{'='*60}\n")
