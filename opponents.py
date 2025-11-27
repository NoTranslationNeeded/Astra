"""
Opponent bots for evaluating poker AI
"""
import numpy as np
from poker_env import PokerEnv

class RandomBot:
    """Random action bot - easiest opponent"""
    
    def __init__(self):
        self.name = "RandomBot"
    
    def get_action(self, observation, legal_actions):
        """Choose random legal action"""
        return np.random.choice(legal_actions)


class RuleBasedBot:
    """Rule-based bot with basic poker strategy"""
    
    def __init__(self):
        self.name = "RuleBasedBot"
    
    def get_action(self, observation, legal_actions):
        """
        Basic strategy:
        - Fold if weak hand and facing bet
        - Call/Check if medium hand
        - Raise if strong hand
        
        observation is a 55-element vector:
        - First 52: one-hot encoded cards
        - 53: equity (0-1)
        - 54-55: game state info
        """
        # Extract equity (element 52 in observation)
        equity = observation[52] if len(observation) > 52 else 0.5
        
        # Action mapping (from poker_env.py):
        # 0: Fold
        # 1: Check/Call
        # 2: Raise Half Pot
        # 3: Raise Pot
        # 4: All-in
        
        # Strategy based on hand strength (equity)
        if equity < 0.3:
            # Weak hand: Fold if possible, otherwise check/call
            if 0 in legal_actions:
                return 0  # Fold
            elif 1 in legal_actions:
                return 1  # Check/Call
            else:
                return legal_actions[0]
        
        elif equity < 0.6:
            # Medium hand: Check/Call
            if 1 in legal_actions:
                return 1  # Check/Call
            else:
                return legal_actions[0]
        
        elif equity < 0.8:
            # Good hand: Raise half pot
            if 2 in legal_actions:
                return 2  # Raise Half Pot
            elif 1 in legal_actions:
                return 1  # Check/Call
            else:
                return legal_actions[0]
        
        else:
            # Very strong hand: Raise pot or all-in
            if 3 in legal_actions:
                return 3  # Raise Pot
            elif 4 in legal_actions:
                return 4  # All-in
            elif 2 in legal_actions:
                return 2  # Raise Half Pot
            else:
                return 1  # Check/Call


class PastSelfBot:
    """Loads a checkpoint of past AI for evaluation"""
    
    def __init__(self, checkpoint_path):
        self.name = "PastSelfBot"
        self.checkpoint_path = checkpoint_path
        # TODO: Load model from checkpoint
        # For now, acts like RandomBot
        self.fallback = RandomBot()
    
    def get_action(self, observation, legal_actions):
        """Use loaded model to choose action"""
        # TODO: Implement model inference
        return self.fallback.get_action(observation, legal_actions)
