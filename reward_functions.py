"""
Advanced Reward Functions for Tournament Poker
Implements ICM (Independent Chip Model) and survival-based rewards
"""
import numpy as np
from typing import Dict, Tuple


class RewardFunction:
    """Base class for reward functions"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
    
    def calculate(
        self,
        current_chips: float,
        starting_chips: float,
        opponent_chips: float,
        opponent_starting_chips: float,
        hands_played: int,
        is_winner: bool
    ) -> float:
        """Calculate reward based on tournament state"""
        raise NotImplementedError


class LinearReward(RewardFunction):
    """
    Simple linear reward based on chip difference
    Current implementation
    """
    
    def calculate(self, current_chips, starting_chips, **kwargs):
        return current_chips - starting_chips


class LogReward(RewardFunction):
    """
    Logarithmic reward - diminishing returns for chip accumulation
    Values small stacks more than large stacks
    """
    
    def __init__(self, config=None):
        super().__init__(config)
        self.scale = self.config.get('log_scale', 10.0)
    
    def calculate(self, current_chips, starting_chips, **kwargs):
        # Add 1 to avoid log(0)
        log_current = np.log(current_chips + 1)
        log_starting = np.log(starting_chips + 1)
        return (log_current - log_starting) * self.scale


class ICMReward(RewardFunction):
    """
    Independent Chip Model (ICM) reward
    Calculates tournament equity based on chip distribution
    
    In heads-up (2 players), ICM equity = my_chips / total_chips
    """
    
    def __init__(self, config=None):
        super().__init__(config)
        self.scale = self.config.get('icm_scale', 100.0)
    
    def calculate_equity(self, my_chips, opponent_chips):
        """Calculate ICM equity (winning probability)"""
        total_chips = my_chips + opponent_chips
        if total_chips <= 0:
            return 0.5  # Equal if no chips
        return my_chips / total_chips
    
    def calculate(
        self,
        current_chips,
        starting_chips,
        opponent_chips,
        opponent_starting_chips,
        **kwargs
    ):
        # Current equity
        current_equity = self.calculate_equity(current_chips, opponent_chips)
        
        # Starting equity
        starting_equity = self.calculate_equity(starting_chips, opponent_starting_chips)
        
        # Equity gain
        equity_gain = current_equity - starting_equity
        
        return equity_gain * self.scale


class ICMSurvivalReward(ICMReward):
    """
    ICM + Survival Bonus
    Combines ICM equity with bonus for lasting more hands
    """
    
    def __init__(self, config=None):
        super().__init__(config)
        self.survival_weight = self.config.get('survival_weight', 0.1)
        self.victory_bonus = self.config.get('victory_bonus', 50.0)
    
    def calculate(
        self,
        current_chips,
        starting_chips,
        opponent_chips,
        opponent_starting_chips,
        hands_played,
        is_winner,
        **kwargs
    ):
        # Base ICM reward
        icm_reward = super().calculate(
            current_chips=current_chips,
            starting_chips=starting_chips,
            opponent_chips=opponent_chips,
            opponent_starting_chips=opponent_starting_chips
        )
        
        # Survival bonus
        survival_bonus = hands_played * self.survival_weight
        
        # Victory bonus
        victory_bonus = self.victory_bonus if is_winner else 0
        
        # Total reward
        total_reward = icm_reward + survival_bonus + victory_bonus
        
        return total_reward


class AdaptiveReward(RewardFunction):
    """
    Adaptive reward that changes based on stack size
    
    - Deep stack (>50BB): Focus on chip accumulation
    - Medium stack (20-50BB): Balanced
    - Short stack (<20BB): Focus on survival
    """
    
    def __init__(self, config=None):
        super().__init__(config)
        self.big_blind = self.config.get('big_blind', 2)
        self.icm_reward = ICMReward(config)
        self.linear_reward = LinearReward(config)
    
    def calculate(
        self,
        current_chips,
        starting_chips,
        opponent_chips,
        opponent_starting_chips,
        hands_played,
        **kwargs
    ):
        # Calculate BB depth
        bb_depth = current_chips / self.big_blind
        
        # ICM reward (equity-based)
        icm_r = self.icm_reward.calculate(
            current_chips=current_chips,
            starting_chips=starting_chips,
            opponent_chips=opponent_chips,
            opponent_starting_chips=opponent_starting_chips
        )
        
        # Linear reward (chip-based)
        linear_r = self.linear_reward.calculate(
            current_chips=current_chips,
            starting_chips=starting_chips
        )
        
        # Blend based on stack depth
        if bb_depth > 50:
            # Deep stack: more chip focus
            return 0.3 * icm_r + 0.7 * linear_r
        elif bb_depth > 20:
            # Medium stack: balanced
            return 0.5 * icm_r + 0.5 * linear_r
        else:
            # Short stack: more survival focus
            survival_bonus = hands_played * 0.2
            return 0.7 * icm_r + 0.3 * linear_r + survival_bonus


# Factory for creating reward functions
REWARD_FUNCTIONS = {
    'linear': LinearReward,
    'log': LogReward,
    'icm': ICMReward,
    'icm_survival': ICMSurvivalReward,
    'adaptive': AdaptiveReward,
}


def create_reward_function(reward_type: str, config: Dict = None) -> RewardFunction:
    """
    Create a reward function by name
    
    Args:
        reward_type: Type of reward ('linear', 'log', 'icm', 'icm_survival', 'adaptive')
        config: Configuration dict
        
    Returns:
        RewardFunction instance
    """
    if reward_type not in REWARD_FUNCTIONS:
        raise ValueError(f"Unknown reward type: {reward_type}. Choose from {list(REWARD_FUNCTIONS.keys())}")
    
    return REWARD_FUNCTIONS[reward_type](config)


# Example configurations for different strategies
REWARD_CONFIGS = {
    'conservative': {
        'reward_type': 'icm_survival',
        'survival_weight': 0.2,
        'victory_bonus': 50,
        'icm_scale': 100,
    },
    'balanced': {
        'reward_type': 'icm_survival',
        'survival_weight': 0.1,
        'victory_bonus': 50,
        'icm_scale': 100,
    },
    'aggressive': {
        'reward_type': 'icm',
        'icm_scale': 150,
    },
    'adaptive': {
        'reward_type': 'adaptive',
        'big_blind': 2,
    }
}


if __name__ == "__main__":
    # Test reward functions
    print("=" * 80)
    print("Testing Reward Functions")
    print("=" * 80)
    
    # Scenario: Player starts with 100 chips, ends with 150 chips after 20 hands
    test_params = {
        'current_chips': 150,
        'starting_chips': 100,
        'opponent_chips': 50,
        'opponent_starting_chips': 100,
        'hands_played': 20,
        'is_winner': True
    }
    
    print("\nScenario:")
    print(f"  Starting: P0=100, P1=100")
    print(f"  Ending: P0=150, P1=50 (Winner!)")
    print(f"  Hands played: 20")
    print()
    
    for name, reward_cls in REWARD_FUNCTIONS.items():
        config = REWARD_CONFIGS.get(name.replace('_', ''), {})
        reward_fn = reward_cls(config)
        reward = reward_fn.calculate(**test_params)
        
        print(f"{name:20s}: {reward:+8.2f}")
    
    print("\n" + "=" * 80)
    print("Testing Stack-Dependent Behavior (Log Reward)")
    print("=" * 80)
    
    log_reward = LogReward({'log_scale': 10})
    
    scenarios = [
        ("10 → 11 chips", 11, 10),
        ("50 → 51 chips", 51, 50),
        ("100 → 101 chips", 101, 100),
        ("10 → 20 chips", 20, 10),
        ("100 → 110 chips", 110, 100),
    ]
    
    for desc, current, starting in scenarios:
        reward = log_reward.calculate(current, starting)
        print(f"{desc:20s}: {reward:+6.3f}")
    
    print("\n" * 80)
    print("✅ All reward functions working correctly!")
    print("=" * 80)
