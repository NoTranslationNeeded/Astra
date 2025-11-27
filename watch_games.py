"""
Watch live poker games with detailed logging
Shows AI decision-making process in real-time
"""
import ray
from ray.rllib.algorithms.ppo import PPO
from poker_env import PokerEnv
import time

def watch_game(checkpoint_path=None, num_games=3):
    """
    Watch AI play poker games with detailed logging
    
    Args:
        checkpoint_path: Path to trained model (None = random actions)
        num_games: Number of games to watch
    """
    
    # Initialize environment
    env = PokerEnv()
    
    # Load trained model if checkpoint provided
    policy = None
    if checkpoint_path:
        ray.init(ignore_reinit_error=True)
        algo = PPO.from_checkpoint(checkpoint_path)
        policy = algo.get_policy("player_0")
        print(f"✅ Loaded checkpoint: {checkpoint_path}\n")
    else:
        print("🎲 Playing with random actions (no checkpoint loaded)\n")
    
    # Play games
    for game_num in range(1, num_games + 1):
        print("=" * 80)
        print(f"🎮 GAME {game_num}")
        print("=" * 80)
        
        obs = env.reset()
        done = False
        turn = 1
        total_pot = 0
        
        while not done:
            current_player = env.current_player
            legal_actions = env.get_legal_actions()
            
            # Get RLCard state
            state = env.env.get_state(current_player)
            
            # Get player hand from raw observation
            raw_obs = state['obs']
            # Hand cards are in the raw observation string representation
            try:
                # Extract hand info from state
                if 'raw_obs' in state:
                    hand_str = str(state['raw_obs'].get('hand', 'Hidden'))
                else:
                    hand_str = "Hidden"
            except:
                hand_str = "Hidden"
            
            # Get equity if available
            equity = obs[52] if len(obs) > 52 else 0.5
            
            print(f"\n--- Turn {turn} ---")
            print(f"Player {current_player}'s turn")
            print(f"  Hand: {hand_str}")
            print(f"  Equity: {equity:.1%}")
            print(f"  Legal actions: {[action_name(a) for a in legal_actions]}")
            
            # Choose action
            if policy and current_player == 0:
                # Use trained AI for Player 0
                action, _, _ = policy.compute_single_action(obs, explore=False)
            else:
                # Random action for Player 1 (or if no policy)
                import numpy as np
                action = np.random.choice(legal_actions)
            
            print(f"  → Action taken: {action_name(action)}")
            
            # Step environment
            obs, reward, done, info = env.step(action)
            turn += 1
            
            # Small delay for readability
            time.sleep(0.3)
        
        # Game ended
        payoffs = env.env.get_payoffs()
        print("\n" + "=" * 80)
        print("🏁 GAME OVER")
        print(f"  Player 0: {payoffs[0]:+.1f} chips")
        print(f"  Player 1: {payoffs[1]:+.1f} chips")
        
        if payoffs[0] > 0:
            print("  🏆 Player 0 WINS!")
        elif payoffs[1] > 0:
            print("  🏆 Player 1 WINS!")
        else:
            print("  🤝 TIE!")
        
        print("=" * 80 + "\n")
        time.sleep(1)
    
    if checkpoint_path:
        ray.shutdown()


def action_name(action_id):
    """Convert action ID to readable name"""
    actions = {
        0: "FOLD",
        1: "CHECK/CALL",
        2: "RAISE HALF POT",
        3: "RAISE POT",
        4: "ALL-IN"
    }
    return actions.get(action_id, f"ACTION_{action_id}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Watch AI poker games")
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to checkpoint (optional, uses random if not provided)"
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=3,
        help="Number of games to watch (default: 3)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("🎰 POKER GAME WATCHER")
    print("=" * 80)
    
    watch_game(args.checkpoint, args.num_games)
