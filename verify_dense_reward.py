from tournament_poker_env import TournamentPokerEnv
import numpy as np

def verify_reward_normalization():
    print("=" * 80)
    print("Dense Reward Verification (BB Normalization)")
    print("=" * 80)
    
    env = TournamentPokerEnv(randomize_stacks=True)
    
    print("\nTesting 3 episodes with different BB ranges...\n")
    
    for episode in range(3):
        obs = env.reset()
        bb = env.big_blind
        starting_chips = env.chips[0]
        starting_bb = starting_chips / bb
        
        print(f"\n--- Episode {episode + 1} ---")
        print(f"Starting Chips: {starting_chips:.0f}")
        print(f"Big Blind: {bb}")
        print(f"Starting Stack: {starting_bb:.1f} BB")
        
        # Play one hand
        step_count = 0
        hand_ended = False
        
        while step_count < 50 and not hand_ended:
            legal_actions = env.get_legal_actions()
            if not legal_actions:
                break
                
            action = np.random.choice(legal_actions)
            obs, reward, done, info = env.step(action)
            
            step_count += 1
            
            if info.get('hand_ended'):
                hand_ended = True
                payoffs = info['hand_payoffs']
                
                bb_payoff = payoffs[0] / bb
                expected_reward = bb_payoff / 250.0
                
                print(f"\n  Hand Result:")
                print(f"    Chip Payoff: {payoffs[0]:+.0f}")
                print(f"    BB Payoff: {bb_payoff:+.2f} BB")
                print(f"    Reward: {reward:+.6f}")
                print(f"    Expected: {expected_reward:+.6f}")
                print(f"    Match: {'✅' if abs(reward - expected_reward) < 0.0001 else '❌'}")
                
            if done:
                print(f"\n  Tournament Over!")
                print(f"    Winner: Player {info['tournament_winner']}")
                break
    
    print("\n" + "=" * 80)
    print("Verification Complete!")
    print("=" * 80)

if __name__ == "__main__":
    verify_reward_normalization()
