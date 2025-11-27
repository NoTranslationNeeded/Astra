from tournament_poker_env import TournamentPokerEnv
import numpy as np

def verify_variable_stack():
    print("Initializing TournamentPokerEnv...")
    env = TournamentPokerEnv(randomize_stacks=True)
    
    print("=" * 80)
    print("Testing Variable Stack Depth & Random Blinds")
    print("=" * 80)
    
    depths = []
    blinds = []
    
    for i in range(10):
        env.reset()
        bb = env.big_blind
        chips = env.chips[0]
        depth = chips / bb
        
        depths.append(depth)
        blinds.append(bb)
        
        print(f"Episode {i+1}: Chips={chips:.0f}, BB={bb}, Depth={depth:.1f} BB")
        
        # Verify constraints
        if not (1 <= depth <= 250):
            print(f"  ERROR: Depth {depth} out of range (1-250)!")
        if not (250 <= bb <= 5000):
            print(f"  ERROR: BB {bb} out of range (250-5000)!")
            
    print("\nSummary:")
    print(f"Min Depth: {min(depths):.1f} BB, Max Depth: {max(depths):.1f} BB")
    print(f"Min BB: {min(blinds)}, Max BB: {max(blinds)}")
    
    if min(depths) < 1 or max(depths) > 250:
        print("FAIL: Depth range violation")
    else:
        print("PASS: Depth range correct")
        
    if min(blinds) < 250 or max(blinds) > 5000:
        print("FAIL: BB range violation")
    else:
        print("PASS: BB range correct")

if __name__ == "__main__":
    verify_variable_stack()
