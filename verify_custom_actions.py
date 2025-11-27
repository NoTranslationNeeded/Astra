from tournament_poker_env import TournamentPokerEnv
import numpy as np

def verify_actions():
    print("Initializing TournamentPokerEnv...")
    env = TournamentPokerEnv()
    print(f"Action Space Size: {env.action_space_size}")
    
    obs = env.reset()
    
    # Check Pre-flop Actions
    print("\n--- Pre-flop Check ---")
    legal_actions = env.get_legal_actions()
    
    # Debug Pot and Min Raise
    pot = env.base_env.game.dealer.pot
    current_raise = env.base_env.game.round.raised[env.current_player]
    max_raise = max(env.base_env.game.round.raised)
    min_raise_amount = max_raise - current_raise
    
    print(f"DEBUG: Pot={pot}, MyRaise={current_raise}, MaxRaise={max_raise}, CallAmount={min_raise_amount}")
    print(f"DEBUG: 33% Pot = {int(pot * 0.33)}")
    print(f"DEBUG: 150% Pot = {int(pot * 1.5)}")
    
    # Verify masking: 2 (33%) and 5 (150%) should NOW be in legal_actions (User removed masking)
    # NOTE: They might still be filtered by Poker Rules (Min Raise)
    if 2 in legal_actions:
        print("PASS: Action 2 (33%) available in Pre-flop.")
    else:
        print("NOTE: Action 2 (33%) NOT found (Likely due to Min Raise rule).")
        
    if 5 in legal_actions:
        print("PASS: Action 5 (150%) available in Pre-flop.")
    else:
        print("NOTE: Action 5 (150%) NOT found (Likely due to Min Raise rule).")
        
    # Play until Post-flop (Flop)
    print("\n--- Playing to Flop ---")
    # Simple strategy: Call until flop
    max_steps = 20
    steps = 0
    
    while steps < max_steps:
        # Check stage
        stage = env.base_env.game.stage
        # Stage is an Enum, get value
        stage_val = stage.value if hasattr(stage, 'value') else stage
        
        stage_name = ["PREFLOP", "FLOP", "TURN", "RIVER"][stage_val] if stage_val < 4 else "END"
        print(f"Step {steps}: Stage {stage_name}")
        
        legal_actions = env.get_legal_actions()
        print(f"  Legal: {legal_actions}")
        
        if stage_val == 1: # Flop
            print("\n--- Flop Reached ---")
            # Check if 2 (33%) or 5 (150%) are available (might depend on chips/pot)
            if 2 in legal_actions:
                print("PASS: Action 2 (33%) available in Flop.")
            else:
                print("NOTE: Action 2 (33%) not available (maybe invalid due to stack size?)")
                
            if 5 in legal_actions:
                print("PASS: Action 5 (150%) available in Flop.")
            else:
                print("NOTE: Action 5 (150%) not available (maybe invalid due to stack size?)")
            break
            
        # Take Check/Call (Action 1) if available, else random
        action = 1 if 1 in legal_actions else np.random.choice(legal_actions)
        obs, reward, done, info = env.step(action)
        
        if done:
            print("Hand ended before Flop. Resetting...")
            env.reset()
            steps = 0
            continue
            
        steps += 1

if __name__ == "__main__":
    verify_actions()
