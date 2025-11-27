import rlcard
import numpy as np

def inspect_rlcard():
    print("🔍 Inspecting RLCard Configuration Options...")
    
    # Test 1: Default configuration
    env = rlcard.make('no-limit-holdem')
    env.reset()
    print(f"\n[Default] Game Class: {type(env.game)}")
    
    print("\n[Inspection] Dealer Attributes:")
    if hasattr(env.game, 'dealer'):
        for attr in dir(env.game.dealer):
            if not attr.startswith('__'):
                print(f"  {attr}")
    
    print("\n[Inspection] Game Attributes:")
    for attr in dir(env.game):
        if not attr.startswith('__'):
            val = getattr(env.game, attr)
            if isinstance(val, (int, float, str, list, dict)):
                print(f"  {attr}: {val}")

    # Check initial chips
    print(f"\n[Default] Initial Chips (P0): {env.game.players[0].remained_chips}")
    print(f"[Default] Initial Chips (P1): {env.game.players[1].remained_chips}")
    
    # Test 2: Try passing 'chips_for_each_player' in config
    print(f"\n[Test 2] Trying 'chips_for_each_player' = [500, 300]")
    config = {
        'game_num_players': 2,
        'chips_for_each_player': [500, 300]
    }
    try:
        env2 = rlcard.make('no-limit-holdem', config=config)
        env2.reset()
        print(f"[Test 2] Chips (P0): {env2.game.players[0].remained_chips}")
        print(f"[Test 2] Chips (P1): {env2.game.players[1].remained_chips}")
    except Exception as e:
        print(f"[Test 2] Failed: {e}")

    # Test 3: Try passing 'stack' or 'chips'
    print(f"\n[Test 3] Trying 'stack' = 500")
    config = {
        'game_num_players': 2,
        'stack': 500
    }
    try:
        env3 = rlcard.make('no-limit-holdem', config=config)
        env3.reset()
        print(f"[Test 3] Chips (P0): {env3.game.players[0].remained_chips}")
    except Exception as e:
        print(f"[Test 3] Failed: {e}")
        
    # Test 4: Configure method
    print(f"\n[Test 4] Testing configure method")
    if hasattr(env.game, 'configure'):
        try:
            # Try configuring chips
            config = {'chips_for_each_player': [500, 500], 'big_blind': 10}
            env.game.configure(config)
            env.reset()
            print(f"  [After Configure] Chips (P0): {env.game.players[0].remained_chips}")
            print(f"  [After Configure] Big Blind: {getattr(env.game, 'big_blind', 'N/A')}")
        except Exception as e:
            print(f"  [Test 4] Configure failed: {e}")
    
    # Test 5: Direct Modification
    print(f"\n[Test 5] Testing Direct Modification")
    try:
        env5 = rlcard.make('no-limit-holdem')
        env5.reset()
        
        # Modify chips directly
        env5.game.players[0].remained_chips = 888
        env5.game.players[1].remained_chips = 999
        
        # Modify blind directly (if attribute exists)
        if hasattr(env5.game, 'big_blind'):
            env5.game.big_blind = 50
        
        # Check if it sticks (without reset, because reset might revert it)
        print(f"  [Direct] Chips (P0): {env5.game.players[0].remained_chips}")
        print(f"  [Direct] Big Blind: {getattr(env5.game, 'big_blind', 'N/A')}")
        
        # Check if it survives reset (unlikely for chips, but maybe for blind)
        # Note: reset() usually re-initializes players, so chips might reset to default
        env5.reset()
        print(f"  [After Reset] Chips (P0): {env5.game.players[0].remained_chips}")
        print(f"  [After Reset] Big Blind: {getattr(env5.game, 'big_blind', 'N/A')}")
        
    except Exception as e:
        print(f"  [Test 5] Failed: {e}")

    # Test 6: Float Chips
    print(f"\n[Test 6] Testing Float Chips")
    try:
        env6 = rlcard.make('no-limit-holdem')
        env6.reset()
        
        # Set float chips
        env6.game.players[0].remained_chips = 100.5
        env6.game.players[1].remained_chips = 99.5
        
        print(f"  [Float] Chips set to 100.5 and 99.5")
        
        # Try to play a hand (step)
        # We need to take an action to trigger pot calculation eventually
        # But just setting it might be enough if we call step
        
        # P0 calls/raises
        action = 2 # Call (check if legal)
        print("  [Float] P0 performs action")
        next_state, next_player = env6.step(action)
        print("  [Float] Step successful")
        
    except Exception as e:
        print(f"  [Test 6] Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    inspect_rlcard()
