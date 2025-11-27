"""
Quick test script to verify the game logger works correctly
"""
from tournament_logger import game_logger
import os

# Test logging
print("Testing game logger...")

# Write some test logs
game_logger.info("=" * 80)
game_logger.info("TEST LOG - Verifying poker_games.log functionality")
game_logger.info("=" * 80)
game_logger.info("Starting Chips: P0=100, P1=100")
game_logger.info("Initial Blinds: 1/2")
game_logger.info("")

game_logger.info("─" * 80)
game_logger.info("Hand #1 | Blinds: 1/2")
game_logger.info("─" * 80)
game_logger.info("  P0 Cards: [Ah Kd]")
game_logger.info("  P1 Cards: [7s 8s]")
game_logger.info("  Community: [Jh Tc 9d]")
game_logger.info("")
game_logger.info("  Result: P0 wins (pot: 10.0)")
game_logger.info("  Payoffs: P0=(+5.0), P1=(-5.0)")
game_logger.info("  Chips After: P0=105.0, P1=95.0")
game_logger.info("")

game_logger.info("=" * 80)
game_logger.info("Tournament Result")
game_logger.info("=" * 80)
game_logger.info("  Duration: 1 hands")
game_logger.info("  Winner: Player 0")
game_logger.info("  Final Chips: P0=105.0, P1=95.0")
game_logger.info("=" * 80)
game_logger.info("")

# Verify file was created
if os.path.exists('poker_games.log'):
    print("✓ poker_games.log file created successfully!")
    
    with open('poker_games.log', 'r', encoding='utf-8') as f:
        content = f.read()
    
    print(f"✓ File size: {len(content)} bytes")
    print(f"✓ Number of lines: {len(content.splitlines())}")
    
    print("\n" + "="*80)
    print("Content preview:")
    print("="*80)
    print(content)
    
    print("\n✅ Logger test PASSED!")
else:
    print("❌ poker_games.log file was NOT created!")
    print("Logger test FAILED!")
