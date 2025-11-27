import os
import sys
import subprocess

def run_verification_final():
    # Define relative path
    relative_path = "./ray_results/tournament_poker_icm_survival/PPO_tournament_poker_icm_b0e6d_00000_0_2025-11-27_18-51-48/checkpoint_000019"
    
    # Convert to absolute path
    abs_path = os.path.abspath(relative_path)
    print(f"🔍 Using absolute checkpoint path: {abs_path}")
    
    if not os.path.exists(abs_path):
        print("❌ Checkpoint path does not exist!")
        return

    # Run watch_tournament.py
    cmd = [
        sys.executable, 
        "watch_tournament.py", 
        "--checkpoint", abs_path,
        "--num-games", "1"
    ]
    
    print("🚀 Starting verification...")
    
    # Run and capture output
    with open("final_verification_log.txt", "w", encoding="utf-8") as f:
        process = subprocess.Popen(
            cmd, 
            stdout=f, 
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            errors="replace"
        )
        process.wait()
        
    print(f"✨ Verification complete! Exit code: {process.returncode}")
    print("  Log saved to final_verification_log.txt")

if __name__ == "__main__":
    run_verification_final()
