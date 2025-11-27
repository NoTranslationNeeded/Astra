import os
import glob
import subprocess
import sys

def find_latest_checkpoint(base_dir="./ray_results"):
    # Find all checkpoint directories
    checkpoints = glob.glob(os.path.join(base_dir, "**", "checkpoint_000*"), recursive=True)
    
    if not checkpoints:
        print("❌ No checkpoints found!")
        return None
        
    # Sort by modification time
    latest_checkpoint = max(checkpoints, key=os.path.getmtime)
    return latest_checkpoint

def run_verification():
    print("🔍 Searching for latest checkpoint...")
    checkpoint = find_latest_checkpoint()
    
    if not checkpoint:
        sys.exit(1)
        
    print(f"✅ Found latest checkpoint: {checkpoint}")
    print("\n🚀 Starting verification (1 tournament)...")
    
    # Run watch_tournament.py
    cmd = [
        sys.executable, 
        "watch_tournament.py", 
        "--checkpoint", checkpoint,
        "--num-games", "1"
    ]
    
    # Run and capture output
    with open("verification_log.txt", "w", encoding="utf-8") as f_out, open("verification_error.txt", "w", encoding="utf-8") as f_err:
        process = subprocess.Popen(
            cmd, 
            stdout=f_out, 
            stderr=f_err,
            encoding="utf-8",
            errors="replace"
        )
        process.wait()
    
    print(f"\n✨ Verification complete!")
    print(f"  Log: verification_log.txt")
    print(f"  Errors: verification_error.txt")

if __name__ == "__main__":
    run_verification()
