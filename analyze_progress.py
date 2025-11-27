import pandas as pd
import glob
import os

# Find the latest progress.csv
search_path = r"c:\Users\99san\.gemini\antigravity\playground\glacial-supernova\ray_results\tournament_poker_icm_survival\PPO_tournament_poker_icm_22285_00000_0_2025-11-27_17-36-13\progress.csv"

try:
    df = pd.read_csv(search_path)
    
    print(f"Total Iterations: {len(df)}")
    
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    print("Available 'reward' columns:")
    for c in df.columns:
        if "reward" in c:
            print(f" - {c}: {df.iloc[-1][c]}")
            
    print("\nAvailable 'mean' columns:")
    for c in df.columns:
        if "mean" in c and "time" not in c:
            print(f" - {c}: {df.iloc[-1][c]}")
            
    if "env_runners/episode_len_mean" in df.columns:
        print(f"\n*** EPISODE LENGTH: {df.iloc[-1]['env_runners/episode_len_mean']} ***")
            
    print("\nAvailable 'entropy' columns:")
    for c in df.columns:
        if "entropy" in c:
            print(f" - {c}: {df.iloc[-1][c]}")

except Exception as e:
    print(f"Error reading CSV: {e}")
