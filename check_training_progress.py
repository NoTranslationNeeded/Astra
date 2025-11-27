import pandas as pd
import os

# Find the latest experiment directory
base_dir = "ray_results/deepstack_7actions_dense_v1"
exp_dirs = [d for d in os.listdir(base_dir) if d.startswith("PPO_")]
exp_dirs.sort(key=lambda x: os.path.getmtime(os.path.join(base_dir, x)), reverse=True)

if exp_dirs:
    latest_exp = exp_dirs[0]
    csv_path = os.path.join(base_dir, latest_exp, "progress.csv")
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        
        # Get the last row
        latest = df.iloc[-1]
        
        print("=" * 80)
        print("TRAINING PROGRESS REPORT")
        print("=" * 80)
        print(f"\n📊 Iteration: {int(latest.get('training_iteration', 0))}")
        print(f"⏱️  Time Elapsed: {latest.get('time_total_s', 0) / 60:.1f} minutes")
        print(f"📈 Timesteps: {int(latest.get('num_env_steps_sampled_lifetime', 0)):,} / 1,000,000")
        
        print("\n🎮 Episode Metrics:")
        print(f"   Reward Mean: {latest.get('env_runners/episode_return_mean', 0):.4f}")
        print(f"   Reward Max:  {latest.get('env_runners/episode_return_max', 0):.4f}")
        print(f"   Reward Min:  {latest.get('env_runners/episode_return_min', 0):.4f}")
        print(f"   Episode Len: {latest.get('env_runners/episode_len_mean', 0):.1f} hands")
        
        print("\n🧠 Learning Metrics:")
        print(f"   Policy Loss: {latest.get('learner/default_policy/learner_stats/policy_loss', 0):.6f}")
        print(f"   Value Loss:  {latest.get('learner/default_policy/learner_stats/vf_loss', 0):.6f}")
        print(f"   Entropy:     {latest.get('learner/default_policy/learner_stats/entropy', 0):.4f}")
        print(f"   KL:          {latest.get('learner/default_policy/learner_stats/kl', 0):.6f}")
        
        print("\n⚙️  Performance:")
        print(f"   Learn Time:  {latest.get('timers/learner_grad_time_ms', 0):.1f} ms")
        print(f"   Sample Time: {latest.get('timers/env_runner_sampling_time_ms', 0):.1f} ms")
        
        # Progress percentage
        progress = (latest.get('num_env_steps_sampled_lifetime', 0) / 1_000_000) * 100
        print(f"\n🚀 Progress: {progress:.1f}% complete")
        
        # Estimate time remaining
        if latest.get('num_env_steps_sampled_lifetime', 0) > 0:
            time_per_step = latest.get('time_total_s', 0) / latest.get('num_env_steps_sampled_lifetime', 1)
            remaining_steps = 1_000_000 - latest.get('num_env_steps_sampled_lifetime', 0)
            remaining_time_min = (remaining_steps * time_per_step) / 60
            print(f"⏳ Est. Time Remaining: {remaining_time_min:.1f} minutes")
        
        print("\n" + "=" * 80)
    else:
        print("progress.csv not found!")
else:
    print("No experiment directories found!")
