import pandas as pd

df = pd.read_csv('ray_results/deepstack_7actions_dense_v1/PPO_tournament_poker_icm_95fcf_00000_0_2025-11-28_00-48-58/progress.csv')
latest = df.iloc[-1]

print(f"Iteration: {int(latest['training_iteration'])}")
print(f"Timesteps: {int(latest['num_env_steps_sampled_lifetime']):,} / 1,000,000")
print(f"Progress: {(latest['num_env_steps_sampled_lifetime'] / 1_000_000 * 100):.1f}%")
print(f"Reward Mean: {latest.get('env_runners/episode_return_mean', 0):.4f}")
print(f"Entropy: {latest.get('learner/default_policy/learner_stats/entropy', 0):.4f}")
print(f"Policy Loss: {latest.get('learner/default_policy/learner_stats/policy_loss', 0):.6f}")
