import numpy as np
from collections import deque
from poker_env import PokerEnv
from agent import Agent
import torch
import os

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        return self.buffer[idx]

    def sample(self, batch_size):
        import random
        return random.sample(self.buffer, batch_size)

def train(num_episodes=10000, save_interval=1000):
    env = PokerEnv()
    agent = Agent(env.observation_space_size, env.action_space_size)
    replay_buffer = ReplayBuffer(10000)
    
    # Load existing model if available
    if os.path.exists("poker_ai.pth"):
        print("Loading existing model...")
        agent.load("poker_ai.pth")
    
    print(f"Starting training for {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # Select action
            action = agent.select_action(state)
            
            # Step environment
            next_state, reward, done, _ = env.step(action)
            
            # Store transition
            # Note: In self-play, we might want to store transitions for both players
            # But here we are training a single agent that plays both sides (conceptually)
            # or we just train from the perspective of the current player.
            # Since the state includes "current_player", the agent learns to play from any position.
            
            replay_buffer.push(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            
            # Train
            agent.train(replay_buffer)
        
        # Update epsilon
        agent.update_epsilon()
        
        # Update target network
        if episode % 100 == 0:
            agent.update_target_network()
            
        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")
            
        if episode % save_interval == 0:
            agent.save("poker_ai.pth")
            print(f"Model saved at episode {episode}")

    agent.save("poker_ai.pth")
    print("Training complete!")

if __name__ == "__main__":
    train()
