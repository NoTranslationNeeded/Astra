import argparse
from trainer import train
from poker_env import PokerEnv
from agent import Agent
import torch
import numpy as np
import eval7
from rlcard.games.nolimitholdem.judger import Judger

# Monkey-patch RLCard Judger for speed optimization using eval7
def optimized_judge_game(self, players, hands):
    scores = []
    for hand_cards in hands:
        if not hand_cards:
            scores.append(-1)
            continue
        
        eval7_cards = []
        for c in hand_cards:
            rank = c.rank
            suit = c.suit.lower()
            eval7_cards.append(eval7.Card(rank + suit))
            
        scores.append(eval7.evaluate(eval7_cards))
        
    best_score = max(scores)
    winners = [i for i, s in enumerate(scores) if s == best_score]
    
    total_pot = sum(p.in_chips for p in players)
    payoffs = []
    win_amount = total_pot / len(winners)
    
    for i in range(len(players)):
        if i in winners:
            payoffs.append(win_amount - players[i].in_chips)
        else:
            payoffs.append(-players[i].in_chips)
            
    return payoffs

Judger.judge_game = optimized_judge_game
print("Optimization: Replaced RLCard Judger with Eval7 (High Performance)")

def play_vs_ai():
    env = PokerEnv()
    agent = Agent(env.observation_space_size, env.action_space_size)
    
    try:
        agent.load("poker_ai.pth")
        print("AI loaded successfully.")
    except:
        print("No trained model found. AI will play randomly.")
    
    print("Starting game vs AI!")
    print("You are Player 0. AI is Player 1.")
    
    while True:
        state = env.reset()
        done = False
        print("\n--- New Hand ---")
        
        while not done:
            # Display Game State
            pot = env.get_pot()
            community_cards = env.get_community_cards()
            stacks = env.get_stacks()
            my_hand = env.get_hand_cards(0)
            
            print(f"\nPot: {pot}")
            print(f"Community Cards: {community_cards}")
            print(f"Your Stack: {stacks[0]}, AI Stack: {stacks[1]}")
            print(f"Your Hand: {my_hand}")
            
            # Calculate and display equity using OMPEval
            equity = env.calculate_equity_ompeval(my_hand, community_cards)
            print(f"Your Equity (OMPEval): {equity:.1f}%")
            
            legal_actions = env.get_legal_actions()
            
            if env.current_player == 0:
                # Human turn
                print(f"Legal Actions: {legal_actions}")
                print("Actions: 0:Fold, 1:Check/Call, 2:Raise Half-Pot, 3:Raise Pot, 4:All-In")
                try:
                    action = int(input("Enter action: "))
                    if action not in legal_actions:
                        print(f"Invalid action. Legal actions are: {legal_actions}")
                        continue
                except ValueError:
                    print("Invalid input.")
                    continue
            else:
                # AI turn
                # We need to pass the AI's state. 
                # If it's AI's turn, env.last_state should be AI's state.
                # But wait, env.step() updates last_state to the NEXT player.
                # So if we are here, env.last_state IS the current player's state.
                action = agent.select_action(state, training=False)
                
                # Check legality for AI
                if action not in legal_actions:
                    # Fallback to Check/Call or Fold
                    if 1 in legal_actions: action = 1
                    elif 0 in legal_actions: action = 0
                    else: action = legal_actions[0]
                    
                action_str = ["Fold", "Check/Call", "Raise Half-Pot", "Raise Pot", "All-In"][action]
                print(f"AI chooses: {action_str}")
            
            next_state, reward, done, _ = env.step(action)
            state = next_state
            
            if done:
                print("\n--- Hand Over ---")
                print(f"Winner: Player {env.winner}")
                print(f"Your Hand: {env.get_hand_cards(0)}")
                print(f"AI Hand: {env.get_hand_cards(1)}")
                print(f"Community Cards: {env.get_community_cards()}")
                
                if env.winner == 0:
                    print("You Win!")
                elif env.winner == 1:
                    print("AI Wins!")
                else:
                    print("Tie!")

        play_again = input("\nPlay again? (y/n): ")
        if play_again.lower() != 'y':
            break

def main():
    parser = argparse.ArgumentParser(description="Texas Hold'em AI")
    parser.add_argument('--mode', type=str, default='play', choices=['train', 'play'], help='Mode: train or play')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(num_episodes=args.episodes)
    else:
        play_vs_ai()

if __name__ == "__main__":
    main()
