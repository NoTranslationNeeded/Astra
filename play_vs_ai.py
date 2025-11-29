"""
Play Poker vs AI
Interactive heads-up no-limit Texas Hold'em against a trained AI

Usage:
    python play_vs_ai.py --checkpoint <path_to_checkpoint>
"""
import ray
from ray.rllib.algorithms.ppo import PPO
from tournament_poker_env import TournamentPokerEnv
import numpy as np
import argparse
import os


class HumanVsAI:
    def __init__(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path
        self.env = None
        self.policy = None
        
        # Card symbols
        self.suits = {'S': '♠', 'H': '♥', 'D': '♦', 'C': '♣'}
        self.action_names = {
            0: "Fold",
            1: "Check/Call",
            2: "Bet 33% pot",
            3: "Bet 75% pot",
            4: "Bet 100% pot",
            5: "Bet 150% pot",
            6: "All-in"
        }
    
    def load_ai(self):
        """Load the trained AI policy"""
        print("\n" + "=" * 80)
        print("LOADING AI MODEL")
        print("=" * 80)
        
        try:
            from ray.rllib.algorithms.ppo import PPO
            from ray.tune.registry import register_env
            from tournament_pettingzoo import TournamentPokerParallelEnv
            from ray.rllib.models import ModelCatalog
            from transformer_model import TransformerPokerModel
            
            # Convert to absolute path (PyArrow requires absolute paths)
            checkpoint_abs = os.path.abspath(self.checkpoint_path)
            print(f"Checkpoint absolute path: {checkpoint_abs}")
            
            # Register custom model
            ModelCatalog.register_custom_model("transformer_poker", TransformerPokerModel)
            
            # Register environment (needed for checkpoint loading)
            def env_creator(config):
                return TournamentPokerParallelEnv(
                    starting_chips=100,
                    randomize_stacks=True,
                    max_hands=config.get("max_hands", 1000)
                )
            
            ray.init(ignore_reinit_error=True, include_dashboard=False)
            register_env("tournament_poker_dense", env_creator)
            
            # Load algorithm and extract policy
            print("Loading algorithm...")
            algo = PPO.from_checkpoint(checkpoint_abs)
            self.policy = algo.get_policy("player_0")
            print(f"AI loaded successfully!")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("   Make sure the path is correct and Ray versions match.")
            raise
    
    def format_card(self, card_str):
        """Convert RLCard format to readable (e.g., 'SA' -> 'A♠')"""
        if len(card_str) < 2:
            return "??"
            
        # Check if first char is suit
        if card_str[0] in self.suits:
            suit = self.suits[card_str[0]]
            rank = card_str[1]
            return f"{rank}{suit}"
        # Check if second char is suit (e.g. 'Ah')
        elif card_str[1].upper() in self.suits:
            suit = self.suits[card_str[1].upper()]
            rank = card_str[0]
            return f"{rank}{suit}"
            
        return card_str
    
    def get_human_hand(self):
        """Get Human (Player 0) hand directly from game engine"""
        if not hasattr(self.env, 'base_env'):
            return []
            
        try:
            # Access RLCard player object
            player = self.env.base_env.game.players[0]
            # Convert Card objects to string format (e.g. 'SA', 'HT')
            return [f"{c.suit}{c.rank}" for c in player.hand]
        except Exception:
            return []

    def display_hand_start(self, hand_num):
        """Display hand header with cards"""
        print("\n" + "═" * 80)
        print(f"{'  HAND ' + str(hand_num) + ' ':═^80}")
        print("═" * 80)
        
        # Get and display human hand
        cards = self.get_human_hand()
        if cards:
            card1 = self.format_card(cards[0])
            card2 = self.format_card(cards[1])
            print(f"\nYOUR HAND: [{card1}] [{card2}]")
        
        # Blind info
        print(f"Blinds: {self.env.small_blind}/{self.env.big_blind}")
        print(f"Your stack: {self.env.chips[0]:.0f} chips ({self.env.chips[0]/self.env.big_blind:.1f} BB)")
        print(f"AI stack: {self.env.chips[1]:.0f} chips ({self.env.chips[1]/self.env.big_blind:.1f} BB)")
    
    def display_game_state(self, state, pot_size):
        """Display community cards and pot"""
        raw_obs = state.get('raw_obs', {})
        public_cards = raw_obs.get('public_cards', [])
        
        print("\n" + "─" * 80)
        
        # Community cards
        if public_cards:
            formatted_cards = [self.format_card(c) for c in public_cards]
            cards_str = " ".join([f"[{c}]" for c in formatted_cards])
            print(f"COMMUNITY: {cards_str}")
        else:
            print(f"COMMUNITY: (Pre-flop)")
        
        # Pot
        print(f"POT: {pot_size:.0f} chips")
        print("─" * 80)
    
    def get_human_action(self, legal_actions, pot_size):
        """Get action from human player"""
        # Display hand again for convenience
        cards = self.get_human_hand()
        if cards:
            c1 = self.format_card(cards[0])
            c2 = self.format_card(cards[1])
            print(f"\nYOUR HAND: [{c1}] [{c2}]")

        print("\nAVAILABLE ACTIONS:")
        print("─" * 40)
        
        action_map = {}
        choice_num = 1
        for action_id in sorted(legal_actions):
            action_name = self.action_names.get(action_id, f"Action {action_id}")
            
            # Add extra info for bet actions
            if action_id == 1:  # Call
                action_name = "Check/Call"
            elif action_id == 2:
                amount = int(pot_size * 0.33)
                action_name = f"Bet 33% pot ({amount} chips)"
            elif action_id == 3:
                amount = int(pot_size * 0.75)
                action_name = f"Bet 75% pot ({amount} chips)"
            elif action_id == 4:
                amount = int(pot_size * 1.0)
                action_name = f"Bet 100% pot ({amount} chips)"
            elif action_id == 5:
                amount = int(pot_size * 1.5)
                action_name = f"Bet 150% pot ({amount} chips)"
            elif action_id == 6:
                action_name = f"All-in ({self.env.chips[0]:.0f} chips)"
            
            print(f"  [{choice_num}] {action_name}")
            action_map[choice_num] = action_id
            choice_num += 1
        
        print("─" * 40)
        
        # Get input
        while True:
            try:
                choice = input(f"Your choice (1-{len(action_map)}): ").strip()
                choice_int = int(choice)
                if choice_int in action_map:
                    return action_map[choice_int]
                else:
                    print(f"Invalid choice. Please choose 1-{len(action_map)}")
            except ValueError:
                print("Please enter a number")
            except KeyboardInterrupt:
                print("\n\nGame interrupted. Goodbye!")
                raise
    
    def play_tournament(self):
        """Play a tournament against AI"""
        self.env = TournamentPokerEnv(randomize_stacks=True)
        obs = self.env.reset()
        
        print("\n" + "═" * 80)
        print("TOURNAMENT START")
        print("═" * 80)
        print(f"Starting chips: {self.env.chips[0]:.0f}")
        print(f"Initial blinds: {self.env.small_blind}/{self.env.big_blind}")
        print(f"Play until one player has 0 chips (max 200 hands)")
        
        done = False
        current_hand = 0
        hand_just_started = False
        
        while not done:
            current_player = self.env.current_player
            
            # Display hand start
            if self.env.hand_count != current_hand:
                current_hand = self.env.hand_count
                hand_just_started = True
                
                # Get player's hole cards
                # raw_obs = self.env.current_state.get('raw_obs', {})
                # player_hand = raw_obs.get('hand', [])
                self.display_hand_start(current_hand)
            
            # Get game state
            pot_size = 0
            public_cards = []
            if hasattr(self.env, 'base_env'):
                game = self.env.base_env.game
                public_cards = [str(c) for c in game.public_cards] if game.public_cards else []
                
                # Calculate Pot Size
                # RLCard's in_chips appears to be cumulative for the hand and persists across rounds.
                # dealer.pot is updated at round end but in_chips is not cleared immediately.
                # Therefore, summing them causes double counting.
                # Using max() handles both:
                # 1. Active betting: sum(in_chips) > dealer.pot -> Use sum(in_chips)
                # 2. Folded players/Pot collection: dealer.pot might be larger -> Use dealer.pot
                pot_size = max(game.dealer.pot, sum([p.in_chips for p in game.players]))
                
                print(f"DEBUG: Dealer Pot: {game.dealer.pot}, In Chips: {[p.in_chips for p in game.players]}, Total: {pot_size}")
            
            # Display community cards
            if not hand_just_started or current_player == 0:
                self.display_game_state(self.env.current_state, pot_size)
            hand_just_started = False
            
            # Choose action
            legal_actions = self.env.get_legal_actions()
            
            if current_player == 0:
                # Human player
                action = self.get_human_action(legal_actions, pot_size)
                print(f"\nYou chose: {self.action_names.get(action, f'Action {action}')}")
            else:
                # AI player
                print("\nAI is thinking...")
                action, _, _ = self.policy.compute_single_action(obs, explore=False)
                print(f"AI chose: {self.action_names.get(action, f'Action {action}')}")
                input("   Press Enter to continue...")
            
            # Execute action
            obs, reward, done, info = self.env.step(action)
            
            # Check if hand ended
            if info.get('hand_ended'):
                payoffs = info['hand_payoffs']
                print("\n" + "─" * 80)
                print("HAND RESULT")
                print("─" * 80)
                
                # Display cards and ranks
                cards_info = info.get('cards', {})
                p0_hand = cards_info.get('player0', [])
                p1_hand = cards_info.get('player1', [])
                p0_rank = cards_info.get('p0_rank', 'Unknown')
                p1_rank = cards_info.get('p1_rank', 'Unknown')
                p0_best = cards_info.get('p0_best_cards', [])
                p1_best = cards_info.get('p1_best_cards', [])
                
                # Format cards
                h0_str = " ".join([f"[{self.format_card(c)}]" for c in p0_hand])
                h1_str = " ".join([f"[{self.format_card(c)}]" for c in p1_hand])
                
                # Format best 5 cards
                best0_str = " ".join([f"[{self.format_card(c)}]" for c in p0_best])
                best1_str = " ".join([f"[{self.format_card(c)}]" for c in p1_best])
                
                print(f"Your Hand: {h0_str}  ({p0_rank})")
                if best0_str:
                    print(f"           Best 5: {best0_str}")
                    
                print(f"AI Hand:   {h1_str}  ({p1_rank})")
                if best1_str:
                    print(f"           Best 5: {best1_str}")
                print("─" * 40)
                
                if payoffs[0] > 0:
                    print(f"You won {payoffs[0]:+.0f} chips!")
                elif payoffs[0] < 0:
                    print(f"You lost {abs(payoffs[0]):.0f} chips")
                else:
                    print(f"Split pot")
                
                print(f"\nYour stack: {self.env.chips[0]:.0f} chips")
                print(f"AI stack: {self.env.chips[1]:.0f} chips")
                print("─" * 80)
                
                if not done:
                    input("\n   Press Enter for next hand...")
        
        # Tournament over
        print("\n" + "═" * 80)
        print("TOURNAMENT OVER")
        print("═" * 80)
        
        winner = info.get('tournament_winner', -1)
        if winner == 0:
            print("CONGRATULATIONS! You won the tournament!")
        elif winner == 1:
            print("AI won the tournament. Better luck next time!")
        else:
            print("It's a tie!")
        
        print(f"\nStatistics:")
        print(f"   Hands played: {info.get('hands_played', 0)}")
        print(f"   Final chips - You: {info.get('final_chips', [0,0])[0]:.0f}, AI: {info.get('final_chips', [0,0])[1]:.0f}")
        print("═" * 80)


def main():
    parser = argparse.ArgumentParser(description="Play poker against a trained AI")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to AI checkpoint"
    )
    
    args = parser.parse_args()
    
    # Verify checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        print("\nTip: Find checkpoints in ray_results/omega/PPO_*/checkpoint_*")
        return
    
    try:
        game = HumanVsAI(args.checkpoint)
        game.load_ai()
        game.play_tournament()
    except KeyboardInterrupt:
        print("\n\nThanks for playing!")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
