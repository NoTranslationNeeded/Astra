import rlcard
from rlcard.agents import RandomAgent
import numpy as np
import torch
import eval7

class PokerEnv:
    def __init__(self):
        # Initialize RLCard environment for No-Limit Texas Hold'em
        self.env = rlcard.make('no-limit-holdem', config={'seed': 42})
        
        # RLCard Action Space for No-Limit Hold'em:
        # 0: Fold
        # 1: Check/Call
        # 2: Raise Half Pot
        # 3: Raise Pot
        # 4: All-in
        self.action_space_size = self.env.num_actions
        
        # Observation space
        # RLCard returns a dictionary. We need to flatten it for our DQN.
        # 'obs': 54 cards (one-hot) + betting history etc.
        # Standard RLCard NLH observation size is 54.
        # We add 1 for Equity.
        self.observation_space_size = self.env.state_shape[0][0] + 1

        self.game_over = False
        self.winner = -1
        self.current_player = 0
        
        # Keep track of last state for each player to return in step
        self.last_state = None

    def reset(self):
        state, player_id = self.env.reset()
        self.current_player = player_id
        self.game_over = False
        self.winner = -1
        self.last_state = state
        return self._process_state(state)

    def _process_state(self, state):
        # Extract 'obs' from the state dictionary
        obs = state['obs']
        
        # Calculate Equity
        # We need raw cards to calculate equity
        # state['raw_obs'] contains 'hand' and 'public_cards'
        raw_obs = state['raw_obs']
        hand = raw_obs['hand'] # List of strings e.g. ['SA', 'HT']
        board = raw_obs['public_cards'] # List of strings
        
        # Use eval7 for equity calculation (Faster Approximation)
        # We use the method we implemented: calculate_equity_eval7
        # Note: calculate_equity_eval7 returns percentage 0-100.
        # We should normalize to 0-1 for NN input.
        equity = self.calculate_equity_eval7(hand, board) / 100.0
        
        # Append equity to obs
        # obs is a numpy array. We need to append.
        new_obs = np.append(obs, equity)
        
        return new_obs.astype(np.float32)

    def step(self, action):
        # Capture who is acting before the step
        acting_player = self.current_player
        
        # RLCard expects action as an integer
        # We need to ensure the action is legal.
        # RLCard provides 'legal_actions' in the state.
        
        legal_actions = list(self.last_state['legal_actions'].keys())
        if action not in legal_actions:
            # Fallback: Check/Call (usually 1 or 0) or Fold
            # Let's pick the first legal action
            action = legal_actions[0]
            
        next_state, player_id = self.env.step(action)
        self.current_player = player_id
        self.last_state = next_state
        
        reward = 0
        done = self.env.is_over()
        
        if done:
            self.game_over = True
            # RLCard returns payoffs for all players
            payoffs = self.env.get_payoffs()
            
            # Since the game is over, we can determine the winner for display
            if payoffs[0] > 0:
                self.winner = 0
            elif payoffs[1] > 0:
                self.winner = 1
            else:
                self.winner = -1 # Tie
                
            # Return reward for the player who just acted
            reward = payoffs[acting_player]

        return self._process_state(next_state), reward, done, {}

    def calculate_equity_eval7(self, hand_cards, board_cards):
        # hand_cards: list of RLCard strings e.g. ['SA', 'HT']
        # board_cards: list of RLCard strings e.g. ['D3', 'C9', 'SQ']
        
        # Convert RLCard format to eval7 format
        # RLCard: Suit (0) + Rank (1) e.g. 'SA'
        # eval7: Rank + Suit e.g. 'As'
        
        def to_eval7_card(rlcard_str):
            suit = rlcard_str[0].lower()
            rank = rlcard_str[1]
            return eval7.Card(rank + suit)
            
        try:
            hero_hand = [to_eval7_card(c) for c in hand_cards]
            board = [to_eval7_card(c) for c in board_cards]
        except:
            return 0.0
        
        # Monte Carlo Simulation
        deck = eval7.Deck()
        # Remove known cards
        known_cards = hero_hand + board
        for c in known_cards:
            if c in deck.cards:
                deck.cards.remove(c)
        
        # Convert remaining deck to list for sampling
        remaining_deck = list(deck.cards)
        
        wins = 0
        n_sims = 100 # Fast approximation
        import random
        
        for _ in range(n_sims):
            # Sample necessary cards (2 for villain + missing board)
            n_board_missing = 5 - len(board)
            n_needed = 2 + n_board_missing
            
            sampled_cards = random.sample(remaining_deck, n_needed)
            
            villain_hand = sampled_cards[:2]
            board_extension = sampled_cards[2:]
            
            run_board = board + board_extension
                
            # Evaluate
            hero_score = eval7.evaluate(hero_hand + run_board)
            villain_score = eval7.evaluate(villain_hand + run_board)
            
            if hero_score > villain_score:
                wins += 1
            elif hero_score == villain_score:
                wins += 0.5
                
        return (wins / n_sims) * 100.0

    def get_legal_actions(self):
        if self.last_state:
            return list(self.last_state['legal_actions'].keys())
        return []

    # Helper for main.py to display cards
    def get_hand_cards(self, player_id):
        # RLCard state has 'raw_obs' which contains 'hand'
        # state['raw_obs']['hand'] is a list of strings like ['D3', 'H5']
        if self.last_state:
             # Note: last_state is for the current_player.
             # If we want P0's hand, and it's P1's turn, we might not see P0's hand in P1's observation.
             # RLCard environment has `env.get_state(player_id)`?
             # No, `env.get_state(player_id)` exists.
             state = self.env.get_state(player_id)
             return state['raw_obs']['hand']
        return []

    def get_community_cards(self):
        if self.last_state:
            return self.last_state['raw_obs']['public_cards']
        return []
    
    def get_stacks(self):
        # RLCard NLH raw_obs has 'my_chips', 'all_chips'?
        if self.last_state:
            return [self.env.game.players[0].remained_chips, self.env.game.players[1].remained_chips]
        return [0, 0]

    def get_pot(self):
        # Accessing internal game state for display
        return self.env.game.dealer.pot
