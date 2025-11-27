"""
Tournament-style Poker Environment
Multiple hands until one player's stack reaches 0
With blind level escalation and randomized starting stacks
"""
import rlcard
import numpy as np

# eval7 is optional - only needed for equity calculation
try:
    import eval7
    HAS_EVAL7 = True
except ImportError:
    HAS_EVAL7 = False
    print("! eval7 not available - equity calculation disabled (using fixed 0.5)")

from reward_functions import create_reward_function, REWARD_CONFIGS

class TournamentPokerEnv:
    """
    Tournament poker where players play multiple hands
    until one player loses all chips
    """
    
    def __init__(self, starting_chips=100, small_blind=1, big_blind=2, randomize_stacks=True, 
                 reward_type='icm_survival', reward_config=None):
        self.base_starting_chips = starting_chips
        self.base_small_blind = small_blind
        self.base_big_blind = big_blind
        self.randomize_stacks = randomize_stacks
        
        # Reward function configuration
        self.reward_type = reward_type
        if reward_config is None:
            reward_config = REWARD_CONFIGS.get('balanced', {})
            reward_config['big_blind'] = big_blind
        self.reward_function = create_reward_function(reward_type, reward_config)
        
        # Blind level structure: Fixed at 125/250 for Deep Stack play
        # The user requested BB to be fixed at 250.
        self.blind_levels = [
            (125, 250),
        ]
        self.hands_per_level = 999999  # Effectively infinite, blinds do not increase
        
        # RLCard base environment
        self.base_env = None
        self.chips = [starting_chips, starting_chips]
        self.hand_count = 0
        self.current_blind_level = 0
        self.small_blind = small_blind
        self.big_blind = big_blind
        
        # Action and observation space
        # User requested 7 actions: Fold, Check/Call, 33%, 75%, 100%, 150%, All-in
        self.action_space_size = 7
        self.observation_space_size = 55 + 5  # 55 (base) + 5 (equity + chip info + blind level)
        
        # Action Mapping: Agent Action Index -> RLCard Action ID
        # RLCard Actions: 
        # 0: FOLD, 1: CHECK_CALL, 2: HALF_POT (Unused), 3: POT, 4: ALL_IN
        # 5: 33_POT, 6: 75_POT, 7: 150_POT
        self.action_mapping = {
            0: 0,  # Fold
            1: 1,  # Check/Call
            2: 5,  # 33% Pot
            3: 6,  # 75% Pot
            4: 3,  # 100% Pot
            5: 7,  # 150% Pot
            6: 4   # All-in
        }
        # Reverse mapping for legal action calculation
        self.rlcard_to_agent_action = {v: k for k, v in self.action_mapping.items()}
        
        self.current_player = 0
        self.current_state = None  # Store current rlcard state
        self.tournament_over = False

    # ... (reset and _start_new_hand methods remain unchanged) ...

    def step(self, action):
        """Execute one action in the tournament"""
        # Map agent action (0-6) to RLCard action ID
        rlcard_action = self.action_mapping.get(action, 1) # Default to Check/Call if invalid
        
        # Execute action in current hand
        next_state, next_player = self.base_env.step(rlcard_action)
        self.current_state = next_state  # Store state for legal actions
        self.current_player = next_player
        
        # ... (rest of step method remains unchanged) ...
        
        # Check if hand is over
        if self.base_env.is_over():
            # Hand finished - update chips
            payoffs = self.base_env.get_payoffs()
            
            self.chips[0] += payoffs[0]
            self.chips[1] += payoffs[1]
            
            # Check if tournament is over
            if self.chips[0] <= 0 or self.chips[1] <= 0:
                self.tournament_over = True
                
                # Calculate final reward using reward function
                is_winner_0 = self.chips[0] > 0
                is_winner_1 = self.chips[1] > 0
                
                final_reward_0 = self.reward_function.calculate(
                    current_chips=self.chips[0],
                    starting_chips=self.starting_chips[0],
                    opponent_chips=self.chips[1],
                    opponent_starting_chips=self.starting_chips[1],
                    hands_played=self.hand_count,
                    is_winner=is_winner_0
                )
                
                final_reward_1 = self.reward_function.calculate(
                    current_chips=self.chips[1],
                    starting_chips=self.starting_chips[1],
                    opponent_chips=self.chips[0],
                    opponent_starting_chips=self.starting_chips[0],
                    hands_played=self.hand_count,
                    is_winner=is_winner_1
                )
                
                obs = np.zeros(self.observation_space_size, dtype=np.float32)
                done = True
                info = {
                    'tournament_winner': 0 if self.chips[0] > 0 else 1,
                    'hands_played': self.hand_count,
                    'final_chips': self.chips.copy(),
                    'starting_chips': self.starting_chips.copy()
                }
                
                return obs, final_reward_0, done, info
            else:
                # Tournament continues - start new hand
                # Dense Reward: Calculate reward for this hand (BB normalization)
                # Formula: (chip_payoff / big_blind) / 250.0
                bb_payoff_0 = payoffs[0] / self.big_blind
                bb_payoff_1 = payoffs[1] / self.big_blind
                
                hand_reward_0 = bb_payoff_0 / 250.0
                hand_reward_1 = bb_payoff_1 / 250.0
                
                # Extract card information from the ended hand
                card_info = self._extract_card_info(next_state)
                
                obs = self._start_new_hand()
                done = False
                info = {
                    'hand_ended': True,
                    'hand_payoffs': payoffs,
                    'current_chips': self.chips.copy(),
                    'blind_level': self.current_blind_level,
                    'cards': card_info
                }
                
                return obs, hand_reward_0, done, info
        else:
            # Hand continues
            obs = self._process_state(next_state, next_player)
            reward = 0.0
            done = False
            info = {}
            
            return obs, reward, done, info

    # ... (_process_state, _calculate_equity, _extract_card_info, _card_to_eval7 methods remain unchanged) ...

    def get_legal_actions(self):
        """Get currently legal actions from stored state"""
        if self.current_state and 'legal_actions' in self.current_state:
            rlcard_legal = list(self.current_state['legal_actions'].keys())
            
            # Filter and Map to Agent Actions
            agent_legal = []
            
            # Check Stage for Pre-flop masking
            # RLCard Stage: 0=Preflop, 1=Flop, 2=Turn, 3=River
            # We can access stage from current_state['raw_obs']['stage'] (if available) or env.game.stage
            is_preflop = False
            if self.base_env and hasattr(self.base_env, 'game'):
                 # Stage enum: PREFLOP=0
                 if self.base_env.game.stage == 0: 
                     is_preflop = True
            
            for action_id in rlcard_legal:
                if action_id in self.rlcard_to_agent_action:
                    agent_action = self.rlcard_to_agent_action[action_id]
                    
                    # Pre-flop Masking Rule:
                    # Allowed: Fold(0), Check/Call(1), 75%(3), 100%(4), All-in(6)
                    # Forbidden: 33%(2), 150%(5)
                    if is_preflop:
                        if agent_action in [2, 5]: # 33% and 150%
                            continue
                            
                    agent_legal.append(agent_action)
            
            return sorted(agent_legal)
        return []
        
    def reset(self):
        """Reset tournament - both players start with randomized chips"""
        # Randomize starting stacks for better generalization
        if self.randomize_stacks:
            # Variable Stack Depth: 1 BB to 250 BB
            bb_depth = np.random.randint(1, 251)
            
            # Random Big Blind: 250 to 5000
            # Ensure even number for integer Small Blind
            new_bb = np.random.randint(250, 5001)
            if new_bb % 2 != 0:
                new_bb += 1
            new_sb = int(new_bb / 2)
            
            # Calculate Starting Chips
            start_chips = bb_depth * new_bb
            
            self.chips = [float(start_chips), float(start_chips)]
            self.starting_chips = self.chips.copy()
            
            # Update blind levels for this episode (Fixed for the episode)
            self.blind_levels = [(new_sb, new_bb)]
            self.current_blind_level = 0
            self.small_blind = new_sb
            self.big_blind = new_bb
            
        else:
            # Default fixed deep stack (25000 chips, 100/200 blinds)
            self.chips = [25000.0, 25000.0]

    def _process_state(self, state, player_id):
        """
        Convert RLCard state to observation vector
        Adds chip information and blind level
        """
        # Base observation (54 elements)
        base_obs = state['obs']
        
        # Calculate equity (0.5 if eval7 not available)
        try:
            equity = self._calculate_equity(state, player_id)
        except:
            equity = 0.5
        
        # Chip information (normalized)
        # Normalize by Max Possible Depth (250 BB) relative to current Big Blind
        # This makes the input "How many BBs do I have?" scaled to 0-1 range (where 1 = 250 BB)
        # If chips > 250 BB (e.g. after winning), it can go > 1.0, which is fine.
        MAX_BB_DEPTH = 250.0
        current_bb = self.big_blind
        
        my_chips_bb = (self.chips[player_id] / current_bb)
        opp_chips_bb = (self.chips[1 - player_id] / current_bb)
        
        my_chips_norm = my_chips_bb / MAX_BB_DEPTH
        opp_chips_norm = opp_chips_bb / MAX_BB_DEPTH
        
        chip_ratio = my_chips_norm / (my_chips_norm + opp_chips_norm) if (my_chips_norm + opp_chips_norm) > 0 else 0.5
        
        # Pot ratio: Pot / My Chips
        # If I have 0 chips, pot ratio is effectively infinite (cap at 10?)
        pot_ratio = (self.small_blind + self.big_blind) / self.chips[player_id] if self.chips[player_id] > 0 else 10.0
        
        # Blind level (normalized 0-1)
        if len(self.blind_levels) > 1:
            blind_level_normalized = self.current_blind_level / (len(self.blind_levels) - 1)
        else:
            blind_level_normalized = 0.0  # Fixed blind level
        
        # Combine into observation
        obs = np.concatenate([
            base_obs.flatten(),
            [equity],
            [my_chips_norm],
            [opp_chips_norm],
            [chip_ratio],
            [pot_ratio],
            [blind_level_normalized]
        ]).astype(np.float32)
        
        return obs
    
    def _start_new_hand(self):
        """Start a new hand within the tournament"""
        self.hand_count += 1
        
        # Update blind level every hands_per_level hands
        new_blind_level = min(
            (self.hand_count - 1) // self.hands_per_level,
            len(self.blind_levels) - 1
        )
        
        if new_blind_level != self.current_blind_level:
            self.current_blind_level = new_blind_level
            self.small_blind, self.big_blind = self.blind_levels[self.current_blind_level]
            print(f" Blind level increased to {self.small_blind}/{self.big_blind} (Level {self.current_blind_level + 1})")
        
        # Toggle dealer for subsequent hands (Hand 1 uses the one set in reset)
        if self.hand_count > 1:
            self.current_dealer_id = 1 - self.current_dealer_id

        # Create new RLCard environment for this hand
        config = {
            'seed': np.random.randint(0, 100000),
            'game_num_players': 2,
            'dealer_id': self.current_dealer_id
        }
        self.base_env = rlcard.make('no-limit-holdem', config=config)
        
        # [NEW] Configure blinds explicitly
        if hasattr(self.base_env.game, 'small_blind'):
            self.base_env.game.small_blind = self.small_blind
        if hasattr(self.base_env.game, 'big_blind'):
            self.base_env.game.big_blind = self.big_blind
        
        # Get initial state
        state, player_id = self.base_env.reset()
        
        # [NEW] Synchronize chip stacks
        # rlcard's reset() automatically posts blinds, so we must account for them
        for i in range(2):
            player = self.base_env.game.players[i]
            # Calculate remaining chips: Tournament Stack - Chips already put in pot (Blinds)
            # Cast to int to prevent rlcard TypeError (float64 -> int64 casting error)
            player.remained_chips = int(max(0, self.chips[i] - player.in_chips))
            
        self.current_state = state  # Store state for legal actions
        self.current_player = player_id
        
        # Process observation with chip context
        obs = self._process_state(state, player_id)
        
        return obs
    
    def step(self, action):
        """Execute one action in the tournament"""
        # Execute action in current hand
        next_state, next_player = self.base_env.step(action)
        self.current_state = next_state  # Store state for legal actions
        self.current_player = next_player
        
        # Check if hand is over
        if self.base_env.is_over():
            # Hand finished - update chips
            payoffs = self.base_env.get_payoffs()
            
            self.chips[0] += payoffs[0]
            self.chips[1] += payoffs[1]
            
            # Check if tournament is over
            if self.chips[0] <= 0 or self.chips[1] <= 0:
                self.tournament_over = True
                
                # Calculate final reward using reward function
                is_winner_0 = self.chips[0] > 0
                is_winner_1 = self.chips[1] > 0
                
                final_reward_0 = self.reward_function.calculate(
                    current_chips=self.chips[0],
                    starting_chips=self.starting_chips[0],
                    opponent_chips=self.chips[1],
                    opponent_starting_chips=self.starting_chips[1],
                    hands_played=self.hand_count,
                    is_winner=is_winner_0
                )
                
                final_reward_1 = self.reward_function.calculate(
                    current_chips=self.chips[1],
                    starting_chips=self.starting_chips[1],
                    opponent_chips=self.chips[0],
                    opponent_starting_chips=self.starting_chips[0],
                    hands_played=self.hand_count,
                    is_winner=is_winner_1
                )
                
                obs = np.zeros(self.observation_space_size, dtype=np.float32)
                done = True
                info = {
                    'tournament_winner': 0 if self.chips[0] > 0 else 1,
                    'hands_played': self.hand_count,
                    'final_chips': self.chips.copy(),
                    'starting_chips': self.starting_chips.copy()
                }
                
                return obs, final_reward_0, done, info
            else:
                # Tournament continues - start new hand
                hand_reward_0 = 0.0
                hand_reward_1 = 0.0
                
                # Extract card information from the ended hand
                card_info = self._extract_card_info(next_state)
                
                obs = self._start_new_hand()
                done = False
                info = {
                    'hand_ended': True,
                    'hand_payoffs': payoffs,
                    'current_chips': self.chips.copy(),
                    'blind_level': self.current_blind_level,
                    'cards': card_info
                }
                
                return obs, hand_reward_0, done, info
        else:
            # Hand continues
            obs = self._process_state(next_state, next_player)
            reward = 0.0
            done = False
            info = {}
            
            return obs, reward, done, info
    
    def _process_state(self, state, player_id):
        """
        Convert RLCard state to observation vector
        Adds chip information and blind level
        """
        # Base observation (54 elements)
        base_obs = state['obs']
        
        # Calculate equity (0.5 if eval7 not available)
        try:
            equity = self._calculate_equity(state, player_id)
        except:
            equity = 0.5
        
        # Chip information (normalized)
        # Normalize by max possible starting chips (50000) or current pot context
        MAX_CHIPS = 50000.0
        my_chips = self.chips[player_id] / MAX_CHIPS
        opp_chips = self.chips[1 - player_id] / MAX_CHIPS
        chip_ratio = my_chips / (my_chips + opp_chips) if (my_chips + opp_chips) > 0 else 0.5
        pot_ratio = (self.small_blind + self.big_blind) / self.chips[player_id] if self.chips[player_id] > 0 else 0.0
        
        # Blind level (normalized 0-1)
        if len(self.blind_levels) > 1:
            blind_level_normalized = self.current_blind_level / (len(self.blind_levels) - 1)
        else:
            blind_level_normalized = 0.0  # Fixed blind level
        
        # Combine into observation
        obs = np.concatenate([
            base_obs.flatten(),
            [equity],
            [my_chips],
            [opp_chips],
            [chip_ratio],
            [pot_ratio],
            [blind_level_normalized]
        ]).astype(np.float32)
        
        return obs
    
    def _calculate_equity(self, state, player_id):
        """Calculate hand equity using eval7 (if available)"""
        if not HAS_EVAL7:
            return 0.5  # Neutral equity if eval7 not available
        
        try:
            # Get hand and community cards
            raw_obs = state['raw_obs']
            hand = raw_obs['hand']
            public_cards = raw_obs['public_cards']
            
            # Convert to eval7 format
            hand_eval7 = [self._card_to_eval7(c) for c in hand]
            public_eval7 = [self._card_to_eval7(c) for c in public_cards]
            
            # Monte Carlo simulation
            equity = eval7.py_hand_vs_range_monte_carlo(
                hand_eval7,
                eval7.HandRange("xx"),
                public_eval7,
                100
            )
            
            return equity
        except:
            return 0.5
    
    def _extract_card_info(self, state):
        """Extract card information from RLCard state"""
        try:
            raw_obs = state['raw_obs']
            
            # Get hands for both players (if available)
            hands = raw_obs.get('all_hands', [raw_obs.get('hand', [])])
            
            # Get community cards
            public_cards = raw_obs.get('public_cards', [])
            
            # Convert card strings to readable format
            def format_card(card_str):
                """Convert RLCard format to readable (e.g., 'SA' -> 'As')"""
                suit_map = {'S': 's', 'H': 'h', 'D': 'd', 'C': 'c'}
                if len(card_str) >= 2:
                    suit = suit_map.get(card_str[0], card_str[0])
                    rank = card_str[1]
                    return f"{rank}{suit}"
                return card_str
            
            return {
                'player_0_hand': [format_card(c) for c in (hands[0] if len(hands) > 0 else [])],
                'player_1_hand': [format_card(c) for c in (hands[1] if len(hands) > 1 else [])],
                'community': [format_card(c) for c in public_cards]
            }
        except Exception as e:
            return {
                'player_0_hand': [],
                'player_1_hand': [],
                'community': []
            }
    
    def _card_to_eval7(self, card_str):
        """Convert RLCard card string to eval7.Card"""
        if not HAS_EVAL7:
            return None
        
        rank_map = {'A': 14, 'K': 13, 'Q': 12, 'J': 11, 'T': 10}
        suit_map = {'S': 1, 'H': 2, 'D': 4, 'C': 8}
        
        rank = rank_map.get(card_str[1], int(card_str[1]))
        suit = suit_map[card_str[0]]
        
        return eval7.Card(rank * 4 + suit - 5)
    
    def get_legal_actions(self):
        """Get currently legal actions from stored state"""
        if self.current_state and 'legal_actions' in self.current_state:
            rlcard_legal = list(self.current_state['legal_actions'].keys())
            
            # Filter and Map to Agent Actions
            agent_legal = []
            
            # Check Stage for Pre-flop masking (REMOVED as per user request)
            # User requested ALL 7 actions to be available at ALL times.
            
            for action_id in rlcard_legal:
                if action_id in self.rlcard_to_agent_action:
                    agent_action = self.rlcard_to_agent_action[action_id]
                    agent_legal.append(agent_action)
            
            return sorted(agent_legal)
        return []


if __name__ == "__main__":
    # Test the tournament environment
    env = TournamentPokerEnv(randomize_stacks=True)
    
    print("=" * 80)
    print("Testing Tournament Poker Environment (Random Agent)")
    print("=" * 80)
    
    # Action name map for display
    action_names = {
        0: "Fold",
        1: "Check/Call",
        2: "Bet 33% Pot",
        3: "Bet 75% Pot",
        4: "Bet 100% Pot",
        5: "Bet 150% Pot",
        6: "All-in"
    }
    
    for episode in range(1): # Run 1 episode as requested
        print(f"\n Episode {episode + 1}")
        obs = env.reset()
        print(f"Starting chips - Player 0: {env.chips[0]:.1f}, Player 1: {env.chips[1]:.1f}")
        print(f"Initial blinds: {env.small_blind}/{env.big_blind} (Depth: {env.chips[0]/env.big_blind:.1f} BB)\n")
        
        done = False
        hand_count = 0
        
        while not done and hand_count < 100:
            legal_actions = env.get_legal_actions()
            if not legal_actions:
                break
            
            # Pick random action
            action = np.random.choice(legal_actions)
            action_name = action_names.get(action, "Unknown")
            
            print(f"  Hand {env.hand_count} | Player {env.current_player} acts: {action_name} (Action {action})")
            
            obs, reward, done, info = env.step(action)
            
            if info.get('hand_ended'):
                print(f"  -- Hand Ended --")
                print(f"     Result: {info['hand_payoffs']}")
                print(f"     Chips: P0={env.chips[0]:.1f}, P1={env.chips[1]:.1f}")
                print(f"     Cards: {info['cards']}")
                print("-" * 40)
                hand_count += 1
            
            if done:
                print(f"\n Tournament Over!")
                print(f"Winner: Player {info['tournament_winner']}")
                print(f"Total hands: {info['hands_played']}")
