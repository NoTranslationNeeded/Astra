import rlcard
from rlcard.games.nolimitholdem.judger import Judger
from rlcard.games.base import Card
import numpy as np
import eval7

# Mock Player class
class Player:
    def __init__(self, id, chips):
        self.player_id = id
        self.remained_chips = chips
        self.in_chips = 0 # Chips put in pot this round/game
        self.status = 'alive' # or 'folded'

# Original judger for comparison
original_judger = Judger(np.random.RandomState(42))

# New judge_game using eval7
def judge_game_eval7(self, players, hands):
    # hands: list of list of RLCard Card objects
    # players: list of Player objects
    
    scores = []
    for hand_cards in hands:
        if not hand_cards: # Folded players might have None? RLCard usually passes cards for all alive players?
            # If a player folded, they shouldn't be in the showdown usually, 
            # but judge_game might be called with all players.
            # RLCard's judge_game iterates over 'hands' which corresponds to 'players'.
            # If a player folded, their hand might be None or empty?
            # Let's assume alive players have cards.
            scores.append(-1)
            continue
            
        # Convert to eval7 cards
        eval7_cards = []
        for c in hand_cards:
            rank = c.rank
            suit = c.suit.lower()
            # RLCard 'T' -> eval7 'T'
            eval7_cards.append(eval7.Card(rank + suit))
            
        # Evaluate
        score = eval7.evaluate(eval7_cards)
        scores.append(score)
        
    # Determine winner(s)
    # eval7: Higher score is better
    # We only consider players who are 'alive' (not folded)?
    # RLCard's judge_game assumes it's called at showdown for remaining players?
    # Or it handles folded players?
    # RLCard's judge_game doc says "Judge the winner of the game".
    # It usually iterates all players.
    
    # Let's look at how we should handle it.
    # We should find the max score among those who have cards.
    
    valid_scores = [s for s in scores if s != -1]
    if not valid_scores:
        return [0] * len(players) # Should not happen
        
    best_score = max(valid_scores)
    winners = [i for i, s in enumerate(scores) if s == best_score]
    
    # Calculate payoffs
    # Total pot is sum of in_chips of all players (including folded ones if passed)
    total_pot = sum(p.in_chips for p in players)
    
    payoffs = []
    win_amount = total_pot / len(winners)
    
    for i in range(len(players)):
        if i in winners:
            payoffs.append(win_amount - players[i].in_chips)
        else:
            payoffs.append(-players[i].in_chips)
            
    return payoffs

# Setup Test Data
p1 = Player(0, 100)
p1.in_chips = 10
p2 = Player(1, 100)
p2.in_chips = 10
players = [p1, p2]

# Hand 1: Royal Flush
hand1_strs = ['SA', 'SK', 'SQ', 'SJ', 'ST', 'H2', 'D3'] 
# Hand 2: Pair of 2s
hand2_strs = ['H2', 'C2', 'D5', 'D6', 'D7', 'S8', 'S9']

def str_to_card(s):
    return Card(s[0], s[1])

hand1_cards = [str_to_card(s) for s in hand1_strs]
hand2_cards = [str_to_card(s) for s in hand2_strs]

print("Original Judger Result:")
try:
    res = original_judger.judge_game(players, [hand1_cards, hand2_cards])
    print(res)
except Exception as e:
    print("Original Judger Failed:", e)

print("\nEval7 Judger Result:")
res_eval7 = judge_game_eval7(None, players, [hand1_cards, hand2_cards])
print(res_eval7)

# Verify
if res_eval7[0] > 0 and res_eval7[1] < 0:
    print("SUCCESS: Hand 1 won as expected.")
else:
    print("FAILURE: Unexpected result.")
