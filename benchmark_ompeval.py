import time
from poker_env import PokerEnv

env = PokerEnv()
hand = ['SA', 'HK']
board = ['D3', 'C9', 'SQ']

print("Benchmarking ompeval equity calculation...")
start = time.time()
for i in range(10):
    eq = env.calculate_equity_ompeval(hand, board)
end = time.time()

print(f"10 calculations took {end - start:.4f} seconds")
print(f"Average: {(end - start) / 10:.4f} seconds per call")
