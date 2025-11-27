import rlcard
from rlcard.games.nolimitholdem.judger import Judger
import inspect

print("Judger methods:")
print(dir(Judger))

print("\njudge_game signature:")
try:
    print(inspect.signature(Judger.judge_game))
except:
    print("Could not get signature")

print("\nJudger.judge_game docstring:")
print(Judger.judge_game.__doc__)
