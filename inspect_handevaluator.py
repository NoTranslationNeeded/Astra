import ompeval
from ompeval import HandEvaluator

print(dir(HandEvaluator))
evaluator = HandEvaluator()
try:
    evaluator.evaluate(12345)
except Exception as e:
    print("evaluate(int) error:", e)
