import ompeval
from ompeval import CardRange, Hand, HandEvaluator

print("Exploring ompeval...")

try:
    cr = CardRange("AsKs")
    print("CardRange created")
    # Is it iterable?
    try:
        for h in cr:
            print("Item in CardRange:", h, type(h))
            break
    except:
        print("CardRange not iterable")

except Exception as e:
    print("CardRange error:", e)

# Try Hand with list of ints
try:
    # 0 to 51
    h = Hand([0, 1, 2, 3, 4]) # 5 cards
    print("Hand([0,1,2,3,4]) worked:", h)
except Exception as e:
    print("Hand([list]) error:", e)

# Try to find mapping
# If Hand takes int mask, we need to know how to build it.
# Maybe there is a static method?
