import ompeval
from ompeval import EquityCalculator, CardRange

try:
    calc = EquityCalculator()
    
    # Create CardRange objects
    r1 = CardRange("AhKh")
    r2 = CardRange("random")
    
    ranges = [r1, r2]
    
    # Board is int (bitmask). 0 for empty.
    # If we had board cards, we'd need to convert them to bitmask.
    # ompeval.Hand might have a value we can use? Or we need to construct it.
    # For now, empty board.
    
    print("Starting calculation...")
    calc.start(ranges, 0)
    calc.wait()
    results = calc.get_results()
    
    # Results object likely has 'equity' attribute which is a list?
    # Or 'wins', 'hands', etc.
    print("Results Type:", type(results))
    print("Results Dir:", dir(results))
    
    # Check equity
    # Usually results.equity is a list of equities for each player
    print("Equity:", results.equity)

except Exception as e:
    print("Error:", e)
