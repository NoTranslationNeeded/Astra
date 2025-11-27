try:
    from rlcard.games.nolimitholdem.game import Action
    print("Found Action in rlcard.games.nolimitholdem.game")
    for a in Action:
        print(f"{a.name}: {a.value}")
except ImportError:
    print("Could not import Action from rlcard.games.nolimitholdem.game")
    
    # Try another path
    try:
        from rlcard.games.nolimitholdem import Action
        print("Found Action in rlcard.games.nolimitholdem")
        for a in Action:
            print(f"{a.name}: {a.value}")
    except ImportError:
        print("Could not import Action from rlcard.games.nolimitholdem")
