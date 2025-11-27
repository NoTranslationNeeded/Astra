# Texas Hold'em AI Walkthrough

## Setup
1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    (Ensure you have `torch` and `numpy` installed)

## Training the AI
To train the AI from scratch, run:
```bash
python main.py --mode train --episodes 1000
```
- This will train the AI for 1000 episodes (hands).
- The model will be saved to `poker_ai.pth`.
- You will see progress logs in the terminal.

## Playing against the AI
Once trained (or to play against a random/untrained AI), run:
```bash
python main.py --mode play
```
- You will be Player 0.
- The AI is Player 1.
- Follow the on-screen prompts to Fold, Call, or Raise.

## Files Overview
- `poker_env.py`: The game engine (Deck, Rules, Hand Evaluation).
- `agent.py`: The Brain (DQN Neural Network).
- `trainer.py`: The Gym (Training loop).
- `main.py`: The Interface.
