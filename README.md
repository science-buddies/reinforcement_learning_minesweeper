# 🧠 Minesweeper Reinforcement Learning Project

This project trains and evaluates reinforcement learning agents to play Minesweeper.
This project includes a Minesweeper environment (Gymnasium-style) and a Deep Q-Network (DQN) agent with a CNN backbone.
The original project environment can be found here: https://github.com/markov-labs/RL-Minesweeper


## 📐 Project Structure

```
minesweeper-rl/
├── backend/                 # Core game logic, board generation, game state
│   ├── __init__.py
│   ├── board.py             # MinesweeperBoard class: logic, reveal, flag, etc.
│   ├── game.py              # GameSession class: player actions, win/loss, resets
│   └── utils.py             # Helper functions (e.g., random board gen, display)

├── frontend/                # Local web interface (Flask + JS or full SPA)
│   ├── static/              # JS, CSS, images
│   ├── templates/           # HTML templates
│   ├── app.py               # Flask app (serves game + API for model interaction)
│   └── api.py               # Defines REST endpoints (e.g., /new_game, /step, /state)

├── models/                             # Folder for RL agents
│   ├── base_agent.py                   # BaseAgent class (standard API: act(), observe(), train())
│   ├── dqn_agent/                      # DQN + CNN Agent
│   │   ├── dqn_agent.py                # DQNAgent(BaseAgent)
│   │   └── config.yaml                 # Custom Configuration
│   └── registry.py                     # Auto-discovery / loading of available models

├── evaluation/              # Code for running and comparing models
│   ├── evaluate.py
│   ├── leaderboard.json     # Optional: shared results
│   └── visualizer.py        # For replay rendering, statistics, heatmaps

├── notebooks/               # Optional: for experimentation, debugging, analysis

├── config/                  # Game or training configs (YAML or JSON)
│   ├── game_config.yaml
│   └── training_config.yaml

├── tests/                   # Unit tests for backend, models
│   └── test_board.py

├── README.md
├── requirements.txt
└── setup.py                 # Optional: make it pip-installable as a package
```

## Installation

```bash
python -m venv .venv
# Windows: 
source venv/scripts/activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt

```

## Quick Start
To train the DQNAgent, you can run:

```bash
python -m models.dqn_agent.dqn_agent --episodes 100
```

Similarly, to train the DQN_CNN_Agent, you can run:
```bash
python -m models.dqn_cnn_agent.dqn_cnn_agent --episodes 100
```

Common outputs:
- A saved model checkpoint (.pth)
- A CSV of training states (loss, reward, epsilon, steps)

## Evaluate a Saved Model
To evaluate a saved model, you can run:
```bash
python -m evaluation.evaluate

```
