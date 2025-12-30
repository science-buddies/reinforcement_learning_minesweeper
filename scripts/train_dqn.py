#!/usr/bin/env python
"""Simple training entrypoint for the DQN agent.

Usage examples (PowerShell):
  python .\scripts\train_dqn.py --episodes 500 --save-path models/dqn_agent/checkpoints/dqn_final.pth

This script loads `config/training_config.yaml` by default and uses the
registered Gym environment id `Minesweeper-v0`.
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import os
import yaml
import gymnasium as gym

from models.dqn_agent.dqn_agent import DQNAgent

from gymnasium.envs.registration import register
from environment.minesweeper_env import MinesweeperEnv

register(
    id='Minesweeper-v0',
    entry_point='environment.minesweeper_env:MinesweeperEnv',
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/training_config.yaml", help="path to training config yaml")
    p.add_argument("--episodes", type=int, default=None, help="number of episodes to train (overrides config)")
    p.add_argument("--agent-config", default="minesweeper1", help="agent config name inside models/dqn_agent/config.yaml")
    p.add_argument("--save-path", default="dqn_minesweeper.pth", help="where to save the final model")
    args = p.parse_args()

    # Load training config (optional)
    training_cfg = {}
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            training_cfg = yaml.safe_load(f) or {}

    training_section = training_cfg.get("training", {})
    cfg_episodes = training_section.get("episodes")
    board_size = tuple(training_section.get("board_size", [8, 8]))
    mines = training_section.get("mines", 10)

    episodes = args.episodes if args.episodes is not None else (cfg_episodes or 500)

    print(f"Creating environment Minesweeper-v0 board_size={board_size} mines={mines}")
    env = gym.make("Minesweeper-v0", board_size=board_size, num_mines=mines)

    print(f"Instantiating DQNAgent using config '{args.agent_config}'")
    agent = DQNAgent(config_name=args.agent_config)

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    agent.train_agent(env, num_episodes=episodes, save_path=args.save_path)


if __name__ == "__main__":
    main()
