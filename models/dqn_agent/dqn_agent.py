import random
import time
from typing import Dict, Tuple, Any

from flask import Blueprint
from models.base_agent import BaseAgent

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import yaml
import os

import csv

class DQN(nn.Module):
    # Basic DQN network
    # def __init__(self, input_dim: int, output_dim: int, fc1_nodes: int=256):

    class ConvModule(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
            super(DQN.ConvModule, self).__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
            self.bn = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.conv(x)
            x = self.bn(x)
            x = self.relu(x)
            return x

    def __init__(self, input_channels=10, board_size=8, output_dim=64):
        
        super().__init__()
        self.conv1 = self.ConvModule(in_channels=input_channels, out_channels=64, kernel_size=3, padding=1)
        self.convs = nn.Sequential(*[self.ConvModule(in_channels=64, out_channels=64, kernel_size=3, padding=1) for _ in range(3)])
        self.out_conv = nn.Conv2d(in_channels=64, out_channels=output_dim, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1,1))

        # Fully connected layers
        fc_input_dim = output_dim
        hidden_dim = 256

        self.fc1 = nn.Linear(fc_input_dim, hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.final_fc = nn.Linear(hidden_dim, output_dim)

        print(f"Initialized CNN DQN with input shape ({input_channels}, {board_size}, {board_size}) → output_dim={output_dim}")

    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(x)

        for layer in self.convs:
            residual = x
            x = layer(x)

            x = x + residual

        x = self.out_conv(x)
 
        x = self.pool(x)

        x = x.view(x.size(0), -1)


        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.final_fc(x)

        return x

    
class ReplayMemory:
    # Replay buffer
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, transition):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)


class DQNAgent(BaseAgent):
    def __init__(self, config_name="minesweeper_1", config_path=os.path.join(os.path.dirname(__file__), "config.yaml")):
        
        super().__init__()
        with open(config_path, "r") as f:
            all_configs = yaml.safe_load(f)

            print("Available config_name values:", list(all_configs.keys()))

            cfg = all_configs[config_name]

        # Store config
        self.config = cfg
        env_cfg = cfg["env_make_params"]["board"]

        self.width = env_cfg["width"]
        self.height = env_cfg["height"]
        self.num_mines = env_cfg["num_mines"]

        # Hyperparameters
        self.gamma = cfg["discount_factor_g"]
        self.epsilon = cfg["epsilon_init"]
        self.epsilon_decay = cfg["epsilon_decay"]
        self.epsilon_min = cfg["epsilon_min"]
        self.batch_size = cfg["mini_batch_size"]
        self.lr = cfg["learning_rate_a"]
        self.sync_rate = cfg["network_sync_rate"]
        self.fc1_nodes = cfg["fc1_nodes"]
        self.tau = cfg.get("tau", 0.005)  # for soft update

        # If flagging is enabled
        self.enable_flagging = cfg.get("enable_flagging", True)

        # Dimensions
        self.input_dim = self.width * self.height
        if self.enable_flagging:
            self.output_dim = self.input_dim * 2  # reveal/flag for each cell
        else:
            self.output_dim = self.input_dim  # only reveal actions

        # Networks
        # self.policy_net = DQN(self.input_dim, self.output_dim, self.fc1_nodes)
        # self.target_net = DQN(self.input_dim, self.output_dim, self.fc1_nodes)

        # CNN DQN: using one-hot channels and 8×8 board
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        self.policy_net = DQN(input_channels=13, board_size=self.width, output_dim=self.output_dim).to(device)
        self.target_net = DQN(input_channels=13, board_size=self.width, output_dim=self.output_dim).to(device)

        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, weight_decay=1e-4)
        self.memory = ReplayMemory(cfg["replay_memory_size"])

        self.steps_done = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net.to(self.device)
        self.target_net.to(self.device)

        # Load trained model
        model_path = os.path.join(os.path.dirname(__file__), "..", "saved_models", "dqn_model.pth")
        model_path = os.path.abspath(model_path)

        if os.path.exists(model_path):
            self.load(model_path)
            print(f"✅ Loaded pretrained model from {model_path}")
            self.model_loaded = True
        else:
            print(f"⚠️ No pretrained model found at {model_path}. Starting fresh.")
            self.model_loaded = False


    def encode_board(self, board: np.ndarray) -> torch.Tensor:
        """
        Encode the integer board from the environment into a (C, H, W) one-hot tensor.

        Board values (from env):
            -3 : unrevealed
            -2 : flagged (agent flag)
            -4 : false flag (revealed at game over)
            -1 : mine / blast / M (revealed at game over)
            0 : revealed empty
        1–8 : revealed numbers

        Channels:
            0  : hidden
            1  : revealed 0
            2–9: revealed 1–8
            10 : flagged
            11 : mine
            12 : false flag
        """
        board = np.asarray(board)
        H, W = board.shape

        num_channels = 13
        tensor = torch.zeros((num_channels, H, W), dtype=torch.float32)

        for r in range(H):
            for c in range(W):
                v = board[r, c]

                if v == -3:            # hidden
                    tensor[0, r, c] = 1.0
                elif v == 0:           # revealed 0
                    tensor[1, r, c] = 1.0
                elif 1 <= v <= 8:      # revealed numbers
                    tensor[1 + v, r, c] = 1.0  # 1→2, 2→3, ..., 8→9
                elif v == -2:          # flagged
                    tensor[10, r, c] = 1.0
                elif v == -1:          # mine / blast
                    tensor[11, r, c] = 1.0
                elif v == -4:          # false flag (only after game over)
                    tensor[12, r, c] = 1.0
                else:
                    # Shouldn't happen, but keep it from silently breaking
                    tensor[0, r, c] = 1.0  # treat as hidden

        return tensor


    def get_action_mask(self, board):
        H, W = board.shape
        mask = np.zeros(self.output_dim, dtype=bool)

        for r in range(H):
            for c in range(W):
                cell = board[r][c]
                base_idx = (r * W + c) * (2 if self.enable_flagging else 1)

                # legal only if hidden or flagged (flagging allowed)
                hidden = (cell == -3 or cell is None)
                flagged = (cell == -2 or cell == "F")

                # REVEAL mask
                if hidden:
                    mask[base_idx + 0] = True   # reveal allowed

                # FLAG mask
                if self.enable_flagging:
                    # flag allowed only on hidden cells
                    if hidden:
                        mask[base_idx + 1] = True

        # fallback
        if not mask.any():
            mask[:] = True

        return mask



    # This one definitely works
    def act(self, observation: Dict) -> Tuple[str, int, int]:
        """
        Decide on an action based on the current game observation.

        Returns:
            A tuple: (action_type, row, col), where
            - action_type: "reveal" or "flag"
            - row, col: coordinates of the selected cell
        """

        # Act the same as random agent 
        if random.random() < self.epsilon:
            board = np.array(observation["board"])

            height, width = board.shape[:2]

            candidates = np.argwhere((board == -3) | (board == None)).tolist()

            if not candidates:
                return (0, 0, "reveal")  # fallback

            row, col = random.choice(candidates)

            if self.enable_flagging:
                action = random.choice([0, 1])
            else:
                action = 0  # only reveal

            return (row, col, action)
        # Otherwise, act on policy
        else:

            board = np.array(observation["board"])
            state_tensor = self.encode_board(board).unsqueeze(0).to(self.device)  # shape (1,10,8,8)


            with torch.no_grad():
                q_values = self.policy_net(state_tensor)

            board_np = np.array(observation["board"])
            mask = self.get_action_mask(board_np)

            q = q_values.squeeze().cpu().numpy()

            # Mask illegal actions to -inf
            q_masked = np.where(mask, q, -1e9)

            action_idx = int(np.argmax(q_masked))

            if self.enable_flagging:
                cell_idx = action_idx // 2
                row, col = divmod(cell_idx, self.width)
                action_type = 0 if action_idx % 2 == 0 else 1  # 0=reveal, 1=flag
            else:
                row, col = divmod(action_idx, self.width)
                action_type = 0  # reveal only

            return (row, col, action_type)
        


    def observe(self, transition: Dict[str, Any]):
        """
        Optional: record experience from environment (e.g., for replay buffer).

        transition example:
        {
            "state": ...,         # current observation
            "action": ("reveal", 3, 4),
            "reward": -1,
            "next_state": ...,
            "done": True
        }
        """
        self.memory.push(transition)

    def train(self):
        """
        Optional: run one training step (e.g., from replay buffer).
        """
        if len(self.memory) < self.batch_size:
            return
        
        batch = self.memory.sample(self.batch_size)
        states = torch.stack([self.encode_board(np.array(t["state"]["board"])) for t in batch]).to(self.device)
        next_states = torch.stack([self.encode_board(np.array(t["next_state"]["board"])) for t in batch]).to(self.device)
        rewards = torch.FloatTensor([t["reward"] for t in batch]).to(self.device)
        dones = torch.FloatTensor([float(t["done"]) for t in batch]).to(self.device)

        # compute action indices
        action_indices = []
        for t in batch:
            # print("action in batch:", t["action"])
            r, c, action_type = t["action"]
            if self.enable_flagging:
                # idx = ((r * self.width) + c) * 2 + (0 if action_type == "reveal" else 1)
                 idx = ((r * self.width) + c) * 2 + int(action_type)
            else:
                idx = (r * self.width) + c
            action_indices.append(idx)
        action_indices = torch.LongTensor(action_indices).unsqueeze(1).to(self.device)
        # print(action_indices)

        # Compute Q values
        model_outputs = self.policy_net(states)
        q_values = self.policy_net(states).gather(1, action_indices)
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        next_q_values = next_q_values.clamp(-1, 1)
        target = rewards + self.gamma * next_q_values * (1 - dones)

        # loss = nn.functional.mse_loss(q_values.squeeze(), target)
        loss = nn.functional.smooth_l1_loss(q_values.squeeze(), target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()

        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.steps_done += 1

        for param, target_param in zip(self.policy_net.parameters(), self.target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        return loss.item()
    
    def prefill_replay_buffer(self, env, num_steps: int):
            """
            Optional: pre-fill the replay buffer with random experiences.
            """
            obs, info = env.reset()
            state = obs
            done = False

            for _ in range(num_steps):
                if done:
                    obs, info = env.reset()
                    state = obs
                    done = False

                action = env.action_space.sample()
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                reward = np.clip(reward, -1, 1)


                transition = {
                    "state": state,
                    "action": action,
                    "reward": reward,
                    "next_state": next_obs,
                    "done": done,
                }
                self.observe(transition)
                state = next_obs


    def train_for_episodes(self, env, num_episodes: int, save_path="saved_models/dqn_model.pth"):
        """
        Train the DQN agent in the given Minesweeper environment for a set number of episodes.
        Automatically continues episode numbering if training_data.csv already exists.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # CSV path inside models/dqn_agent/
        csv_path = os.path.join(os.path.dirname(__file__), "training_data.csv")
        print(f"Logging training data to: {csv_path}")

        start_episode = 1
        if os.path.exists(csv_path):
            # Read last episode number
            try:
                with open(csv_path, "r") as f:
                    last_line = None
                    for last_line in f:  # iterate to last line efficiently
                        pass
                    if last_line:
                        parts = last_line.strip().split(",")
                        if parts and parts[0].isdigit():
                            start_episode = int(parts[0]) + 1

                            # TODO: Continue epsilon or not
                            # epsilon = float(parts[3])
                            # self.epsilon = epsilon  # continue epsilon decay
            except Exception as e:
                print(f"Warning: could not read last episode from log ({e}). Starting from 1.")
        else:
            # Create CSV header if new
            with open(csv_path, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Episode", "Total Reward", "Average Loss", "Epsilon", "Episode Length", "Time Elapsed"])

        print(f"Starting training from episode {start_episode}...")

        start_time = time.perf_counter()
        prefill_replay_buffer_steps = max(1000, self.batch_size)
        self.prefill_replay_buffer(env, prefill_replay_buffer_steps)
        print("Prefilled replay buffer with random experiences.")
        print("Memory length:", len(self.memory))

        log_buffer = []
        flush_interval = 10  # Flush to CSV every 10 episodes

        try:
            for episode in range(start_episode, start_episode + num_episodes):
                obs, info = env.reset()
                state = obs
                done = False
                total_reward = 0
                losses = []
                ep_length = 0

                while not done:
                    # Choose an action
                    action = self.act(state)
                    ep_length += 1

                    # Interact with environment
                    next_obs, reward, terminated, truncated, info = env.step(action)

                    reward = np.clip(reward, -1, 1)
                    done = terminated or truncated
                    total_reward += reward

                    transition = {
                        "state": state,
                        "action": action,
                        "reward": reward,
                        "next_state": next_obs,
                        "done": done,
                    }
                    self.observe(transition)
                    loss = self.train()
                    if loss is not None:
                        losses.append(loss)

                    state = next_obs

                avg_loss = np.mean(losses) if losses else 0.0
                elapsed = time.perf_counter() - start_time

                # Add to log buffer
                log_buffer.append([episode, total_reward, avg_loss, self.epsilon, ep_length, elapsed])

                # Periodically flush to disk
                if episode % flush_interval == 0:
                    with open(csv_path, mode="a", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerows(log_buffer)
                    log_buffer = []  # Clear memory

                    print(f"Episode {episode} — Reward: {total_reward:.2f}, Avg Loss: {avg_loss:.4f}, "
                        f"Epsilon: {self.epsilon:.3f}, Steps: {ep_length}, Time: {elapsed:.2f}s")
                if episode % 1000 == 0:
                    # Save model every 1000 episodes
                    torch.save(self.policy_net.state_dict(), save_path)
                    print(f"✅ Model checkpoint saved to {save_path} at episode {episode}")

        finally:
            # Always flush remaining logs
            if log_buffer:
                with open(csv_path, mode="a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerows(log_buffer)
            print(f"Logs saved to {csv_path}")

            # 💾 Save model
            torch.save(self.policy_net.state_dict(), save_path)
            print(f"Model saved to {save_path}")


    def save(self, path: str):
        """
        Optional: save model state to disk.
        """
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path: str):
        """
        Optional: load model state from disk.
        """
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())


if __name__ == "__main__":
    import argparse
    from environment.minesweeper_env import MinesweeperEnv  # adjust path if needed

    parser = argparse.ArgumentParser(description="Train DQN agent on Minesweeper")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to train")
    parser.add_argument("--save_path", type=str, default="models/saved_models/dqn_model.pth", help="Where to save model")
    parser.add_argument("--render", action="store_true", help="Enable rendering (slower)")
    args = parser.parse_args()

    render_mode = "human" if args.render else None

    import os

    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    agent = DQNAgent(config_name="minesweeper_1", config_path=config_path)
    board_cfg = agent.config["env_make_params"]["board"]

    env = MinesweeperEnv(
        board_size=(board_cfg["width"], board_cfg["height"]),
        num_mines=board_cfg["num_mines"],
        render_mode=render_mode
    )


    input("Press Enter to start training...")
    print(f"🚀 Starting training for {args.episodes} episodes...")
    try:
        agent.train_for_episodes(env, num_episodes=args.episodes, save_path=args.save_path)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Exiting gracefully.")
        agent.save(args.save_path)
        print(f"Model saved to {args.save_path}")

    # Make graphs from training_data.csv using your preferred tool using matplotlib
    import os
    import pandas as pd
    import matplotlib.pyplot as plt

    # Path to the CSV file
    csv_path = os.path.join(os.path.dirname(__file__), "training_data.csv")

    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} episodes from {csv_path}")
    print(df.head())

    # Create output directory for plots
    plot_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Plot 1: Total Reward vs Episode
    plt.figure()
    plt.plot(df["Episode"], df["Total Reward"])
    plt.title("Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, "reward_curve.png"))
    plt.close()

    # Plot 2: Average Loss vs Episode
    plt.figure()
    plt.plot(df["Episode"], df["Average Loss"])
    plt.title("Average Loss per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Average Loss")
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, "loss_curve.png"))
    plt.close()

    # Plot 3: Epsilon vs Episode
    plt.figure()
    plt.plot(df["Episode"], df["Epsilon"])
    plt.title("Epsilon Decay over Time")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, "epsilon_curve.png"))
    plt.close()

    # Plot 4: Episode Length vs Episode
    plt.figure()
    plt.plot(df["Episode"], df["Episode Length"])
    plt.title("Episode Length per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Episode Length (steps)")
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, "episode_length_curve.png"))
    plt.close()


    print(f"✅ Saved plots to: {plot_dir}")






