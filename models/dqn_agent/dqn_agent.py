import random
import time
from typing import Dict, Tuple, Any, Optional

from models.base_agent import BaseAgent

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import yaml
import os
import csv


# -------------------------
# Model
# -------------------------
class DQN(nn.Module):
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
        self.convs = nn.Sequential(
            *[self.ConvModule(in_channels=64, out_channels=64, kernel_size=3, padding=1) for _ in range(3)]
        )
        self.out_conv = nn.Conv2d(in_channels=64, out_channels=output_dim, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

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


# -------------------------
# Replay Buffer
# -------------------------
class ReplayMemory:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
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

    def state_dict(self) -> Dict[str, Any]:
        # Note: This uses Python pickling under torch.save; make sure transitions are pickleable.
        return {
            "capacity": self.capacity,
            "memory": self.memory,
            "position": self.position,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.capacity = int(state["capacity"])
        self.memory = state["memory"]
        self.position = int(state["position"]) % max(1, self.capacity)


# -------------------------
# Agent
# -------------------------
class DQNAgent(BaseAgent):
    def __init__(
        self,
        config_name: str = "minesweeper_1",
        config_path: str = os.path.join(os.path.dirname(__file__), "config.yaml"),
        checkpoint_path: Optional[str] = None,
        auto_resume: bool = True,
    ):
        super().__init__()

        with open(config_path, "r") as f:
            all_configs = yaml.safe_load(f)
            print("Available config_name values:", list(all_configs.keys()))
            cfg = all_configs[config_name]

        self.config = cfg
        env_cfg = cfg["env_make_params"]["board"]

        self.width = env_cfg["width"]
        self.height = env_cfg["height"]
        self.num_mines = env_cfg["num_mines"]

        # Hyperparameters
        self.gamma = cfg["discount_factor_g"]
        self.epsilon = float(cfg["epsilon_init"])
        self.epsilon_decay = float(cfg["epsilon_decay"])
        self.epsilon_min = float(cfg["epsilon_min"])
        self.batch_size = int(cfg["mini_batch_size"])
        self.lr = float(cfg["learning_rate_a"])
        self.sync_rate = int(cfg["network_sync_rate"])
        self.fc1_nodes = int(cfg["fc1_nodes"])
        self.tau = float(cfg.get("tau", 0.005))  # soft update

        # If flagging is enabled
        self.enable_flagging = bool(cfg.get("enable_flagging", True))

        # Dimensions
        self.input_dim = self.width * self.height
        if self.enable_flagging:
            self.output_dim = self.input_dim * 2
        else:
            self.output_dim = self.input_dim

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Networks
        self.policy_net = DQN(input_channels=13, board_size=self.width, output_dim=self.output_dim).to(self.device)
        self.target_net = DQN(input_channels=13, board_size=self.width, output_dim=self.output_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Optimizer + Replay
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, weight_decay=1e-4)
        self.memory = ReplayMemory(cfg["replay_memory_size"])

        self.steps_done = 0

        # Checkpoint path
        # Prefer explicit checkpoint_path; otherwise default to "models/saved_models/dqn_model.pth" like your CLI default.
        if checkpoint_path is None:
            checkpoint_path = os.path.join(os.path.dirname(__file__), "..", "saved_models", "dqn_checkpoint.pth")
            checkpoint_path = os.path.abspath(checkpoint_path)
        self.checkpoint_path = checkpoint_path

        # Auto-resume if checkpoint exists
        self.model_loaded = False
        if auto_resume and self.checkpoint_path and os.path.exists(self.checkpoint_path):
            self.load_checkpoint(self.checkpoint_path)
            print(f"✅ Resumed checkpoint from {self.checkpoint_path}")
            self.model_loaded = True
        else:
            print(f"⚠️ No checkpoint found at {self.checkpoint_path}. Starting fresh.")
            self.model_loaded = False

    # -------------------------
    # Board Encoding / Mask
    # -------------------------
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
                    tensor[1 + v, r, c] = 1.0
                elif v == -2:          # flagged
                    tensor[10, r, c] = 1.0
                elif v == -1:          # mine / blast
                    tensor[11, r, c] = 1.0
                elif v == -4:          # false flag (only after game over)
                    tensor[12, r, c] = 1.0
                else:
                    tensor[0, r, c] = 1.0  # treat unknown as hidden

        return tensor

    def get_action_mask(self, board):
        H, W = board.shape
        mask = np.zeros(self.output_dim, dtype=bool)

        for r in range(H):
            for c in range(W):
                cell = board[r][c]
                base_idx = (r * W + c) * (2 if self.enable_flagging else 1)

                hidden = (cell == -3 or cell is None)
                # flagged = (cell == -2 or cell == "F")  # not needed for legality below

                # reveal allowed on hidden
                if hidden:
                    mask[base_idx + 0] = True

                # flag allowed only on hidden
                if self.enable_flagging and hidden:
                    mask[base_idx + 1] = True

        if not mask.any():
            mask[:] = True

        return mask

    # -------------------------
    # Acting / Observing
    # -------------------------
    def act(self, observation: Dict) -> Tuple[int, int, int]:
        """
        Returns: (row, col, action_type)
            action_type: 0=reveal, 1=flag
        """
        # epsilon-greedy
        if random.random() < self.epsilon:
            board = np.array(observation["board"])
            candidates = np.argwhere((board == -3) | (board == None)).tolist()
            if not candidates:
                return (0, 0, 0)  # fallback reveal

            row, col = random.choice(candidates)
            action_type = random.choice([0, 1]) if self.enable_flagging else 0
            return (row, col, int(action_type))

        # greedy
        board = np.array(observation["board"])
        state_tensor = self.encode_board(board).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.policy_net(state_tensor)

        mask = self.get_action_mask(board)
        q = q_values.squeeze().cpu().numpy()
        q_masked = np.where(mask, q, -1e9)

        action_idx = int(np.argmax(q_masked))

        if self.enable_flagging:
            cell_idx = action_idx // 2
            row, col = divmod(cell_idx, self.width)
            action_type = 0 if action_idx % 2 == 0 else 1
        else:
            row, col = divmod(action_idx, self.width)
            action_type = 0

        return (int(row), int(col), int(action_type))

    def observe(self, transition: Dict[str, Any]):
        self.memory.push(transition)

    # -------------------------
    # Training step
    # -------------------------
    def train(self):
        if len(self.memory) < self.batch_size:
            return None

        batch = self.memory.sample(self.batch_size)

        states = torch.stack([self.encode_board(np.array(t["state"]["board"])) for t in batch]).to(self.device)
        next_states = torch.stack([self.encode_board(np.array(t["next_state"]["board"])) for t in batch]).to(self.device)
        rewards = torch.FloatTensor([t["reward"] for t in batch]).to(self.device)
        dones = torch.FloatTensor([float(t["done"]) for t in batch]).to(self.device)

        # action indices
        action_indices = []
        for t in batch:
            r, c, action_type = t["action"]
            if self.enable_flagging:
                idx = ((r * self.width) + c) * 2 + int(action_type)
            else:
                idx = (r * self.width) + c
            action_indices.append(idx)
        action_indices = torch.LongTensor(action_indices).unsqueeze(1).to(self.device)

        # Q(s,a)
        q_values = self.policy_net(states).gather(1, action_indices)

        # target: r + gamma * max_a' Q_target(s',a')
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        next_q_values = next_q_values.clamp(-1, 1)
        target = rewards + self.gamma * next_q_values * (1 - dones)

        loss = F.smooth_l1_loss(q_values.squeeze(), target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()

        # epsilon + steps
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.steps_done += 1

        # soft update target
        for param, target_param in zip(self.policy_net.parameters(), self.target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return float(loss.item())

    # -------------------------
    # Prefill
    # -------------------------
    def prefill_replay_buffer(self, env, num_steps: int):
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

    # -------------------------
    # Checkpointing (NEW)
    # -------------------------
    def save_checkpoint(self, path: str, episode: int):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "episode": int(episode),
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": float(self.epsilon),
                "steps_done": int(self.steps_done),
                "replay_memory": self.memory.state_dict(),
                "config_name": self.config.get("name", None),
            },
            path,
        )

    def load_checkpoint(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device)

        self.policy_net.load_state_dict(ckpt["policy_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])

        self.epsilon = float(ckpt.get("epsilon", self.epsilon))
        self.steps_done = int(ckpt.get("steps_done", 0))

        if "replay_memory" in ckpt and ckpt["replay_memory"] is not None:
            self.memory.load_state_dict(ckpt["replay_memory"])

        last_episode = int(ckpt.get("episode", 0))
        return last_episode

    # Backwards-compatible save/load (kept, but now point to checkpoints)
    def save(self, path: str):
        self.save_checkpoint(path, episode=0)

    def load(self, path: str):
        self.load_checkpoint(path)

    # -------------------------
    # Train loop
    # -------------------------
    def train_for_episodes(self, env, num_episodes: int, save_path: str = "models/saved_models/dqn_checkpoint.pth"):
        """
        Train for num_episodes more episodes.
        - Continues episode numbering using training_data.csv
        - Resumes training state from checkpoint if present (policy/target/optimizer/epsilon/steps/replay)
        - Prefills replay buffer ONLY if empty (fresh run)
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.checkpoint_path = os.path.abspath(save_path)

        # CSV path
        csv_path = os.path.join(os.path.dirname(__file__), "training_data.csv")
        print(f"Logging training data to: {csv_path}")

        # Determine start_episode from CSV
        start_episode = 1
        if os.path.exists(csv_path):
            try:
                with open(csv_path, "r") as f:
                    last_line = None
                    for last_line in f:
                        pass
                    if last_line:
                        parts = last_line.strip().split(",")
                        if parts and parts[0].isdigit():
                            start_episode = int(parts[0]) + 1
            except Exception as e:
                print(f"Warning: could not read last episode from log ({e}). Starting from 1.")
        else:
            with open(csv_path, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Episode", "Total Reward", "Average Loss", "Epsilon", "Episode Length", "Time Elapsed"])

        # If checkpoint exists, resume and ensure episode numbering is consistent with CSV.
        if os.path.exists(self.checkpoint_path):
            last_ckpt_ep = self.load_checkpoint(self.checkpoint_path)
            print(f"✅ Loaded checkpoint episode={last_ckpt_ep}, epsilon={self.epsilon:.4f}, replay_size={len(self.memory)}")

        print(f"Starting training from episode {start_episode}...")

        # Prefill ONLY if memory empty (fresh run)
        if len(self.memory) == 0:
            prefill_replay_buffer_steps = max(1000, self.batch_size)
            self.prefill_replay_buffer(env, prefill_replay_buffer_steps)
            print(f"Prefilled replay buffer with {prefill_replay_buffer_steps} random experiences.")
        print("Memory length:", len(self.memory))

        start_time = time.perf_counter()
        log_buffer = []
        flush_interval = 10

        try:
            for episode in range(start_episode, start_episode + num_episodes):
                obs, info = env.reset()
                state = obs
                done = False
                total_reward = 0.0
                losses = []
                ep_length = 0

                while not done:
                    action = self.act(state)
                    ep_length += 1

                    next_obs, reward, terminated, truncated, info = env.step(action)

                    reward = float(np.clip(reward, -1, 1))
                    done = bool(terminated or truncated)
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

                avg_loss = float(np.mean(losses)) if losses else 0.0
                elapsed = time.perf_counter() - start_time

                log_buffer.append([episode, total_reward, avg_loss, self.epsilon, ep_length, elapsed])

                if episode % flush_interval == 0:
                    with open(csv_path, mode="a", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerows(log_buffer)
                    log_buffer = []

                    print(
                        f"Episode {episode} — Reward: {total_reward:.2f}, Avg Loss: {avg_loss:.4f}, "
                        f"Epsilon: {self.epsilon:.3f}, Steps: {ep_length}, Time: {elapsed:.2f}s"
                    )

                # Save checkpoint more often than 1000 so interrupts don't lose state
                if episode % 100 == 0:
                    self.save_checkpoint(self.checkpoint_path, episode=episode)
                    print(f"Checkpoint saved to {self.checkpoint_path} at episode {episode}")

        finally:
            # Flush logs
            if log_buffer:
                with open(csv_path, mode="a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerows(log_buffer)
            print(f"Logs saved to {csv_path}")

            # Always save checkpoint at end
            # If we’re in finally due to interrupt, episode might be out of scope; use last CSV episode if needed.
            last_episode_to_save = start_episode + num_episodes - 1
            self.save_checkpoint(self.checkpoint_path, episode=last_episode_to_save)
            print(f"Final checkpoint saved to {self.checkpoint_path}")


# -------------------------
# Script entry
# -------------------------
if __name__ == "__main__":
    import argparse
    from environment.minesweeper_env import MinesweeperEnv  # adjust path if needed
    import pandas as pd
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="Train DQN agent on Minesweeper")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to train")
    parser.add_argument(
        "--save_path",
        type=str,
        default="models/saved_models/dqn_checkpoint.pth",
        help="Where to save checkpoint (policy/target/optimizer/epsilon/replay)",
    )
    parser.add_argument("--render", action="store_true", help="Enable rendering (slower)")
    args = parser.parse_args()

    render_mode = "human" if args.render else None

    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")

    agent = DQNAgent(config_name="minesweeper_1", config_path=config_path, checkpoint_path=args.save_path, auto_resume=True)
    board_cfg = agent.config["env_make_params"]["board"]

    env = MinesweeperEnv(
        board_size=(board_cfg["width"], board_cfg["height"]),
        num_mines=board_cfg["num_mines"],
        render_mode=render_mode
    )

    print(f"Starting training for {args.episodes} episodes...")
    try:
        agent.train_for_episodes(env, num_episodes=args.episodes, save_path=args.save_path)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Exiting gracefully.")
        # Save a final checkpoint (episode number is best-effort here)
        agent.save_checkpoint(os.path.abspath(args.save_path), episode=0)
        print(f"Checkpoint saved to {args.save_path}")

    # Make graphs from training_data.csv
    csv_path = os.path.join(os.path.dirname(__file__), "training_data.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} episodes from {csv_path}")
        print(df.head())

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

        print(f"Saved plots to: {plot_dir}")
    else:
        print(f"No training_data.csv found at {csv_path}; skipping plots.")
