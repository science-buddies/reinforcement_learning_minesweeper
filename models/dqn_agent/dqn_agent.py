import os
import csv
import time
import yaml
import random
import pickle
from typing import Dict, Any, Tuple, Optional

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.base_agent import BaseAgent


# -------------------------
# Model
# -------------------------
class DQN(nn.Module):
    class ConvModule(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
            super().__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
            self.bn = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.relu(self.bn(self.conv(x)))

    def __init__(self, input_channels=10, board_size=8, output_dim=64):
        super().__init__()
        self.conv1 = self.ConvModule(in_channels=input_channels, out_channels=64, kernel_size=3, padding=1)
        self.convs = nn.Sequential(*[self.ConvModule(64, 64, 3, 1) for _ in range(3)])
        self.out_conv = nn.Conv2d(64, output_dim, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        hidden_dim = 256
        self.fc1 = nn.Linear(output_dim, hidden_dim)
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

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.final_fc(x)


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


# -------------------------
# Agent
# -------------------------
class DQNAgent(BaseAgent):
    def __init__(
        self,
        config_name: str = "minesweeper_1",
        config_path: str = os.path.join(os.path.dirname(__file__), "config.yaml"),
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
        self.tau = float(cfg.get("tau", 0.005))

        # If flagging is enabled
        self.enable_flagging = bool(cfg.get("enable_flagging", True))

        # Dimensions
        self.input_dim = self.width * self.height
        self.output_dim = self.input_dim * 2 if self.enable_flagging else self.input_dim

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

    # -------------------------
    # Helpers: sidecar paths (policy checkpoint format stays the same)
    # -------------------------
    def _sidecar_paths(self, policy_path: str) -> Dict[str, str]:
        base, _ = os.path.splitext(policy_path)
        return {
            "policy": policy_path,                 # unchanged: state_dict only
            "target": base + ".target.pth",        # state_dict
            "optim": base + ".optim.pth",          # optimizer.state_dict()
            "meta": base + ".meta.yaml",           # epsilon, steps_done
            "replay": base + ".replay.pkl",        # replay memory (pickle)
        }

    # -------------------------
    # Save/load (UNCHANGED FORMAT for policy)
    # -------------------------
    def save(self, path: str):
        """Backwards-compatible: saves ONLY policy_net weights (original format)."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path: str):
        """
        Backwards-compatible: loads ONLY policy_net weights.
        Does NOT overwrite target unless no target sidecar exists.
        """
        state = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(state)

        # If there is no target sidecar, fall back to matching policy.
        sidecars = self._sidecar_paths(path)
        if os.path.exists(sidecars["target"]):
            self.target_net.load_state_dict(torch.load(sidecars["target"], map_location=self.device))
        else:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    # -------------------------
    # New: save/load training state via sidecars
    # -------------------------
    def save_training_state(self, policy_path: str):
        """
        Keeps policy checkpoint format unchanged, but saves training state to sidecars:
        - target weights
        - optimizer state
        - epsilon, steps_done
        - replay buffer
        """
        side = self._sidecar_paths(policy_path)
        os.makedirs(os.path.dirname(policy_path), exist_ok=True)

        # 1) policy weights (unchanged format)
        self.save(side["policy"])

        # 2) target weights
        torch.save(self.target_net.state_dict(), side["target"])

        # 3) optimizer state
        torch.save(self.optimizer.state_dict(), side["optim"])

        # 4) meta
        meta = {
            "epsilon": float(self.epsilon),
            "steps_done": int(self.steps_done),
        }
        with open(side["meta"], "w") as f:
            yaml.safe_dump(meta, f)

        # 5) replay buffer
        # NOTE: This pickles transitions. Should be fine if transitions are simple dicts of numpy/ints/bools.
        with open(side["replay"], "wb") as f:
            pickle.dump(
                {
                    "capacity": int(self.memory.capacity),
                    "position": int(self.memory.position),
                    "memory": self.memory.memory,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    def load_training_state(self, policy_path: str) -> bool:
        """
        Loads policy weights (unchanged), then loads any existing sidecars.
        Returns True if policy_path existed and was loaded.
        """
        side = self._sidecar_paths(policy_path)
        if not os.path.exists(side["policy"]):
            return False

        # policy + target fallback logic handled in load()
        self.load(side["policy"])

        # optimizer state (optional)
        if os.path.exists(side["optim"]):
            try:
                self.optimizer.load_state_dict(torch.load(side["optim"], map_location=self.device))
            except Exception as e:
                print(f"Warning: could not load optimizer state ({e}). Continuing with fresh optimizer.")

        # meta (optional)
        if os.path.exists(side["meta"]):
            try:
                with open(side["meta"], "r") as f:
                    meta = yaml.safe_load(f) or {}
                if "epsilon" in meta:
                    self.epsilon = float(meta["epsilon"])
                if "steps_done" in meta:
                    self.steps_done = int(meta["steps_done"])
            except Exception as e:
                print(f"Warning: could not load meta ({e}).")

        # replay (optional)
        if os.path.exists(side["replay"]):
            try:
                with open(side["replay"], "rb") as f:
                    rb = pickle.load(f)
                # restore
                self.memory.capacity = int(rb.get("capacity", self.memory.capacity))
                self.memory.position = int(rb.get("position", 0)) % max(1, self.memory.capacity)
                self.memory.memory = rb.get("memory", [])
            except Exception as e:
                print(f"Warning: could not load replay buffer ({e}). Continuing with empty buffer.")

        return True

    # -------------------------
    # Board Encoding / Mask
    # -------------------------
    def encode_board(self, board: np.ndarray) -> torch.Tensor:
        board = np.asarray(board)
        H, W = board.shape
        num_channels = 13
        tensor = torch.zeros((num_channels, H, W), dtype=torch.float32)

        for r in range(H):
            for c in range(W):
                v = board[r, c]
                if v == -3:
                    tensor[0, r, c] = 1.0
                elif v == 0:
                    tensor[1, r, c] = 1.0
                elif 1 <= v <= 8:
                    tensor[1 + v, r, c] = 1.0
                elif v == -2:
                    tensor[10, r, c] = 1.0
                elif v == -1:
                    tensor[11, r, c] = 1.0
                elif v == -4:
                    tensor[12, r, c] = 1.0
                else:
                    tensor[0, r, c] = 1.0
        return tensor

    def get_action_mask(self, board):
        H, W = board.shape
        mask = np.zeros(self.output_dim, dtype=bool)

        for r in range(H):
            for c in range(W):
                cell = board[r][c]
                base_idx = (r * W + c) * (2 if self.enable_flagging else 1)
                hidden = (cell == -3 or cell is None)

                if hidden:
                    mask[base_idx + 0] = True
                if self.enable_flagging and hidden:
                    mask[base_idx + 1] = True

        if not mask.any():
            mask[:] = True
        return mask

    # -------------------------
    # Acting / Observing
    # -------------------------
    def act(self, observation: Dict) -> Tuple[int, int, int]:
        if random.random() < self.epsilon:
            board = np.array(observation["board"])
            candidates = np.argwhere((board == -3) | (board == None)).tolist()
            if not candidates:
                return (0, 0, 0)
            row, col = random.choice(candidates)
            action_type = random.choice([0, 1]) if self.enable_flagging else 0
            return (int(row), int(col), int(action_type))

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
            action_type = 0 if (action_idx % 2 == 0) else 1
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

        action_indices = []
        for t in batch:
            r, c, action_type = t["action"]
            if self.enable_flagging:
                idx = ((r * self.width) + c) * 2 + int(action_type)
            else:
                idx = (r * self.width) + c
            action_indices.append(idx)
        action_indices = torch.LongTensor(action_indices).unsqueeze(1).to(self.device)

        q_values = self.policy_net(states).gather(1, action_indices)
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        next_q_values = next_q_values.clamp(-1, 1)
        target = rewards + self.gamma * next_q_values * (1 - dones)

        loss = F.smooth_l1_loss(q_values.squeeze(), target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.steps_done += 1

        # soft update
        for p, tp in zip(self.policy_net.parameters(), self.target_net.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

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
            reward = float(np.clip(reward, -1, 1))

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
    # Train loop
    # -------------------------
    def train_for_episodes(self, env, num_episodes: int, save_path: str = "models/saved_models/dqn_model.pth"):
        """
        Trains for num_episodes more episodes.

        Key behavior:
        - Policy checkpoint format stays unchanged (state_dict only).
        - Training state is saved to sidecars and restored on restart.
        - Replay prefill happens ONLY if replay is empty.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_path = os.path.abspath(save_path)

        # Resume if policy checkpoint exists
        resumed = self.load_training_state(save_path)
        if resumed:
            print(
                f"✅ Resumed from {save_path} | epsilon={self.epsilon:.4f} "
                f"| steps_done={self.steps_done} | replay_size={len(self.memory)}"
            )
        else:
            print(f"⚠️ No checkpoint found at {save_path}. Starting fresh.")

        # CSV logging
        csv_path = os.path.join(os.path.dirname(__file__), "training_data.csv")
        print(f"Logging training data to: {csv_path}")

        start_episode = 1
        if os.path.exists(csv_path):
            try:
                with open(csv_path, "r") as f:
                    last = None
                    for last in f:
                        pass
                    if last:
                        parts = last.strip().split(",")
                        if parts and parts[0].isdigit():
                            start_episode = int(parts[0]) + 1
            except Exception as e:
                print(f"Warning: could not parse last episode from CSV ({e}). Starting from 1.")
        else:
            with open(csv_path, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["Episode", "Total Reward", "Average Loss", "Epsilon", "Episode Length", "Time Elapsed"]
                )

        # Prefill ONLY if replay is empty
        if len(self.memory) == 0:
            prefill_steps = max(1000, self.batch_size)
            self.prefill_replay_buffer(env, prefill_steps)
            print(f"Prefilled replay buffer with {prefill_steps} random experiences.")
        print("Memory length:", len(self.memory))

        start_time = time.perf_counter()
        buffer_rows = []
        flush_interval = 10

        try:
            for episode in range(start_episode, start_episode + num_episodes):
                obs, info = env.reset()
                state = obs
                done = False
                total_reward = 0.0
                losses = []
                ep_len = 0

                while not done:
                    action = self.act(state)
                    ep_len += 1

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
                buffer_rows.append([episode, total_reward, avg_loss, self.epsilon, ep_len, elapsed])

                if episode % flush_interval == 0:
                    with open(csv_path, "a", newline="") as f:
                        csv.writer(f).writerows(buffer_rows)
                    buffer_rows = []
                    print(
                        f"Episode {episode} — Reward: {total_reward:.2f}, Avg Loss: {avg_loss:.4f}, "
                        f"Epsilon: {self.epsilon:.3f}, Steps: {ep_len}, Time: {elapsed:.2f}s"
                    )

                # Save every 100 episodes (your interrupt cadence)
                if episode % 100 == 0:
                    self.save_training_state(save_path)
                    print(f"Checkpoint saved (policy unchanged + sidecars) at episode {episode}")

        finally:
            if buffer_rows:
                with open(csv_path, "a", newline="") as f:
                    csv.writer(f).writerows(buffer_rows)
            # Always save state at end/interrupt
            self.save_training_state(save_path)
            print(f"Final checkpoint saved to {save_path} (plus sidecars).")

if __name__ == "__main__":
    import argparse
    from environment.minesweeper_env import MinesweeperEnv  # adjust if your path differs

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--save_path", type=str, default="models/saved_models/dqn_model.pth")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    render_mode = "human" if args.render else None

    agent = DQNAgent(
        config_name="minesweeper_1",
        config_path=os.path.join(os.path.dirname(__file__), "config.yaml"),
    )

    # Build env from config
    board_cfg = agent.config["env_make_params"]["board"]
    env = MinesweeperEnv(
        board_size=(board_cfg["width"], board_cfg["height"]),
        num_mines=board_cfg["num_mines"],
        render_mode=render_mode,
    )

    agent.train_for_episodes(env, num_episodes=args.episodes, save_path=args.save_path)
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
