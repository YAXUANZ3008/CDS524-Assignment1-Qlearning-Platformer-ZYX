"""
文件作用：定义DQN网络类与Agent强化学习智能体类（训练与推理共用）。
作者：GitHub Copilot (GPT-5.3-Codex)
依赖：torch、numpy、random、collections.deque、config（全局常量）
"""

from __future__ import annotations

import os
import random
from collections import deque
from typing import Deque, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import ACTION_DIM, FPS, MOVE_SPEED, SCREEN_HEIGHT, SCREEN_WIDTH, STATE_DIM


class DQN(nn.Module):
    """DQN深度Q网络（固定4层全连接结构）。"""

    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM) -> None:
        """初始化固定网络层结构。"""
        super().__init__()

        if state_dim != STATE_DIM:
            raise ValueError(f"state_dim must be {STATE_DIM}, got {state_dim}")
        if action_dim != ACTION_DIM:
            raise ValueError(f"action_dim must be {ACTION_DIM}, got {action_dim}")

        # 自动检测设备，优先使用GPU。
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 第1层：30 -> 256。
        self.fc1 = nn.Linear(state_dim, 256)
        # 第1层激活函数：ReLU。
        self.relu1 = nn.ReLU()

        # 第2层：256 -> 128。
        self.fc2 = nn.Linear(256, 128)
        # 第2层激活函数：ReLU。
        self.relu2 = nn.ReLU()

        # 第3层：128 -> 64。
        self.fc3 = nn.Linear(128, 64)
        # 第3层激活函数：ReLU。
        self.relu3 = nn.ReLU()

        # 输出层：64 -> 5（动作Q值，不使用激活函数）。
        self.fc4 = nn.Linear(64, action_dim)

        # 将模型迁移到目标设备。
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播，输出每个动作的Q值。"""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.to(self.device)

        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        q_values = self.fc4(x)
        return q_values


class Agent:
    """DQN强化学习智能体。"""

    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM) -> None:
        """初始化智能体、超参数、网络与回放池。"""
        if state_dim != STATE_DIM:
            raise ValueError(f"state_dim must be {STATE_DIM}, got {state_dim}")
        if action_dim != ACTION_DIM:
            raise ValueError(f"action_dim must be {ACTION_DIM}, got {action_dim}")

        self.state_dim = state_dim
        self.action_dim = action_dim

        # 固定超参数（按作业要求）。
        self.lr = 0.001
        self.gamma = 0.95
        self.epsilon = 0.9
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.batch_size = 32
        self.memory_size = 10000
        self.update_target_step = 10

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 创建在线网络和目标网络。
        self.online_net = DQN(state_dim=self.state_dim, action_dim=self.action_dim).to(self.device)
        self.target_net = DQN(state_dim=self.state_dim, action_dim=self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        # Adam优化器与MSE损失函数。
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

        # 经验回放池。
        self.memory: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(
            maxlen=self.memory_size
        )

        self.step_count = 0
        self._prev_coin_count = 0
        self._last_action_was_jump = False
        self._episode_steps = 0

    def get_state(self, game, ai_character) -> np.ndarray:
        """提取固定30维归一化状态向量。"""
        ai_cx = ai_character.rect.centerx
        ai_cy = ai_character.rect.centery

        platforms = game.level.platforms
        spikes = game.level.spikes
        coins = game.level.coins
        goal_rect = game.level.goal.rect

        # 1) AI基础状态4维。
        ai_x = ai_character.rect.x / SCREEN_WIDTH
        ai_y = ai_character.rect.y / SCREEN_HEIGHT
        ai_vx = (ai_character.vel_x + MOVE_SPEED) / (2.0 * MOVE_SPEED)
        ai_vy = (ai_character.vel_y + 20.0) / 40.0

        # 2) 终点状态2维。
        goal_x = goal_rect.x / SCREEN_WIDTH
        goal_y = goal_rect.y / SCREEN_HEIGHT

        # 3) 最近3个平台12维。
        sorted_platforms = sorted(
            platforms,
            key=lambda p: (p.rect.centerx - ai_cx) ** 2 + (p.rect.centery - ai_cy) ** 2,
        )
        nearest_platforms = sorted_platforms[:3]
        platform_features: List[float] = []
        for platform in nearest_platforms:
            platform_features.append(platform.rect.x / SCREEN_WIDTH)
            platform_features.append(platform.rect.y / SCREEN_HEIGHT)
            platform_features.append(platform.rect.width / SCREEN_WIDTH)
            platform_features.append(platform.rect.height / SCREEN_HEIGHT)
        while len(platform_features) < 12:
            platform_features.append(0.0)

        # 4) 最近2个陷阱8维。
        sorted_spikes = sorted(
            spikes,
            key=lambda s: (s.rect.centerx - ai_cx) ** 2 + (s.rect.centery - ai_cy) ** 2,
        )
        nearest_spikes = sorted_spikes[:2]
        spike_features: List[float] = []
        for spike in nearest_spikes:
            spike_features.append(spike.rect.x / SCREEN_WIDTH)
            spike_features.append(spike.rect.y / SCREEN_HEIGHT)
            spike_features.append(spike.rect.width / SCREEN_WIDTH)
            spike_features.append(spike.rect.height / SCREEN_HEIGHT)
        while len(spike_features) < 8:
            spike_features.append(0.0)

        # 5) 最近1个金币4维。
        coin_features = [0.0, 0.0, 0.0, 0.0]
        if coins:
            nearest_coin = min(
                coins,
                key=lambda c: (c.rect.centerx - ai_cx) ** 2 + (c.rect.centery - ai_cy) ** 2,
            )
            coin_features = [
                nearest_coin.rect.x / SCREEN_WIDTH,
                nearest_coin.rect.y / SCREEN_HEIGHT,
                nearest_coin.rect.width / SCREEN_WIDTH,
                nearest_coin.rect.height / SCREEN_HEIGHT,
            ]

        state = np.array(
            [ai_x, ai_y, ai_vx, ai_vy, goal_x, goal_y]
            + platform_features
            + spike_features
            + coin_features,
            dtype=np.float32,
        )

        if state.shape[0] < self.state_dim:
            state = np.concatenate(
                [state, np.zeros(self.state_dim - state.shape[0], dtype=np.float32)], axis=0
            )
        state = state[: self.state_dim]
        return np.clip(state, 0.0, 1.0)

    def get_reward(self, game, ai_character, old_state: np.ndarray) -> Tuple[float, bool]:
        """按分层规则计算奖励并返回终止标记。"""
        reward = 0.0
        done = False
        self._episode_steps += 1

        old_x_norm = float(old_state[0]) if old_state.shape[0] > 0 else 0.0
        old_y_norm = float(old_state[1]) if old_state.shape[0] > 1 else 0.0

        current_state = self.get_state(game, ai_character)
        current_x_norm = float(current_state[0])
        current_y_norm = float(current_state[1])

        if ai_character.reached_goal:
            reward += 200.0
            done = True

        if not ai_character.is_alive:
            reward -= 100.0
            done = True

        current_coin_count = len(game.level.coins)
        if current_coin_count < self._prev_coin_count:
            reward += 30.0 * float(self._prev_coin_count - current_coin_count)

        if current_x_norm > old_x_norm:
            reward += 0.1

        if ai_character.is_alive:
            reward += 0.05

        if current_x_norm <= old_x_norm:
            reward -= 2.0

        if self._last_action_was_jump and (current_y_norm > old_y_norm + 0.01):
            reward -= 5.0

        if (self._episode_steps >= 60 * FPS) and (not done):
            reward -= 50.0
            done = True

        self._prev_coin_count = current_coin_count

        if done:
            self._episode_steps = 0
            self._last_action_was_jump = False

        return float(reward), done

    def select_action(self, state: np.ndarray) -> int:
        """使用ε-greedy策略选择动作。"""
        if random.random() < self.epsilon:
            action = random.randrange(self.action_dim)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                q_values = self.online_net(state_tensor)
            action = int(torch.argmax(q_values, dim=1).item())

        self._last_action_was_jump = action in (2, 3, 4)
        return action

    def store_memory(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """存储经验五元组到回放池。"""
        self.memory.append((state, action, reward, next_state, done))

    def update(self) -> float | None:
        """采样经验并更新在线网络。"""
        if len(self.memory) <= self.batch_size:
            return None

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        state_tensor = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        action_tensor = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_state_tensor = torch.tensor(
            np.array(next_states), dtype=torch.float32, device=self.device
        )
        done_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        current_q = self.online_net(state_tensor).gather(1, action_tensor)
        with torch.no_grad():
            max_next_q = self.target_net(next_state_tensor).max(dim=1, keepdim=True)[0]
            target_q = reward_tensor + self.gamma * (1.0 - done_tensor) * max_next_q

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.update_target_step == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            if self.epsilon < self.epsilon_min:
                self.epsilon = self.epsilon_min

        return float(loss.item())

    def reset_episode_tracking(self, game) -> None:
        """在新回合开始时重置回合追踪变量。"""
        self._episode_steps = 0
        self._last_action_was_jump = False
        self._prev_coin_count = len(game.level.coins)

    def save(self, model_path: str) -> None:
        """保存模型与训练状态。"""
        model_dir = os.path.dirname(model_path)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
        torch.save(
            {
                "online_net": self.online_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "epsilon": self.epsilon,
                "step_count": self.step_count,
            },
            model_path,
        )

    def load(self, model_path: str) -> None:
        """加载模型与训练状态。"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.online_net.load_state_dict(checkpoint["online_net"])
        if "target_net" in checkpoint:
            self.target_net.load_state_dict(checkpoint["target_net"])
        else:
            self.target_net.load_state_dict(checkpoint["online_net"])
        self.epsilon = float(checkpoint.get("epsilon", self.epsilon_min))
        self.step_count = int(checkpoint.get("step_count", 0))
        self.online_net.eval()
        self.target_net.eval()
