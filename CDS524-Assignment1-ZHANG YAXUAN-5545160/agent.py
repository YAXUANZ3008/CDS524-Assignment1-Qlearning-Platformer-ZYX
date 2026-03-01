"""DQN强化学习智能体实现：状态提取、奖励计算、动作选择、经验回放与网络更新。"""

from __future__ import annotations

import random
from collections import deque
from typing import Deque, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import ACTION_DIM, FPS, MOVE_SPEED, PLAYER_SIZE, SCREEN_HEIGHT, SCREEN_WIDTH, STATE_DIM
from dqn_network import DQN


class Agent:
    """DQN强化学习智能体。

    核心职责：
    - 从游戏对象提取固定30维状态
    - 按分层奖励规则计算当前帧奖励
    - 采用ε-greedy策略选择动作
    - 通过经验回放更新在线网络
    - 周期性同步目标网络提升训练稳定性
    """

    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM) -> None:
        """初始化智能体核心组件与固定超参数。"""
        # 严格固定状态维度为30，保证与作业定义一致。
        if state_dim != STATE_DIM:
            raise ValueError(f"state_dim must be {STATE_DIM}, got {state_dim}")

        # 严格固定动作维度为5，保证与作业定义一致。
        if action_dim != ACTION_DIM:
            raise ValueError(f"action_dim must be {ACTION_DIM}, got {action_dim}")

        # 记录维度信息，供后续张量构建和断言使用。
        self.state_dim = state_dim
        self.action_dim = action_dim

        # -------------------------------
        # 固定超参数（按作业要求）
        # -------------------------------
        # 学习率固定为0.001。
        self.lr = 0.001
        # 折扣因子固定为0.95。
        self.gamma = 0.95
        # epsilon初始值固定为0.9。
        self.epsilon = 0.9
        # epsilon最小值固定为0.05。
        self.epsilon_min = 0.05
        # epsilon衰减系数固定为0.995。
        self.epsilon_decay = 0.995
        # 训练批次大小固定为32。
        self.batch_size = 32
        # 经验池最大容量固定为10000。
        self.memory_size = 10000
        # 目标网络更新频率固定为每10步更新一次。
        self.update_target_step = 10

        # 自动选择计算设备：优先GPU，否则CPU。
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 创建在线网络，用于动作决策和梯度更新。
        self.online_net = DQN(state_dim=self.state_dim, action_dim=self.action_dim).to(self.device)
        # 创建目标网络，用于计算稳定TD目标。
        self.target_net = DQN(state_dim=self.state_dim, action_dim=self.action_dim).to(self.device)
        # 初始化时将目标网络参数同步为在线网络参数。
        self.target_net.load_state_dict(self.online_net.state_dict())
        # 将目标网络置为评估模式。
        self.target_net.eval()

        # 使用Adam优化器更新在线网络参数。
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=self.lr)
        # 损失函数固定为均方误差MSELoss。
        self.loss_fn = nn.MSELoss()

        # 使用deque实现经验回放池，自动维护最大容量。
        self.memory: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(
            maxlen=self.memory_size
        )

        # 全局步数计数器，用于周期性同步目标网络。
        self.step_count = 0

        # 保存上一帧金币数量，用于检测金币收集奖励。
        self._prev_coin_count = 0
        # 保存上一帧是否执行了跳跃动作，用于跳跃失败惩罚判定。
        self._last_action_was_jump = False
        # 保存当前回合步数，用于60秒超时判定。
        self._episode_steps = 0

    def get_state(self, game, ai_character) -> np.ndarray:
        """获取游戏当前状态并输出固定30维归一化向量。

        状态向量顺序严格为：
        1) AI基础状态4维：x, y, vel_x, vel_y
        2) 终点状态2维：goal_x, goal_y
        3) 最近3个平台12维：每个平台x, y, w, h
        4) 最近2个陷阱8维：每个陷阱x, y, w, h
        5) 最近1个金币4维：x, y, w, h

        若某类元素数量不足，按要求补0，保证总维度恒为30。
        """
        # 读取AI当前矩形中心点，作为最近邻距离计算基准。
        ai_cx = ai_character.rect.centerx
        ai_cy = ai_character.rect.centery

        # 读取关卡元素引用，简化后续代码。
        platforms = game.level.platforms
        spikes = game.level.spikes
        coins = game.level.coins
        goal_rect = game.level.goal.rect

        # -------- 1) AI基础状态（4维） --------
        # x坐标归一化到0-1。
        ai_x = ai_character.rect.x / SCREEN_WIDTH
        # y坐标归一化到0-1。
        ai_y = ai_character.rect.y / SCREEN_HEIGHT
        # 水平速度归一化到0-1（范围按[-MOVE_SPEED, MOVE_SPEED]映射）。
        ai_vx = (ai_character.vel_x + MOVE_SPEED) / (2.0 * MOVE_SPEED)
        # 垂直速度归一化到0-1（范围按[-20, 20]映射）。
        ai_vy = (ai_character.vel_y + 20.0) / 40.0

        # -------- 2) 终点状态（2维） --------
        # 终点x坐标归一化。
        goal_x = goal_rect.x / SCREEN_WIDTH
        # 终点y坐标归一化。
        goal_y = goal_rect.y / SCREEN_HEIGHT

        # -------- 3) 最近3个平台（12维） --------
        # 按与AI中心点的欧氏距离平方从近到远排序平台。
        sorted_platforms = sorted(
            platforms,
            key=lambda p: (p.rect.centerx - ai_cx) ** 2 + (p.rect.centery - ai_cy) ** 2,
        )
        # 取最近3个平台。
        nearest_platforms = sorted_platforms[:3]

        # 平台特征列表，后续逐项扩展为x,y,w,h。
        platform_features: List[float] = []
        for platform in nearest_platforms:
            # 平台x归一化。
            platform_features.append(platform.rect.x / SCREEN_WIDTH)
            # 平台y归一化。
            platform_features.append(platform.rect.y / SCREEN_HEIGHT)
            # 平台宽度归一化。
            platform_features.append(platform.rect.width / SCREEN_WIDTH)
            # 平台高度归一化。
            platform_features.append(platform.rect.height / SCREEN_HEIGHT)

        # 若不足3个平台则补0，确保该段固定12维。
        while len(platform_features) < 12:
            platform_features.append(0.0)

        # -------- 4) 最近2个陷阱（8维） --------
        # 按与AI中心点距离对陷阱排序。
        sorted_spikes = sorted(
            spikes,
            key=lambda s: (s.rect.centerx - ai_cx) ** 2 + (s.rect.centery - ai_cy) ** 2,
        )
        # 取最近2个陷阱。
        nearest_spikes = sorted_spikes[:2]

        # 陷阱特征列表，后续逐项扩展为x,y,w,h。
        spike_features: List[float] = []
        for spike in nearest_spikes:
            # 陷阱x归一化。
            spike_features.append(spike.rect.x / SCREEN_WIDTH)
            # 陷阱y归一化。
            spike_features.append(spike.rect.y / SCREEN_HEIGHT)
            # 陷阱宽度归一化。
            spike_features.append(spike.rect.width / SCREEN_WIDTH)
            # 陷阱高度归一化。
            spike_features.append(spike.rect.height / SCREEN_HEIGHT)

        # 若不足2个陷阱则补0，确保该段固定8维。
        while len(spike_features) < 8:
            spike_features.append(0.0)

        # -------- 5) 最近1个金币（4维） --------
        # 初始化金币特征为全0，表示没有可用金币目标。
        coin_features = [0.0, 0.0, 0.0, 0.0]
        # 若存在金币则选最近一个并提取x,y,w,h。
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

        # 按要求顺序拼接完整状态向量。
        state = np.array(
            [ai_x, ai_y, ai_vx, ai_vy, goal_x, goal_y]
            + platform_features
            + spike_features
            + coin_features,
            dtype=np.float32,
        )

        # 若长度不足30则补0（理论上不会触发，但用于防御性保证）。
        if state.shape[0] < self.state_dim:
            padding = np.zeros(self.state_dim - state.shape[0], dtype=np.float32)
            state = np.concatenate([state, padding], axis=0)

        # 若长度超过30则截断到30维（防御性处理）。
        state = state[: self.state_dim]

        # 裁剪到0-1区间，保证归一化结果稳定。
        state = np.clip(state, 0.0, 1.0)

        # 返回固定30维状态向量。
        return state

    def get_reward(self, game, ai_character, old_state: np.ndarray) -> Tuple[float, bool]:
        """计算当前帧奖励并返回(done)标记。

        奖励分层逻辑严格实现：
        1) 通关奖励：+200，done=True
        2) 死亡惩罚：-100，done=True
        3) 金币奖励：每收集1个+30
        4) 进度奖励：向终点方向移动每帧+0.1
        5) 生存奖励：每存活1帧+0.05
        6) 摆烂惩罚：反向移动或原地不动每帧-2
        7) 跳跃失败惩罚：跳后掉到更低平台每帧-5
        8) 超时惩罚：60秒未通关-50，done=True
        """
        # 初始化本帧奖励值。
        reward = 0.0
        # 默认本帧未终止。
        done = False

        # 每调用一次奖励函数即视为一个时间步。
        self._episode_steps += 1

        # 读取旧状态中的x/y坐标（已归一化），用于比较位移与高度变化。
        old_x_norm = float(old_state[0]) if old_state.shape[0] > 0 else 0.0
        old_y_norm = float(old_state[1]) if old_state.shape[0] > 1 else 0.0

        # 获取当前最新状态，提取当前x/y用于对比。
        current_state = self.get_state(game, ai_character)
        current_x_norm = float(current_state[0])
        current_y_norm = float(current_state[1])

        # ---------------- 1) 通关奖励 ----------------
        # 若到达终点，直接给予高额奖励并终止回合。
        if ai_character.reached_goal:
            reward += 200.0
            done = True

        # ---------------- 2) 死亡惩罚 ----------------
        # 若AI死亡（触碰陷阱或掉出地图），给予惩罚并终止回合。
        if not ai_character.is_alive:
            reward -= 100.0
            done = True

        # ---------------- 3) 金币收集奖励 ----------------
        # 读取当前关卡剩余金币数量。
        current_coin_count = len(game.level.coins)
        # 若当前金币数量小于上一帧，说明发生了收集。
        if current_coin_count < self._prev_coin_count:
            # 按减少的金币数累加奖励，每个+30。
            reward += 30.0 * float(self._prev_coin_count - current_coin_count)

        # ---------------- 4) 进度奖励 ----------------
        # 若x增大，表示朝终点方向前进，给予小额正奖励。
        if current_x_norm > old_x_norm:
            reward += 0.1

        # ---------------- 5) 生存奖励 ----------------
        # 若本帧仍存活，给予生存奖励。
        if ai_character.is_alive:
            reward += 0.05

        # ---------------- 6) 摆烂惩罚 ----------------
        # 若反向移动或原地不动（x未前进），给予惩罚。
        if current_x_norm <= old_x_norm:
            reward -= 2.0

        # ---------------- 7) 跳跃失败惩罚 ----------------
        # 若上一动作是跳跃，且当前位置y更大（更靠下），视为掉到更低平台并惩罚。
        if self._last_action_was_jump and (current_y_norm > old_y_norm + 0.01):
            reward -= 5.0

        # ---------------- 8) 超时惩罚 ----------------
        # 若超过60秒仍未终止，触发超时惩罚并终止回合。
        if (self._episode_steps >= 60 * FPS) and (not done):
            reward -= 50.0
            done = True

        # 更新上一帧金币数量缓存，供下一帧比较。
        self._prev_coin_count = current_coin_count

        # 若回合结束，重置回合计数与跳跃标记，便于下一回合干净开始。
        if done:
            self._episode_steps = 0
            self._last_action_was_jump = False

        # 返回本帧奖励与终止标记。
        return float(reward), done

    def select_action(self, state: np.ndarray) -> int:
        """使用ε-greedy策略选择动作。"""
        # 以epsilon概率执行随机探索动作。
        if random.random() < self.epsilon:
            action = random.randrange(self.action_dim)
        else:
            # 将状态转为张量并增加batch维。
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            # 在无梯度模式下推理Q值。
            with torch.no_grad():
                q_values = self.online_net(state_tensor)
            # 选择Q值最大的动作作为利用策略。
            action = int(torch.argmax(q_values, dim=1).item())

        # 记录该动作是否属于跳跃动作，供奖励函数惩罚逻辑使用。
        self._last_action_was_jump = action in (2, 3, 4)

        # 返回离散动作编号。
        return action

    def store_memory(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """存储经验五元组到经验回放池。"""
        # 将(s, a, r, s', done)按元组格式压入deque。
        self.memory.append((state, action, reward, next_state, done))

    def update(self) -> float | None:
        """采样经验并更新在线网络。

        Returns:
            本次更新损失值；若样本不足不更新则返回None。
        """
        # 仅当经验数量大于batch_size时才进行训练更新。
        if len(self.memory) <= self.batch_size:
            return None

        # 从经验池随机采样一个batch，打破样本时序相关性。
        batch = random.sample(self.memory, self.batch_size)

        # 解包batch为独立列表，便于转换张量。
        states, actions, rewards, next_states, dones = zip(*batch)

        # 构建状态张量，形状为(batch_size, state_dim)。
        state_tensor = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        # 构建动作张量，形状为(batch_size, 1)。
        action_tensor = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        # 构建奖励张量，形状为(batch_size, 1)。
        reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        # 构建下一状态张量，形状为(batch_size, state_dim)。
        next_state_tensor = torch.tensor(
            np.array(next_states), dtype=torch.float32, device=self.device
        )
        # 构建终止标记张量，形状为(batch_size, 1)。
        done_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        # 计算当前Q(s,a)：先前向得到所有动作Q，再按动作索引gather。
        current_q = self.online_net(state_tensor).gather(1, action_tensor)

        # 在无梯度模式下计算目标Q值，避免反向传播到目标网络。
        with torch.no_grad():
            # 目标网络计算next_state下最大动作Q值。
            max_next_q = self.target_net(next_state_tensor).max(dim=1, keepdim=True)[0]
            # 按DQN目标公式构造TD目标。
            target_q = reward_tensor + self.gamma * (1.0 - done_tensor) * max_next_q

        # 计算MSE损失。
        loss = self.loss_fn(current_q, target_q)

        # 反向传播前先清空历史梯度。
        self.optimizer.zero_grad()
        # 反向传播计算参数梯度。
        loss.backward()
        # 执行一步参数更新。
        self.optimizer.step()

        # 训练步计数+1，用于目标网络周期更新。
        self.step_count += 1

        # 按固定频率同步目标网络参数。
        if self.step_count % self.update_target_step == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        # 每次参数更新后衰减epsilon，并限制不低于最小值。
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            if self.epsilon < self.epsilon_min:
                self.epsilon = self.epsilon_min

        # 返回标量损失值，便于训练日志记录。
        return float(loss.item())

    def reset_episode_tracking(self, game) -> None:
        """重置回合级追踪变量。

        在每个新回合开始前调用，确保奖励统计与超时逻辑正确。
        """
        # 重置回合步数计数。
        self._episode_steps = 0
        # 重置跳跃动作追踪标记。
        self._last_action_was_jump = False
        # 将上一帧金币数量同步为当前关卡金币数。
        self._prev_coin_count = len(game.level.coins)
