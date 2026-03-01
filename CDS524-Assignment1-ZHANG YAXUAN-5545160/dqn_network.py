"""DQN深度Q网络定义：固定4层全连接结构（PyTorch）。"""

from __future__ import annotations

import torch
import torch.nn as nn

from config import ACTION_DIM, STATE_DIM


class DQN(nn.Module):
    """DQN深度Q网络。

    结构严格固定为：
    - 输入层：state_dim=30
    - 全连接层1：256 + ReLU
    - 全连接层2：128 + ReLU
    - 全连接层3：64 + ReLU
    - 输出层：action_dim=5（线性输出Q值，无激活）

    该类同时提供设备兼容能力：自动检测CUDA是否可用，优先GPU，否则CPU。
    """

    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM) -> None:
        """初始化网络结构与设备。

        Args:
            state_dim: 状态维度，默认且固定为全局STATE_DIM=30。
            action_dim: 动作维度，默认且固定为全局ACTION_DIM=5。
        """
        # 调用父类构造函数，完成nn.Module基础初始化。
        super().__init__()

        # 严格检查输入维度，确保与作业固定状态空间一致。
        if state_dim != STATE_DIM:
            raise ValueError(f"state_dim must be {STATE_DIM}, got {state_dim}")

        # 严格检查输出维度，确保与作业固定动作空间一致。
        if action_dim != ACTION_DIM:
            raise ValueError(f"action_dim must be {ACTION_DIM}, got {action_dim}")

        # 自动检测设备：优先CUDA（GPU），否则使用CPU。
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 第1层全连接：将30维状态映射到256维隐藏特征。
        self.fc1 = nn.Linear(state_dim, 256)
        # 第1层激活函数：ReLU，引入非线性表达能力。
        self.relu1 = nn.ReLU()

        # 第2层全连接：将256维特征映射到128维隐藏特征。
        self.fc2 = nn.Linear(256, 128)
        # 第2层激活函数：ReLU。
        self.relu2 = nn.ReLU()

        # 第3层全连接：将128维特征映射到64维隐藏特征。
        self.fc3 = nn.Linear(128, 64)
        # 第3层激活函数：ReLU。
        self.relu3 = nn.ReLU()

        # 输出层全连接：将64维特征映射到5维动作Q值。
        self.fc4 = nn.Linear(64, action_dim)

        # 将网络整体移动到自动选择的设备上。
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：输入状态张量，输出每个动作的Q值。

        Args:
            x: 形状为(batch_size, 30)或(30,)的状态张量。

        Returns:
            形状为(batch_size, 5)的动作Q值张量。
        """
        # 若输入是一维状态向量，则扩展为批维度为1的二维张量。
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # 将输入张量放到与网络一致的设备上（GPU或CPU）。
        x = x.to(self.device)

        # 通过第1层线性变换提取高维特征。
        x = self.fc1(x)
        # 通过ReLU激活引入非线性。
        x = self.relu1(x)

        # 通过第2层线性变换继续抽象特征。
        x = self.fc2(x)
        # 第2次ReLU激活。
        x = self.relu2(x)

        # 通过第3层线性变换压缩到64维特征。
        x = self.fc3(x)
        # 第3次ReLU激活。
        x = self.relu3(x)

        # 输出层给出5个离散动作的Q值，不加激活函数。
        q_values = self.fc4(x)

        # 返回动作价值向量。
        return q_values
