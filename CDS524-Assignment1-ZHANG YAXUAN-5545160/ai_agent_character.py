"""AI角色类定义：基于Player扩展离散动作执行接口。"""

from __future__ import annotations

from config import JUMP_FORCE, MOVE_SPEED, PURPLE
from player import Player


class AIAgentCharacter(Player):
    """AI控制角色类。

    该类继承Player，复用其物理更新、碰撞检测与重置逻辑，
    仅补充DQN离散动作到角色速度/跳跃行为的映射方法。
    """

    def __init__(self, x: int, y: int) -> None:
        """初始化AI角色。

        Args:
            x: AI角色初始x坐标。
            y: AI角色初始y坐标。
        """
        # 调用父类构造函数初始化碰撞盒与基础状态。
        super().__init__(x, y)
        # 将角色颜色改为PURPLE，用于与玩家角色区分。
        self.color = PURPLE

    def execute_action(self, action: int) -> None:
        """执行DQN输出动作。

        动作空间严格定义如下：
        - action=0：左移（vel_x = -MOVE_SPEED）
        - action=1：右移（vel_x = MOVE_SPEED）
        - action=2：原地跳跃（仅on_ground为True时触发；vel_x保持0）
        - action=3：左跳（vel_x = -MOVE_SPEED，且仅on_ground为True时触发跳跃）
        - action=4：右跳（vel_x = MOVE_SPEED，且仅on_ground为True时触发跳跃）

        Args:
            action: DQN输出的离散动作编号。
        """
        # 每帧先重置水平速度，确保动作之间互斥且行为可控。
        self.vel_x = 0.0

        # 动作0：向左水平移动。
        if action == 0:
            self.vel_x = -MOVE_SPEED

        # 动作1：向右水平移动。
        elif action == 1:
            self.vel_x = MOVE_SPEED

        # 动作2：原地垂直跳跃，仅允许在地面起跳。
        elif action == 2:
            if self.on_ground:
                self.vel_y = JUMP_FORCE
                self.on_ground = False

        # 动作3：向左跳跃，包含水平位移和条件跳跃。
        elif action == 3:
            self.vel_x = -MOVE_SPEED
            if self.on_ground:
                self.vel_y = JUMP_FORCE
                self.on_ground = False

        # 动作4：向右跳跃，包含水平位移和条件跳跃。
        elif action == 4:
            self.vel_x = MOVE_SPEED
            if self.on_ground:
                self.vel_y = JUMP_FORCE
                self.on_ground = False

        # 对非法动作不执行额外行为，保持当前速度重置后的安全状态。
