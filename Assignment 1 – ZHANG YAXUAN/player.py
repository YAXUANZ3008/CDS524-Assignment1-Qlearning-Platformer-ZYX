"""玩家角色类定义：基于GameObject实现输入、物理、碰撞与状态管理。"""

from __future__ import annotations

from typing import Dict

import pygame

from config import (
    BLUE,
    GRAVITY,
    JUMP_FORCE,
    MOVE_SPEED,
    PLAYER_SIZE,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
)
from level_objects import Level, Platform, Coin, Spike, GameObject


class Player(GameObject):
    """玩家角色类。

    该类继承`GameObject`，使用全局固定尺寸和颜色创建玩家实体，
    并封装了玩家在平台跳跃游戏中的核心行为：
    - 键盘输入处理
    - 重力与速度更新
    - 平台碰撞修正（水平/垂直分离处理）
    - 陷阱、金币、终点检测
    - 重置逻辑
    """

    def __init__(self, x: int, y: int) -> None:
        """初始化玩家对象。

        Args:
            x: 玩家初始x坐标。
            y: 玩家初始y坐标。
        """
        # 使用全局PLAYER_SIZE创建玩家碰撞盒，颜色固定为BLUE。
        super().__init__(pygame.Rect(x, y, PLAYER_SIZE, PLAYER_SIZE), BLUE)

        # 初始化水平速度，默认静止。
        self.vel_x = 0.0
        # 初始化垂直速度，默认静止。
        self.vel_y = 0.0
        # 初始化地面状态，创建时默认离地，后续由碰撞逻辑修正。
        self.on_ground = False
        # 初始化生存状态，玩家默认存活。
        self.is_alive = True
        # 初始化分数，默认0分。
        self.score = 0
        # 初始化终点状态，默认未到达。
        self.reached_goal = False

    def handle_input(self) -> Dict[str, bool]:
        """处理玩家键盘输入。

        按键映射：
        - A：左移
        - D：右移
        - 空格：跳跃（仅在on_ground为True时触发）
        - R：重启标记
        - ESC：退出标记

        Returns:
            一个包含控制标记的字典：
            - restart: 是否请求重启
            - quit: 是否请求退出
        """
        # 初始化控制标记，默认都为False。
        command = {"restart": False, "quit": False}

        # 每帧先清空水平速度，确保按键松开后能停止。
        self.vel_x = 0.0

        # 读取当前按键状态（持续按压有效）。
        keys = pygame.key.get_pressed()

        # 按A时向左移动，速度使用全局MOVE_SPEED。
        if keys[pygame.K_a]:
            self.vel_x = -MOVE_SPEED

        # 按D时向右移动，速度使用全局MOVE_SPEED。
        if keys[pygame.K_d]:
            self.vel_x = MOVE_SPEED

        # 按空格且在地面时触发跳跃，避免无限跳。
        if keys[pygame.K_SPACE] and self.on_ground:
            # 跳跃速度使用全局JUMP_FORCE（向上为负）。
            self.vel_y = JUMP_FORCE
            # 起跳后立即标记离地，防止同帧重复触发。
            self.on_ground = False

        # 按R标记为请求重启（由外部主循环执行重置）。
        if keys[pygame.K_r]:
            command["restart"] = True

        # 按ESC标记为请求退出（由外部主循环结束游戏）。
        if keys[pygame.K_ESCAPE]:
            command["quit"] = True

        # 返回控制标记给调用方。
        return command

    def update(self, level: Level) -> None:
        """更新角色状态（每帧执行一次）。

        更新顺序严格遵循：
        1. 应用重力更新垂直速度
        2. 更新水平位置并进行水平碰撞修正
        3. 更新垂直位置并进行垂直碰撞修正
        4. 边界检测（上下越界死亡）
        5. 陷阱检测（触碰即死亡）
        6. 金币检测（触碰后移除，+30分）
        7. 终点检测（触碰即到达）

        Args:
            level: 当前关卡对象，提供平台/陷阱/金币/终点数据。
        """
        # 若玩家已死亡或已到达终点，则不再更新物理与碰撞状态。
        if (not self.is_alive) or self.reached_goal:
            return

        # 先施加重力到垂直速度，使用全局GRAVITY。
        self.vel_y += GRAVITY

        # 更新水平位置（先移动，再做水平方向碰撞修正）。
        self.rect.x += int(self.vel_x)
        self._check_horizontal_collision(level)

        # 更新垂直位置（先移动，再做垂直方向碰撞修正）。
        self.rect.y += int(self.vel_y)
        self._check_vertical_collision(level)

        # 做左右边界夹取，避免角色水平跑出窗口导致不可见。
        if self.rect.left < 0:
            self.rect.left = 0
        if self.rect.right > SCREEN_WIDTH:
            self.rect.right = SCREEN_WIDTH

        # 边界检测：若角色掉出窗口上下边界，直接判定死亡。
        if self.rect.top > SCREEN_HEIGHT or self.rect.bottom < 0:
            self.is_alive = False
            return

        # 陷阱碰撞检测：任一陷阱碰撞都立即死亡。
        for spike in level.spikes:
            # 使用pygame.Rect原生碰撞检测。
            if self.rect.colliderect(spike.rect):
                self.is_alive = False
                return

        # 金币收集检测：碰撞金币即移除并加分。
        collected_indices = []
        for index, coin in enumerate(level.coins):
            # 使用pygame.Rect原生碰撞检测。
            if self.rect.colliderect(coin.rect):
                # 记录被收集金币索引，循环后统一移除避免边遍历边删除。
                collected_indices.append(index)

        # 按逆序删除金币，保证索引有效。
        for index in reversed(collected_indices):
            level.coins.pop(index)
            # 每个金币固定加30分。
            self.score += 30

        # 终点检测：触碰终点即标记通关成功。
        if self.rect.colliderect(level.goal.rect):
            self.reached_goal = True

    def reset(self, x: int, y: int) -> None:
        """重置角色状态到初始值。

        Args:
            x: 重置后的起始x坐标。
            y: 重置后的起始y坐标。
        """
        # 重置位置到指定起点。
        self.rect.x = x
        self.rect.y = y

        # 重置速度为静止状态。
        self.vel_x = 0.0
        self.vel_y = 0.0

        # 重置地面标记，初始按在地面处理更符合平台起点语义。
        self.on_ground = True

        # 重置生存状态为存活。
        self.is_alive = True

        # 重置分数。
        self.score = 0

        # 重置终点状态为未到达。
        self.reached_goal = False

    def _check_horizontal_collision(self, level: Level) -> None:
        """执行水平方向平台碰撞检测与修正。

        规则：
        - 向右运动撞到平台：角色右边贴平台左边，水平速度清零。
        - 向左运动撞到平台：角色左边贴平台右边，水平速度清零。

        Args:
            level: 当前关卡对象。
        """
        # 遍历所有平台进行碰撞检测。
        for platform in level.platforms:
            # 仅在发生重叠时执行位置修正。
            if self.rect.colliderect(platform.rect):
                # 若当前向右移动，说明是右侧撞墙。
                if self.vel_x > 0:
                    self.rect.right = platform.rect.left
                    self.vel_x = 0.0
                # 若当前向左移动，说明是左侧撞墙。
                elif self.vel_x < 0:
                    self.rect.left = platform.rect.right
                    self.vel_x = 0.0

    def _check_vertical_collision(self, level: Level) -> None:
        """执行垂直方向平台碰撞检测与修正。

        规则：
        - 下落接触平台顶面：角色落地、速度清零、on_ground=True。
        - 上升顶到平台底面：角色头部贴平台底、速度清零。

        Args:
            level: 当前关卡对象。
        """
        # 在垂直碰撞检测前先默认离地，若落地再置True。
        self.on_ground = False

        # 遍历所有平台处理垂直碰撞。
        for platform in level.platforms:
            # 使用pygame.Rect原生重叠判断。
            if self.rect.colliderect(platform.rect):
                # 垂直速度大于0表示正在下落，处理落地。
                if self.vel_y > 0:
                    # 让角色底边贴到平台顶边，防止穿模。
                    self.rect.bottom = platform.rect.top
                    # 落地后清零竖直速度。
                    self.vel_y = 0.0
                    # 落地后允许再次跳跃。
                    self.on_ground = True
                # 垂直速度小于0表示正在上升，处理顶头碰撞。
                elif self.vel_y < 0:
                    # 让角色顶边贴到平台底边，防止穿透。
                    self.rect.top = platform.rect.bottom
                    # 上升被阻挡后清零竖直速度。
                    self.vel_y = 0.0
