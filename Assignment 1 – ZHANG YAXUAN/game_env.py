"""平台闯关环境定义：状态编码、动作执行、奖励设计与渲染。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pygame

from config import (
    ACTION_DIM,
    FPS,
    GRAVITY,
    GREEN,
    MAX_STEPS_PER_EPISODE,
    MOVE_SPEED,
    ORANGE,
    PLAYER_SIZE,
    RED,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    WHITE,
    YELLOW,
    JUMP_FORCE,
)


@dataclass
class StepResult:
    """封装单步交互结果。"""

    state: np.ndarray
    reward: float
    done: bool
    info: Dict[str, float]


class PlatformerEnv:
    """2D横版平台跳跃环境（训练与可视化共用）。"""

    def __init__(self) -> None:
        """初始化关卡、地形与运行时变量。"""
        # 固定随机种子以提高实验可复现性。
        self.random_state = np.random.RandomState(42)
        # 构建固定关卡元素（平台、陷阱、金币、终点）。
        self.platforms, self.traps, self.coins_template, self.goal = self._build_level()
        # 初始化运行时状态（玩家、速度、计时等）。
        self.player_rect = pygame.Rect(40, SCREEN_HEIGHT - 40 - PLAYER_SIZE, PLAYER_SIZE, PLAYER_SIZE)
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.on_ground = False
        self.collected = []
        self.step_count = 0
        self.prev_x = float(self.player_rect.x)
        self.done = False

    def _build_level(
        self,
    ) -> Tuple[List[pygame.Rect], List[pygame.Rect], List[pygame.Rect], pygame.Rect]:
        """创建固定关卡布局。"""
        # 地面平台占据底部区域，保证角色不会无限下落。
        ground = pygame.Rect(0, SCREEN_HEIGHT - 40, SCREEN_WIDTH, 40)
        # 设计多个中间平台，形成跳跃路径。
        platform_1 = pygame.Rect(140, 300, 110, 16)
        platform_2 = pygame.Rect(290, 255, 120, 16)
        platform_3 = pygame.Rect(465, 210, 110, 16)
        platform_4 = pygame.Rect(620, 260, 100, 16)
        platforms = [ground, platform_1, platform_2, platform_3, platform_4]

        # 设计若干陷阱区域（红色），触碰即失败。
        trap_1 = pygame.Rect(210, SCREEN_HEIGHT - 50, 45, 10)
        trap_2 = pygame.Rect(420, SCREEN_HEIGHT - 50, 45, 10)
        trap_3 = pygame.Rect(560, SCREEN_HEIGHT - 50, 45, 10)
        traps = [trap_1, trap_2, trap_3]

        # 设计金币点位，鼓励探索与高回报路径。
        coin_1 = pygame.Rect(165, 270, 14, 14)
        coin_2 = pygame.Rect(340, 225, 14, 14)
        coin_3 = pygame.Rect(510, 180, 14, 14)
        coin_4 = pygame.Rect(650, 230, 14, 14)
        coins = [coin_1, coin_2, coin_3, coin_4]

        # 终点旗帜放置在关卡右侧。
        goal = pygame.Rect(748, SCREEN_HEIGHT - 95, 24, 55)
        return platforms, traps, coins, goal

    def reset(self) -> np.ndarray:
        """重置环境并返回初始状态。"""
        # 重置玩家位置到关卡起点。
        self.player_rect.x = 40
        self.player_rect.y = SCREEN_HEIGHT - 40 - PLAYER_SIZE
        # 重置速度分量。
        self.vel_x = 0.0
        self.vel_y = 0.0
        # 地面状态在第一帧默认置真。
        self.on_ground = True
        # 金币收集标记清空。
        self.collected = [False for _ in self.coins_template]
        # 步计数和结束标记复位。
        self.step_count = 0
        self.prev_x = float(self.player_rect.x)
        self.done = False
        # 返回固定30维归一化状态向量。
        return self._get_state()

    def step(self, action: int) -> StepResult:
        """执行动作并返回转移结果。"""
        # 如果当前回合已结束，则直接回传当前状态并不给予额外奖励。
        if self.done:
            return StepResult(self._get_state(), 0.0, True, {"already_done": 1.0})

        # 记录动作前的横向坐标，用于计算进度增量奖励。
        start_x = float(self.player_rect.x)
        # 应用动作到速度分量（含跳跃触发）。
        self._apply_action(action)
        # 执行物理仿真（重力、移动、碰撞）。
        self._physics_step()
        # 步数递增，用于时间惩罚和超时判定。
        self.step_count += 1

        # 初始化奖励分量字典，便于UI实时展示奖励组成。
        reward_parts = {
            "time_penalty": -0.02,
            "progress": 0.0,
            "coin": 0.0,
            "goal": 0.0,
            "trap": 0.0,
            "timeout": 0.0,
            "backtrack": 0.0,
        }

        # 基础时间惩罚，鼓励更快通关。
        reward = reward_parts["time_penalty"]

        # 根据横向前进量计算进度奖励。
        dx = float(self.player_rect.x) - start_x
        reward_parts["progress"] = dx * 0.01
        reward += reward_parts["progress"]

        # 如果出现明显后退，施加额外惩罚以稳定策略方向。
        if dx < 0:
            reward_parts["backtrack"] = -0.02
            reward += reward_parts["backtrack"]

        # 检查金币碰撞并奖励。
        for idx, coin_rect in enumerate(self.coins_template):
            # 仅对未收集金币进行判定。
            if (not self.collected[idx]) and self.player_rect.colliderect(coin_rect):
                # 标记金币已收集，避免重复记分。
                self.collected[idx] = True
                # 单枚金币固定正奖励。
                reward_parts["coin"] += 2.0
                reward += 2.0

        # 检查陷阱碰撞，触发失败终止。
        for trap_rect in self.traps:
            # 触碰陷阱后本局立即结束。
            if self.player_rect.colliderect(trap_rect):
                self.done = True
                reward_parts["trap"] = -5.0
                reward += reward_parts["trap"]
                break

        # 检查终点碰撞，触发通关终止。
        if self.player_rect.colliderect(self.goal):
            # 抵达终点立即结束并给予高额奖励。
            self.done = True
            reward_parts["goal"] = 10.0
            reward += reward_parts["goal"]

        # 若超过回合最大步数则记为超时失败。
        if self.step_count >= MAX_STEPS_PER_EPISODE and (not self.done):
            self.done = True
            reward_parts["timeout"] = -3.0
            reward += reward_parts["timeout"]

        # 缓存当前位置用于外部调试或分析。
        self.prev_x = float(self.player_rect.x)

        # 生成下一个状态并组合信息字典。
        next_state = self._get_state()
        info = {
            "x": float(self.player_rect.x),
            "y": float(self.player_rect.y),
            "progress_ratio": self.player_rect.x / max(float(self.goal.x), 1.0),
            "coins_collected": float(sum(self.collected)),
            "coins_total": float(len(self.collected)),
            "reward_time": reward_parts["time_penalty"],
            "reward_progress": reward_parts["progress"],
            "reward_coin": reward_parts["coin"],
            "reward_goal": reward_parts["goal"],
            "reward_trap": reward_parts["trap"],
            "reward_timeout": reward_parts["timeout"],
            "reward_backtrack": reward_parts["backtrack"],
        }
        return StepResult(next_state, float(reward), self.done, info)

    def _apply_action(self, action: int) -> None:
        """根据离散动作设置水平速度和跳跃触发。"""
        # 每帧先将水平速度重置，再根据动作覆盖。
        self.vel_x = 0.0

        # 动作0：左移。
        if action == 0:
            self.vel_x = -MOVE_SPEED
        # 动作1：右移。
        elif action == 1:
            self.vel_x = MOVE_SPEED
        # 动作2：原地跳跃（仅在可起跳时生效）。
        elif action == 2 and self.on_ground:
            self.vel_y = JUMP_FORCE
            self.on_ground = False
        # 动作3：左跳（左移 + 跳跃）。
        elif action == 3:
            self.vel_x = -MOVE_SPEED
            if self.on_ground:
                self.vel_y = JUMP_FORCE
                self.on_ground = False
        # 动作4：右跳（右移 + 跳跃）。
        elif action == 4:
            self.vel_x = MOVE_SPEED
            if self.on_ground:
                self.vel_y = JUMP_FORCE
                self.on_ground = False

    def _physics_step(self) -> None:
        """执行单帧物理更新与碰撞处理。"""
        # 先更新水平方向位置。
        self.player_rect.x += int(self.vel_x)
        # 将玩家限制在窗口边界内。
        self.player_rect.x = max(0, min(self.player_rect.x, SCREEN_WIDTH - PLAYER_SIZE))

        # 施加重力后更新竖直速度。
        self.vel_y += GRAVITY
        # 为避免数值爆炸，对下落速度做合理截断。
        self.vel_y = min(self.vel_y, 20.0)
        # 更新竖直方向位置。
        self.player_rect.y += int(self.vel_y)

        # 默认离地，后续若检测到支撑则置回True。
        self.on_ground = False

        # 针对每个平台做竖直碰撞判定。
        for platform in self.platforms:
            # 仅当玩家矩形与平台重叠时处理碰撞。
            if self.player_rect.colliderect(platform):
                # 如果在下落且脚底接触平台顶面，则站上平台。
                if self.vel_y >= 0 and self.player_rect.bottom >= platform.top:
                    self.player_rect.bottom = platform.top
                    self.vel_y = 0.0
                    self.on_ground = True

        # 若角色跌落到窗口外底部，判定为失败终局。
        if self.player_rect.top > SCREEN_HEIGHT:
            self.done = True

    def _nearest_rect_features(self, items: List[pygame.Rect]) -> np.ndarray:
        """提取最近目标矩形的6维归一化特征。"""
        # 若目标列表为空，则返回全零特征。
        if not items:
            return np.zeros(6, dtype=np.float32)

        # 计算玩家中心点，作为距离参考。
        player_cx = self.player_rect.centerx
        player_cy = self.player_rect.centery

        # 按欧氏距离选择最近目标。
        nearest = min(
            items,
            key=lambda rect: (rect.centerx - player_cx) ** 2 + (rect.centery - player_cy) ** 2,
        )

        # 计算相对偏移并归一化到[0,1]附近。
        rel_x = (nearest.centerx - player_cx) / SCREEN_WIDTH
        rel_y = (nearest.centery - player_cy) / SCREEN_HEIGHT

        # 返回固定6维向量：绝对位置/尺寸 + 相对位置。
        return np.array(
            [
                nearest.x / SCREEN_WIDTH,
                nearest.y / SCREEN_HEIGHT,
                nearest.width / SCREEN_WIDTH,
                nearest.height / SCREEN_HEIGHT,
                (rel_x + 1.0) / 2.0,
                (rel_y + 1.0) / 2.0,
            ],
            dtype=np.float32,
        )

    def _get_state(self) -> np.ndarray:
        """构造固定30维归一化状态向量。"""
        # 计算终点相对偏移。
        goal_dx = (self.goal.centerx - self.player_rect.centerx) / SCREEN_WIDTH
        goal_dy = (self.goal.centery - self.player_rect.centery) / SCREEN_HEIGHT

        # 过滤未收集金币，用于最近金币特征。
        remaining_coins = [
            coin_rect
            for idx, coin_rect in enumerate(self.coins_template)
            if not self.collected[idx]
        ]

        # 提取最近平台、最近陷阱、最近金币的局部结构特征。
        nearest_platform = self._nearest_rect_features(self.platforms)
        nearest_trap = self._nearest_rect_features(self.traps)
        nearest_coin = self._nearest_rect_features(remaining_coins)

        # 计算金币收集进度。
        coins_total = max(len(self.collected), 1)
        remaining_ratio = float(sum(not val for val in self.collected)) / float(coins_total)

        # 判断最近陷阱与金币是否在前方。
        trap_ahead = 1.0 if nearest_trap[4] >= 0.5 else 0.0
        coin_ahead = 1.0 if nearest_coin[4] >= 0.5 else 0.0

        # 基础状态特征（8维）。
        base_features = np.array(
            [
                self.player_rect.x / SCREEN_WIDTH,
                self.player_rect.y / SCREEN_HEIGHT,
                (self.vel_x + MOVE_SPEED) / (2.0 * MOVE_SPEED),
                (self.vel_y + 20.0) / 40.0,
                1.0 if self.on_ground else 0.0,
                (goal_dx + 1.0) / 2.0,
                (goal_dy + 1.0) / 2.0,
                self.step_count / max(float(MAX_STEPS_PER_EPISODE), 1.0),
            ],
            dtype=np.float32,
        )

        # 拼接所有子特征，得到目标30维向量。
        state = np.concatenate(
            [
                base_features,
                nearest_platform,
                nearest_trap,
                nearest_coin,
                np.array(
                    [
                        remaining_ratio,
                        trap_ahead,
                        coin_ahead,
                        1.0 if self.done else 0.0,
                    ],
                    dtype=np.float32,
                ),
            ],
            axis=0,
        )

        # 如果维度异常，使用断言立即暴露问题。
        assert state.shape[0] == 30, "State dimension must be exactly 30."

        # 将状态裁剪到[0,1]区间，提高学习稳定性。
        return np.clip(state, 0.0, 1.0).astype(np.float32)

    def draw_level(self, screen: pygame.Surface) -> None:
        """绘制关卡元素（平台、陷阱、金币、终点）。"""
        # 用白色清屏，保证每帧图像干净。
        screen.fill(WHITE)

        # 绘制平台。
        for platform in self.platforms:
            pygame.draw.rect(screen, GREEN, platform)

        # 绘制陷阱。
        for trap in self.traps:
            pygame.draw.rect(screen, RED, trap)

        # 绘制未收集金币。
        for idx, coin in enumerate(self.coins_template):
            if not self.collected[idx]:
                pygame.draw.rect(screen, YELLOW, coin)

        # 绘制终点旗帜。
        pygame.draw.rect(screen, ORANGE, self.goal)

    @staticmethod
    def tick(clock: pygame.time.Clock) -> None:
        """统一帧率控制。"""
        # 保持渲染与逻辑更新在固定FPS。
        clock.tick(FPS)


def action_is_valid(action: int) -> bool:
    """检查动作索引是否在合法范围。"""
    # 动作必须在[0, ACTION_DIM)范围内。
    return 0 <= action < ACTION_DIM
