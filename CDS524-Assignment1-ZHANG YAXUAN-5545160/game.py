"""
文件作用：定义游戏相关的全部类与主入口（GameObject、Level、Player、AIAgentCharacter、Game）。
作者：GitHub Copilot (GPT-5.3-Codex)
依赖：pygame、numpy、config（全局常量与颜色）
"""

from __future__ import annotations

import os
from typing import Optional, Type

import numpy as np
import pygame

from config import (
    ACTION_NAMES,
    BLACK,
    BLUE,
    FPS,
    GRAVITY,
    GREEN,
    JUMP_FORCE,
    MOVE_SPEED,
    ORANGE,
    PLAYER_SIZE,
    PURPLE,
    RED,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    WHITE,
    YELLOW,
)


class GameObject:
    """所有游戏元素的父类。"""

    def __init__(self, rect: pygame.Rect, color: tuple[int, int, int]) -> None:
        """初始化矩形碰撞盒与绘制颜色。"""
        self.rect = rect
        self.color = color

    def draw(self, screen: pygame.Surface) -> None:
        """绘制纯色矩形。"""
        pygame.draw.rect(screen, self.color, self.rect)


class Platform(GameObject):
    """平台对象：角色可站立的碰撞体。"""

    def __init__(self, rect: pygame.Rect) -> None:
        """创建绿色平台。"""
        super().__init__(rect, GREEN)


class Spike(GameObject):
    """陷阱对象：触碰即死亡。"""

    def __init__(self, rect: pygame.Rect) -> None:
        """创建红色尖刺。"""
        super().__init__(rect, RED)


class Coin(GameObject):
    """金币对象：触碰后加分并移除。"""

    def __init__(self, rect: pygame.Rect) -> None:
        """创建黄色金币。"""
        super().__init__(rect, YELLOW)


class Goal(GameObject):
    """终点对象：触碰后通关。"""

    def __init__(self, rect: pygame.Rect) -> None:
        """创建橙色终点旗帜。"""
        super().__init__(rect, ORANGE)


class Level(GameObject):
    """固定单关卡管理类。"""

    def __init__(self) -> None:
        """初始化固定地图布局（平台、陷阱、金币、终点）。"""
        # Level继承GameObject仅为统一类型管理，这里使用全屏管理Rect。
        super().__init__(pygame.Rect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT), GREEN)

        # 创建地面与阶梯平台（含完整地面，且至少5个阶梯平台）。
        self.platforms = [
            Platform(pygame.Rect(0, SCREEN_HEIGHT - 40, SCREEN_WIDTH, 40)),
            Platform(pygame.Rect(90, 320, 120, 16)),
            Platform(pygame.Rect(240, 270, 120, 16)),
            Platform(pygame.Rect(390, 220, 120, 16)),
            Platform(pygame.Rect(540, 170, 120, 16)),
            Platform(pygame.Rect(680, 120, 100, 16)),
        ]

        # 创建3个间隔陷阱，放置在中部地面区域。
        self.spikes = [
            Spike(pygame.Rect(200, SCREEN_HEIGHT - 50, 40, 10)),
            Spike(pygame.Rect(380, SCREEN_HEIGHT - 50, 40, 10)),
            Spike(pygame.Rect(560, SCREEN_HEIGHT - 50, 40, 10)),
        ]

        # 创建3个金币并分布在中高平台上。
        self.coins = [
            Coin(pygame.Rect(285, 245, 14, 14)),
            Coin(pygame.Rect(435, 195, 14, 14)),
            Coin(pygame.Rect(585, 145, 14, 14)),
        ]

        # 将终点放置在最右侧最高平台处。
        self.goal = Goal(pygame.Rect(740, 65, 20, 55))

    def draw(self, screen: pygame.Surface) -> None:
        """绘制关卡中的所有元素。"""
        for platform in self.platforms:
            platform.draw(screen)
        for spike in self.spikes:
            spike.draw(screen)
        for coin in self.coins:
            coin.draw(screen)
        self.goal.draw(screen)


class Player(GameObject):
    """玩家角色类：负责输入、物理与碰撞。"""

    def __init__(self, x: int, y: int) -> None:
        """初始化玩家状态。"""
        super().__init__(pygame.Rect(x, y, PLAYER_SIZE, PLAYER_SIZE), BLUE)
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.on_ground = False
        self.is_alive = True
        self.score = 0
        self.reached_goal = False

    def handle_input(self) -> dict[str, bool]:
        """处理键盘输入并设置速度。"""
        command = {"restart": False, "quit": False}
        self.vel_x = 0.0
        keys = pygame.key.get_pressed()

        if keys[pygame.K_a]:
            self.vel_x = -MOVE_SPEED
        if keys[pygame.K_d]:
            self.vel_x = MOVE_SPEED
        if keys[pygame.K_SPACE] and self.on_ground:
            self.vel_y = JUMP_FORCE
            self.on_ground = False
        if keys[pygame.K_r]:
            command["restart"] = True
        if keys[pygame.K_ESCAPE]:
            command["quit"] = True

        return command

    def update(self, level: Level) -> None:
        """每帧更新角色状态（重力、移动、碰撞、收集、终点）。"""
        if (not self.is_alive) or self.reached_goal:
            return

        # 应用重力更新垂直速度。
        self.vel_y += GRAVITY

        # 先更新水平位置并做水平碰撞修正。
        self.rect.x += int(self.vel_x)
        self._check_horizontal_collision(level)

        # 再更新垂直位置并做垂直碰撞修正。
        self.rect.y += int(self.vel_y)
        self._check_vertical_collision(level)

        # 限制左右边界，防止角色离开屏幕。
        if self.rect.left < 0:
            self.rect.left = 0
        if self.rect.right > SCREEN_WIDTH:
            self.rect.right = SCREEN_WIDTH

        # 上下越界直接死亡。
        if self.rect.top > SCREEN_HEIGHT or self.rect.bottom < 0:
            self.is_alive = False
            return

        # 碰到任意尖刺则死亡。
        for spike in level.spikes:
            if self.rect.colliderect(spike.rect):
                self.is_alive = False
                return

        # 收集金币：碰撞即移除并加30分。
        collected_indices: list[int] = []
        for index, coin in enumerate(level.coins):
            if self.rect.colliderect(coin.rect):
                collected_indices.append(index)

        for index in reversed(collected_indices):
            level.coins.pop(index)
            self.score += 30

        # 触碰终点即通关。
        if self.rect.colliderect(level.goal.rect):
            self.reached_goal = True

    def reset(self, x: int, y: int) -> None:
        """重置玩家到指定起点并清空状态。"""
        self.rect.x = x
        self.rect.y = y
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.on_ground = True
        self.is_alive = True
        self.score = 0
        self.reached_goal = False

    def _check_horizontal_collision(self, level: Level) -> None:
        """处理水平方向与平台的碰撞修正。"""
        for platform in level.platforms:
            if self.rect.colliderect(platform.rect):
                if self.vel_x > 0:
                    self.rect.right = platform.rect.left
                    self.vel_x = 0.0
                elif self.vel_x < 0:
                    self.rect.left = platform.rect.right
                    self.vel_x = 0.0

    def _check_vertical_collision(self, level: Level) -> None:
        """处理垂直方向与平台的碰撞修正。"""
        self.on_ground = False
        for platform in level.platforms:
            if self.rect.colliderect(platform.rect):
                if self.vel_y > 0:
                    self.rect.bottom = platform.rect.top
                    self.vel_y = 0.0
                    self.on_ground = True
                elif self.vel_y < 0:
                    self.rect.top = platform.rect.bottom
                    self.vel_y = 0.0


class AIAgentCharacter(Player):
    """AI角色类：复用Player，仅新增离散动作执行方法。"""

    def __init__(self, x: int, y: int) -> None:
        """初始化AI角色并使用紫色区分玩家。"""
        super().__init__(x, y)
        self.color = PURPLE

    def execute_action(self, action: int) -> None:
        """执行DQN动作编号对应的移动/跳跃控制。"""
        self.vel_x = 0.0

        if action == 0:
            self.vel_x = -MOVE_SPEED
        elif action == 1:
            self.vel_x = MOVE_SPEED
        elif action == 2:
            if self.on_ground:
                self.vel_y = JUMP_FORCE
                self.on_ground = False
        elif action == 3:
            self.vel_x = -MOVE_SPEED
            if self.on_ground:
                self.vel_y = JUMP_FORCE
                self.on_ground = False
        elif action == 4:
            self.vel_x = MOVE_SPEED
            if self.on_ground:
                self.vel_y = JUMP_FORCE
                self.on_ground = False


class Game:
    """游戏主控制类：事件、状态更新、渲染、AI模型接入。"""

    def __init__(self, screen: pygame.Surface, training_mode: bool = False) -> None:
        """初始化窗口、关卡、角色和运行状态。"""
        self.screen = screen
        self.clock = pygame.time.Clock()
        self.training_mode = training_mode

        self.font = pygame.font.SysFont("consolas", 20)
        self.large_font = pygame.font.SysFont("consolas", 30)

        self.level = Level()
        self.player = Player(40, SCREEN_HEIGHT - 40 - PLAYER_SIZE)
        self.ai_character = AIAgentCharacter(40, SCREEN_HEIGHT - 40 - PLAYER_SIZE)

        self.game_state = "start"
        self.ai_model: Optional[object] = None
        self.current_reward = 0.0
        self.current_action = 1
        self.result_text = ""

    def handle_events(self) -> bool:
        """处理全局事件并返回是否继续运行。"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                # 修复说明：
                # 原逻辑仅监听K_y，部分输入法/键盘布局下可能无法稳定触发该键值。
                # 这里扩展开始页启动键为Y/空格/回车，并增加unicode兜底判断，
                # 只影响start页面进入running的入口，不改变其他状态流程。
                if (
                    self.game_state == "start"
                    and (
                        event.key in (pygame.K_y, pygame.K_SPACE, pygame.K_RETURN)
                        or event.unicode.lower() == "y"
                    )
                ):
                    self.reset()
                    self.game_state = "running"
                if event.key == pygame.K_r and self.game_state in ("running", "game_over"):
                    self.reset()
                    self.game_state = "running"

        # 修复说明：
        # 若KEYDOWN事件被短暂丢失，则使用按键状态继续兜底；
        # 同步支持Y/空格/回车三种开始键，提升可用性与兼容性。
        if self.game_state == "start":
            pygame.event.pump()
            keys = pygame.key.get_pressed()
            if keys[pygame.K_y] or keys[pygame.K_SPACE] or keys[pygame.K_RETURN]:
                self.reset()
                self.game_state = "running"

        return True

    def reset(self) -> None:
        """重置关卡与角色到初始状态。"""
        self.level = Level()
        self.player.reset(40, SCREEN_HEIGHT - 40 - PLAYER_SIZE)
        self.ai_character.reset(40, SCREEN_HEIGHT - 40 - PLAYER_SIZE)
        self.current_reward = 0.0
        self.current_action = 1
        self.result_text = ""

    def _build_ai_state(self) -> np.ndarray:
        """构造30维AI状态向量供模型决策。"""
        ai_cx = self.ai_character.rect.centerx
        ai_cy = self.ai_character.rect.centery

        goal_dx = (self.level.goal.rect.centerx - ai_cx) / SCREEN_WIDTH
        goal_dy = (self.level.goal.rect.centery - ai_cy) / SCREEN_HEIGHT

        nearest_platform = min(
            self.level.platforms,
            key=lambda p: (p.rect.centerx - ai_cx) ** 2 + (p.rect.centery - ai_cy) ** 2,
        )
        nearest_spike = min(
            self.level.spikes,
            key=lambda s: (s.rect.centerx - ai_cx) ** 2 + (s.rect.centery - ai_cy) ** 2,
        )

        if self.level.coins:
            nearest_coin = min(
                self.level.coins,
                key=lambda c: (c.rect.centerx - ai_cx) ** 2 + (c.rect.centery - ai_cy) ** 2,
            )
            coin_features = np.array(
                [
                    nearest_coin.rect.x / SCREEN_WIDTH,
                    nearest_coin.rect.y / SCREEN_HEIGHT,
                    nearest_coin.rect.width / SCREEN_WIDTH,
                    nearest_coin.rect.height / SCREEN_HEIGHT,
                    ((nearest_coin.rect.centerx - ai_cx) / SCREEN_WIDTH + 1.0) / 2.0,
                    ((nearest_coin.rect.centery - ai_cy) / SCREEN_HEIGHT + 1.0) / 2.0,
                ],
                dtype=np.float32,
            )
        else:
            coin_features = np.zeros(6, dtype=np.float32)

        base_features = np.array(
            [
                self.ai_character.rect.x / SCREEN_WIDTH,
                self.ai_character.rect.y / SCREEN_HEIGHT,
                (self.ai_character.vel_x + MOVE_SPEED) / (2.0 * MOVE_SPEED),
                (self.ai_character.vel_y + 20.0) / 40.0,
                1.0 if self.ai_character.on_ground else 0.0,
                (goal_dx + 1.0) / 2.0,
                (goal_dy + 1.0) / 2.0,
                0.0,
            ],
            dtype=np.float32,
        )
        platform_features = np.array(
            [
                nearest_platform.rect.x / SCREEN_WIDTH,
                nearest_platform.rect.y / SCREEN_HEIGHT,
                nearest_platform.rect.width / SCREEN_WIDTH,
                nearest_platform.rect.height / SCREEN_HEIGHT,
                ((nearest_platform.rect.centerx - ai_cx) / SCREEN_WIDTH + 1.0) / 2.0,
                ((nearest_platform.rect.centery - ai_cy) / SCREEN_HEIGHT + 1.0) / 2.0,
            ],
            dtype=np.float32,
        )
        spike_features = np.array(
            [
                nearest_spike.rect.x / SCREEN_WIDTH,
                nearest_spike.rect.y / SCREEN_HEIGHT,
                nearest_spike.rect.width / SCREEN_WIDTH,
                nearest_spike.rect.height / SCREEN_HEIGHT,
                ((nearest_spike.rect.centerx - ai_cx) / SCREEN_WIDTH + 1.0) / 2.0,
                ((nearest_spike.rect.centery - ai_cy) / SCREEN_HEIGHT + 1.0) / 2.0,
            ],
            dtype=np.float32,
        )
        tail_features = np.array(
            [
                len(self.level.coins) / 3.0,
                1.0 if nearest_spike.rect.centerx >= ai_cx else 0.0,
                1.0 if (self.level.coins and nearest_coin.rect.centerx >= ai_cx) else 0.0,
                0.0 if self.ai_character.is_alive else 1.0,
            ],
            dtype=np.float32,
        )

        state = np.concatenate(
            [base_features, platform_features, spike_features, coin_features, tail_features],
            axis=0,
        )
        return np.clip(state, 0.0, 1.0).astype(np.float32)

    def _compute_ai_reward(self, prev_x: float) -> float:
        """计算用于UI显示的AI帧级奖励。"""
        reward = -0.01
        delta_x = float(self.ai_character.rect.x) - prev_x
        reward += delta_x * 0.01
        if delta_x < 0:
            reward -= 0.02
        if not self.ai_character.is_alive:
            reward -= 5.0
        if self.ai_character.reached_goal:
            reward += 10.0
        return float(reward)

    def update(self, training_mode: bool = False) -> None:
        """更新游戏状态。"""
        if self.game_state != "running":
            return

        # 非训练模式下处理玩家输入并更新玩家。
        if not (training_mode or self.training_mode):
            command = self.player.handle_input()
            if command["restart"]:
                self.reset()
            if command["quit"]:
                self.game_state = "game_over"
                self.result_text = "Exit requested by player"
                return
            self.player.update(self.level)

        # AI执行动作：有模型则推理动作，无模型则默认右移。
        prev_ai_x = float(self.ai_character.rect.x)
        if self.ai_character.is_alive and (not self.ai_character.reached_goal):
            if self.ai_model is not None:
                state = self._build_ai_state()
                self.current_action = int(self.ai_model.select_action(state))
            else:
                self.current_action = 1
            self.ai_character.execute_action(self.current_action)
            self.ai_character.update(self.level)

        self.current_reward = self._compute_ai_reward(prev_ai_x)

        player_dead = not self.player.is_alive
        ai_dead = not self.ai_character.is_alive
        player_win = self.player.reached_goal
        ai_win = self.ai_character.reached_goal

        if player_win and (not ai_win):
            self.game_state = "game_over"
            self.result_text = "Result: Player Wins"
        elif ai_win and (not player_win):
            self.game_state = "game_over"
            self.result_text = "Result: AI Wins"
        elif player_win and ai_win:
            self.game_state = "game_over"
            self.result_text = "Result: Draw (Both Reached Goal)"
        elif player_dead and ai_dead:
            self.game_state = "game_over"
            self.result_text = "Result: Draw (Both Eliminated)"

    def _draw_text(self, text: str, x: int, y: int, large: bool = False) -> None:
        """绘制单行文本工具。"""
        font = self.large_font if large else self.font
        surface = font.render(text, True, BLACK)
        self.screen.blit(surface, (x, y))

    def draw(self, training_mode: bool = False) -> None:
        """按页面状态绘制UI。"""
        if training_mode or self.training_mode:
            return

        self.screen.fill(WHITE)

        if self.game_state == "start":
            self._draw_text("CDS524 DQN Platformer Race", 190, 70, large=True)
            self._draw_text("Rules:", 80, 130)
            self._draw_text("- Reach the goal flag before your opponent", 80, 160)
            self._draw_text("- Hit spikes: immediate death", 80, 190)
            self._draw_text("- Collect coins for score (+30 each)", 80, 220)
            self._draw_text("Controls: A=Left, D=Right, SPACE=Jump", 80, 270)
            self._draw_text("Press Y to Start", 80, 310)
            self._draw_text("Press ESC to Exit", 80, 340)

        elif self.game_state == "running":
            self.level.draw(self.screen)
            self.player.draw(self.screen)
            self.ai_character.draw(self.screen)

            player_progress = (self.player.rect.x / max(self.level.goal.rect.x, 1)) * 100.0
            ai_progress = (self.ai_character.rect.x / max(self.level.goal.rect.x, 1)) * 100.0

            self._draw_text(f"Game State: {self.game_state}", 10, 8)
            self._draw_text(f"Player Score: {self.player.score}", 10, 34)
            self._draw_text(f"Player Progress: {player_progress:.1f}%", 10, 60)
            self._draw_text(f"AI Progress: {ai_progress:.1f}%", 10, 86)
            self._draw_text(f"AI Action: {ACTION_NAMES.get(self.current_action, 'UNKNOWN')}", 430, 34)
            self._draw_text(f"AI Reward: {self.current_reward:+.3f}", 430, 60)
            self._draw_text(f"Player Alive: {self.player.is_alive}", 430, 86)
            self._draw_text(f"AI Alive: {self.ai_character.is_alive}", 430, 112)
            self._draw_text("R: Restart | ESC: Exit", 10, 360)

        elif self.game_state == "game_over":
            self._draw_text("Game Over", 320, 90, large=True)
            self._draw_text(self.result_text if self.result_text else "Result: Unknown", 180, 150)
            self._draw_text(f"Final Player Score: {self.player.score}", 180, 190)
            self._draw_text(
                f"Final AI Action: {ACTION_NAMES.get(self.current_action, 'UNKNOWN')}",
                180,
                220,
            )
            self._draw_text(f"Final AI Reward: {self.current_reward:+.3f}", 180, 250)
            self._draw_text("Press R to Restart", 180, 300)
            self._draw_text("Press ESC to Exit", 180, 330)

    def load_ai_model(self, model_path: str, agent_class: Type[object]) -> None:
        """按相对路径加载训练好的AI模型。"""
        if not os.path.exists(model_path):
            self.ai_model = None
            return

        agent = agent_class()
        if hasattr(agent, "load"):
            agent.load(model_path)
        if hasattr(agent, "epsilon"):
            agent.epsilon = 0.0
        self.ai_model = agent


def main() -> None:
    """游戏主入口：可独立运行手动闯关窗口。"""
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("CDS524 DQN Platformer")

    game = Game(screen=screen, training_mode=False)

    # 尝试按相对路径加载模型；不存在时回退到默认AI动作。
    try:
        from dqn_agent import Agent  # 局部导入避免循环依赖风险。

        default_model_path = os.path.join("..", "model", "dqn_final.pth")
        game.load_ai_model(default_model_path, Agent)
    except Exception:
        game.ai_model = None

    running = True
    while running:
        running = game.handle_events()
        game.update(training_mode=False)
        game.draw(training_mode=False)
        pygame.display.flip()
        game.clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
