"""Pygame交互入口：支持human / ai / race三种闯关模式。"""

from __future__ import annotations

import argparse
import os
from typing import Tuple

import pygame

from config import ACTION_NAMES, BLACK, BLUE, FPS, MODEL_PATH, PURPLE, SCREEN_HEIGHT, SCREEN_WIDTH
from dqn_agent import DQNAgent
from game_env import PlatformerEnv


def action_from_keyboard(keys: pygame.key.ScancodeWrapper) -> int:
    """将键盘输入映射到5个互斥动作。"""
    # 优先处理组合跳跃动作，保证按键行为明确。
    if keys[pygame.K_q]:
        return 3
    if keys[pygame.K_e]:
        return 4
    # 再处理原地跳跃。
    if keys[pygame.K_w]:
        return 2
    # 处理左右平移。
    if keys[pygame.K_a]:
        return 0
    if keys[pygame.K_d]:
        return 1
    # 无输入时默认右移，避免动作空间之外的“空动作”。
    return 1


def draw_text(
    screen: pygame.Surface,
    font: pygame.font.Font,
    text: str,
    x: int,
    y: int,
    color: Tuple[int, int, int] = BLACK,
) -> None:
    """在指定位置渲染单行文字。"""
    # 将字符串渲染为文本表面。
    text_surface = font.render(text, True, color)
    # 将文本绘制到屏幕目标坐标。
    screen.blit(text_surface, (x, y))


def run_human_mode() -> None:
    """运行玩家手动闯关模式。"""
    # 初始化Pygame系统。
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("CDS524 Platformer - Human Mode")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)

    # 创建环境并初始化状态。
    env = PlatformerEnv()
    state = env.reset()
    last_reward = 0.0
    last_action = 1

    running = True
    while running:
        # 处理窗口事件，支持安全退出。
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # 读取当前键盘状态并映射为动作。
        keys = pygame.key.get_pressed()
        action = action_from_keyboard(keys)
        last_action = action

        # 与环境执行一步交互。
        result = env.step(action)
        state = result.state
        last_reward = result.reward

        # 若回合结束，按R键重开。
        if result.done and keys[pygame.K_r]:
            state = env.reset()
            last_reward = 0.0

        # 绘制关卡元素。
        env.draw_level(screen)
        # 绘制玩家实体（蓝色）。
        pygame.draw.rect(screen, BLUE, env.player_rect)

        # 叠加状态文本，便于调试和演示。
        draw_text(screen, font, "MODE: HUMAN", 10, 8)
        draw_text(screen, font, f"Action: {ACTION_NAMES[last_action]}", 10, 30)
        draw_text(screen, font, f"Reward: {last_reward:+.3f}", 10, 52)
        draw_text(screen, font, f"Progress: {state[0] * 100:.1f}%", 10, 74)
        draw_text(
            screen,
            font,
            f"Coins: {int(sum(env.collected))}/{len(env.collected)}",
            10,
            96,
        )
        if result.done:
            draw_text(screen, font, "Episode Done. Press R to restart.", 10, 118)

        # 刷新屏幕与帧率控制。
        pygame.display.flip()
        clock.tick(FPS)

    # 退出前释放Pygame资源。
    pygame.quit()


def run_ai_mode(model_path: str) -> None:
    """运行AI自主闯关模式。"""
    # 初始化Pygame系统。
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("CDS524 Platformer - AI Mode")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)

    # 创建环境与智能体。
    env = PlatformerEnv()
    agent = DQNAgent()

    # 尝试加载模型；若不存在则用随机初始化网络推理。
    if os.path.exists(model_path):
        agent.load(model_path)
    agent.epsilon = 0.0

    # 初始化回合状态。
    state = env.reset()
    last_reward = 0.0
    last_action = 1

    running = True
    while running:
        # 处理退出事件。
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # 用贪心策略选择动作。
        action = agent.select_action(state, train=False)
        last_action = action

        # 与环境交互。
        result = env.step(action)
        state = result.state
        last_reward = result.reward

        # 结束后按R重置。
        keys = pygame.key.get_pressed()
        if result.done and keys[pygame.K_r]:
            state = env.reset()
            last_reward = 0.0

        # 绘制关卡和AI角色。
        env.draw_level(screen)
        pygame.draw.rect(screen, PURPLE, env.player_rect)

        # 绘制实时状态信息。
        draw_text(screen, font, "MODE: AI", 10, 8)
        draw_text(screen, font, f"Action: {ACTION_NAMES[last_action]}", 10, 30)
        draw_text(screen, font, f"Reward: {last_reward:+.3f}", 10, 52)
        draw_text(screen, font, f"Progress: {state[0] * 100:.1f}%", 10, 74)
        draw_text(
            screen,
            font,
            f"Coins: {int(sum(env.collected))}/{len(env.collected)}",
            10,
            96,
        )
        if result.done:
            draw_text(screen, font, "Episode Done. Press R to restart.", 10, 118)

        # 刷新帧。
        pygame.display.flip()
        clock.tick(FPS)

    # 退出释放资源。
    pygame.quit()


def run_race_mode(model_path: str) -> None:
    """运行人机同屏竞速模式。"""
    # 初始化Pygame系统。
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("CDS524 Platformer - Race Mode")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)

    # 分别创建人类与AI环境，保证双方互不干扰地计算奖励和结束状态。
    human_env = PlatformerEnv()
    ai_env = PlatformerEnv()

    # 创建智能体并加载模型。
    agent = DQNAgent()
    if os.path.exists(model_path):
        agent.load(model_path)
    agent.epsilon = 0.0

    # 重置双方初始状态。
    human_state = human_env.reset()
    ai_state = ai_env.reset()
    human_reward = 0.0
    ai_reward = 0.0
    human_action = 1
    ai_action = 1

    running = True
    while running:
        # 处理窗口退出事件。
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # 读取玩家动作并执行一步。
        keys = pygame.key.get_pressed()
        human_action = action_from_keyboard(keys)
        if not human_env.done:
            human_result = human_env.step(human_action)
            human_state = human_result.state
            human_reward = human_result.reward

        # 由AI策略输出动作并执行一步。
        if not ai_env.done:
            ai_action = agent.select_action(ai_state, train=False)
            ai_result = ai_env.step(ai_action)
            ai_state = ai_result.state
            ai_reward = ai_result.reward

        # 若双方都结束，支持按R统一重开。
        if human_env.done and ai_env.done and keys[pygame.K_r]:
            human_state = human_env.reset()
            ai_state = ai_env.reset()
            human_reward = 0.0
            ai_reward = 0.0
            human_action = 1
            ai_action = 1

        # 使用human环境绘制共同关卡背景。
        human_env.draw_level(screen)

        # 绘制玩家（蓝）和AI（紫）角色位置。
        pygame.draw.rect(screen, BLUE, human_env.player_rect)
        pygame.draw.rect(screen, PURPLE, ai_env.player_rect)

        # 绘制竞速信息。
        draw_text(screen, font, "MODE: RACE", 10, 8)
        draw_text(screen, font, f"Human Action: {ACTION_NAMES[human_action]}", 10, 30)
        draw_text(screen, font, f"Human Reward: {human_reward:+.3f}", 10, 52)
        draw_text(screen, font, f"Human Progress: {human_state[0] * 100:.1f}%", 10, 74)
        draw_text(screen, font, f"AI Action: {ACTION_NAMES[ai_action]}", 420, 30)
        draw_text(screen, font, f"AI Reward: {ai_reward:+.3f}", 420, 52)
        draw_text(screen, font, f"AI Progress: {ai_state[0] * 100:.1f}%", 420, 74)

        # 显示竞速结果状态。
        winner_text = ""
        if human_env.player_rect.colliderect(human_env.goal) and not ai_env.player_rect.colliderect(ai_env.goal):
            winner_text = "Winner: Human"
        elif ai_env.player_rect.colliderect(ai_env.goal) and not human_env.player_rect.colliderect(human_env.goal):
            winner_text = "Winner: AI"
        elif human_env.done and ai_env.done:
            winner_text = "Both finished. Press R to restart."
        if winner_text:
            draw_text(screen, font, winner_text, 10, 100)

        # 刷新显示并维持固定帧率。
        pygame.display.flip()
        clock.tick(FPS)

    # 程序退出时释放资源。
    pygame.quit()


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    # 创建命令行解析器。
    parser = argparse.ArgumentParser(description="CDS524 DQN Platformer")
    # 选择运行模式：human、ai、race。
    parser.add_argument(
        "--mode",
        type=str,
        default="human",
        choices=["human", "ai", "race"],
        help="运行模式：human / ai / race",
    )
    # 指定模型权重路径。
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_PATH,
        help="DQN模型路径",
    )
    return parser.parse_args()


def main() -> None:
    """程序主入口。"""
    # 读取命令行参数。
    args = parse_args()

    # 根据模式分发到对应运行函数。
    if args.mode == "human":
        run_human_mode()
    elif args.mode == "ai":
        run_ai_mode(args.model)
    else:
        run_race_mode(args.model)


if __name__ == "__main__":
    # 以脚本方式执行时进入主程序。
    main()
