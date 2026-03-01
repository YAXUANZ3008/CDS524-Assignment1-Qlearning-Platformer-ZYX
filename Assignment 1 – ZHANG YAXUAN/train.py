"""
文件作用：DQN训练入口，负责按固定轮次训练并输出模型权重与奖励曲线。
作者：GitHub Copilot (GPT-5.3-Codex)
依赖：pygame、matplotlib、numpy、torch、game、dqn_agent
"""

from __future__ import annotations

import os
import sys
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pygame
import torch

from dqn_agent import Agent
from config import FPS, SCREEN_HEIGHT, SCREEN_WIDTH
from game import Game


def ensure_output_dirs() -> tuple[str, str]:
    """创建并返回模型与报告输出目录。"""
    # 获取当前文件所在目录，作为相对路径计算基准。
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # 按要求将模型目录固定到../model。
    model_dir = os.path.abspath(os.path.join(base_dir, "..", "model"))
    # 按要求将报告目录固定到../report。
    report_dir = os.path.abspath(os.path.join(base_dir, "..", "report"))

    # 若目录不存在则自动创建。
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    # 返回两个目录路径供调用方使用。
    return model_dir, report_dir


def save_reward_plot(rewards: List[float], file_path: str) -> None:
    """绘制并保存训练奖励曲线。"""
    # 创建图像画布。
    plt.figure(figsize=(10, 5))

    # 绘制每轮累计奖励原始曲线。
    plt.plot(rewards, label="Episode Reward", alpha=0.45)

    # 计算并绘制10轮滑动平均曲线，提升趋势可读性。
    if len(rewards) >= 10:
        kernel = np.ones(10, dtype=np.float32) / 10.0
        smooth = np.convolve(np.array(rewards, dtype=np.float32), kernel, mode="valid")
        smooth_x = np.arange(len(smooth)) + 9
        plt.plot(smooth_x, smooth, label="Moving Avg(10)", linewidth=2)

    # 设置图表标题和坐标轴说明。
    plt.title("DQN Training Reward Curve")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()

    # 保存图像并释放资源。
    plt.savefig(file_path, dpi=150)
    plt.close()


def save_agent_model(agent: Agent, file_path: str) -> None:
    """保存Agent在线网络权重与训练状态。"""
    # 调用Agent内置save接口，统一保存格式。
    agent.save(file_path)


def train() -> None:
    """执行DQN训练主流程。"""
    # -------------------------------
    # 训练配置（按作业要求固定）
    # -------------------------------
    # 总训练轮数固定为300轮。
    total_episodes = 300
    # 单轮最大步数固定为3600步（60秒 * 60FPS）。
    max_steps_per_episode = 3600

    # 创建输出目录并生成固定输出文件路径。
    model_dir, report_dir = ensure_output_dirs()
    latest_model_path = os.path.join(model_dir, "dqn_latest.pth")
    final_model_path = os.path.join(model_dir, "dqn_final.pth")
    reward_curve_path = os.path.join(report_dir, "training_reward_curve.png")

    # 初始化pygame，供Game对象与事件系统使用。
    pygame.init()

    # 创建窗口对象（训练模式下可隐藏窗口以加速）。
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), flags=pygame.HIDDEN)
    pygame.display.set_caption("DQN Training Mode")

    # 初始化游戏对象为训练模式（关闭UI绘制加速训练）。
    game = Game(screen=screen, training_mode=True)

    # 初始化Agent实例，严格复用Agent内部固定超参数。
    agent = Agent()

    # 记录训练过程指标：累计奖励、平均损失、通关标记。
    episode_rewards: List[float] = []
    episode_losses: List[float] = []
    episode_success: List[int] = []

    # 训练是否被用户中断的标记。
    interrupted = False

    # 主训练循环：共300轮。
    for episode in range(1, total_episodes + 1):
        # 每轮开始前重置游戏环境和角色状态。
        game.reset()
        # 将状态置为running，进入可更新流程。
        game.game_state = "running"

        # 重置Agent回合追踪变量（金币计数、步数、跳跃标记）。
        agent.reset_episode_tracking(game)

        # 获取当前回合初始状态。
        state = agent.get_state(game, game.ai_character)

        # 初始化本轮累计奖励。
        total_reward = 0.0
        # 初始化本轮损失累计。
        total_loss = 0.0
        # 初始化本轮更新次数计数。
        update_count = 0
        # 初始化本轮是否成功通关标记。
        success = 0

        # 单轮步进循环，最多3600步。
        for _step in range(max_steps_per_episode):
            # 处理系统事件，支持训练过程中正常退出。
            for event in pygame.event.get():
                # 若用户关闭窗口，则设置中断标记并结束训练。
                if event.type == pygame.QUIT:
                    interrupted = True
                # 若用户按ESC，则设置中断标记并结束训练。
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    interrupted = True

            # 若检测到中断请求，立即退出当前回合循环。
            if interrupted:
                break

            # 基于ε-greedy策略选择离散动作。
            action = agent.select_action(state)

            # 将动作作用到AI角色。
            game.ai_character.execute_action(action)
            # 执行AI角色一帧物理与碰撞更新。
            game.ai_character.update(game.level)

            # 获取下一状态，供经验存储与网络学习使用。
            next_state = agent.get_state(game, game.ai_character)

            # 计算本帧奖励与是否终止。
            reward, done = agent.get_reward(game, game.ai_character, state)

            # 记录经验到回放池。
            agent.store_memory(state, action, reward, next_state, done)

            # 尝试执行一次网络更新。
            loss = agent.update()
            if loss is not None:
                total_loss += loss
                update_count += 1

            # 推进状态并累计奖励。
            state = next_state
            total_reward += reward

            # 若到达终点，标记本轮通关成功。
            if game.ai_character.reached_goal:
                success = 1

            # 若当前回合终止则结束本轮步进循环。
            if done:
                break

        # 若训练被中断，提前结束总训练循环。
        if interrupted:
            print("Training interrupted by user.")
            break

        # 记录本轮训练指标。
        episode_rewards.append(total_reward)
        episode_success.append(success)
        # 计算本轮平均损失（无更新时记0）。
        avg_loss = total_loss / update_count if update_count > 0 else 0.0
        episode_losses.append(avg_loss)

        # 每10轮打印日志并保存模型与奖励曲线。
        if episode % 10 == 0:
            # 计算最近10轮统计指标。
            recent_reward = float(np.mean(episode_rewards[-10:])) if episode_rewards else 0.0
            recent_loss = float(np.mean(episode_losses[-10:])) if episode_losses else 0.0
            recent_success_rate = (
                float(np.mean(episode_success[-10:])) * 100.0 if episode_success else 0.0
            )

            # 打印训练进度日志。
            print(
                f"Episode {episode:03d}/{total_episodes} | "
                f"AvgReward@10: {recent_reward:8.2f} | "
                f"AvgLoss@10: {recent_loss:8.4f} | "
                f"Success@10: {recent_success_rate:6.2f}% | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

            # 保存最新模型权重。
            save_agent_model(agent, latest_model_path)
            # 保存奖励曲线图，便于作业报告引用。
            save_reward_plot(episode_rewards, reward_curve_path)

    # 训练结束（含提前中断）后保存最终模型。
    save_agent_model(agent, final_model_path)

    # 计算总体训练统计结果。
    total_finished = len(episode_rewards)
    overall_success_rate = (
        float(np.mean(episode_success)) * 100.0 if episode_success else 0.0
    )
    overall_avg_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
    overall_avg_loss = float(np.mean(episode_losses)) if episode_losses else 0.0

    # 打印训练总结，包含通关率等作业关注指标。
    print("=" * 72)
    print("Training Summary")
    print(f"Episodes completed: {total_finished}/{total_episodes}")
    print(f"Overall Success Rate: {overall_success_rate:.2f}%")
    print(f"Overall Avg Reward: {overall_avg_reward:.2f}")
    print(f"Overall Avg Loss: {overall_avg_loss:.4f}")
    print(f"Latest model path: {latest_model_path}")
    print(f"Final model path: {final_model_path}")
    print(f"Reward curve path: {reward_curve_path}")
    print("=" * 72)

    # 释放pygame资源，防止进程残留。
    pygame.quit()


def main() -> None:
    """训练脚本主入口。"""
    try:
        # 启动训练流程。
        train()
    except KeyboardInterrupt:
        # 捕获Ctrl+C中断，确保资源释放并友好退出。
        pygame.quit()
        print("Training stopped by KeyboardInterrupt.")
        sys.exit(0)


if __name__ == "__main__":
    # 以脚本方式运行时执行main。
    main()
