"""DQN训练入口：执行采样、学习、保存模型与绘制曲线。"""

from __future__ import annotations

import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from config import MAX_EPISODES, MAX_STEPS_PER_EPISODE, MODEL_PATH, PLOT_PATH
from dqn_agent import DQNAgent
from game_env import PlatformerEnv


def moving_average(values: List[float], window: int = 10) -> np.ndarray:
    """计算滑动平均用于平滑曲线。"""
    # 若样本长度不足窗口，直接返回原数组。
    if len(values) < window:
        return np.array(values, dtype=np.float32)
    # 使用卷积快速计算窗口均值。
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(np.array(values, dtype=np.float32), kernel, mode="valid")


def train() -> None:
    """执行完整DQN训练流程。"""
    # 创建环境与智能体。
    env = PlatformerEnv()
    agent = DQNAgent()

    # 记录每回合指标，便于分析学习过程。
    episode_rewards: List[float] = []
    episode_lengths: List[int] = []
    success_flags: List[float] = []

    # 记录历史最佳回报，用于最优模型覆盖保存。
    best_reward = -1e9

    # 按固定回合数进行训练。
    for episode in range(1, MAX_EPISODES + 1):
        # 回合开始时重置环境与统计量。
        state = env.reset()
        total_reward = 0.0
        total_loss = 0.0
        loss_updates = 0

        # 在单回合内持续采样直到终止。
        for step in range(1, MAX_STEPS_PER_EPISODE + 1):
            # 根据ε-greedy策略选择动作。
            action = agent.select_action(state, train=True)
            # 与环境交互得到下一状态与奖励。
            result = env.step(action)
            next_state = result.state
            reward = result.reward
            done = result.done

            # 将本次经验写入回放池。
            agent.store_transition(state, action, reward, next_state, done)
            # 尝试执行一次参数更新。
            loss = agent.update()
            if loss is not None:
                total_loss += loss
                loss_updates += 1

            # 推进到下一时刻状态。
            state = next_state
            total_reward += reward

            # 回合结束则提前跳出。
            if done:
                break

        # 回合结束后衰减epsilon。
        agent.decay_epsilon()

        # 记录关键训练指标。
        episode_rewards.append(total_reward)
        episode_lengths.append(step)
        success_flags.append(1.0 if env.player_rect.colliderect(env.goal) else 0.0)

        # 若当前回报超过历史最好，则保存最优模型。
        if total_reward > best_reward:
            best_reward = total_reward
            agent.save(MODEL_PATH)

        # 每回合输出简明日志，展示学习状态。
        avg_loss = total_loss / max(loss_updates, 1)
        win_rate_20 = float(np.mean(success_flags[-20:])) if success_flags else 0.0
        print(
            f"Episode {episode:03d}/{MAX_EPISODES} | "
            f"Reward: {total_reward:7.2f} | Steps: {step:4d} | "
            f"Loss: {avg_loss:7.4f} | Epsilon: {agent.epsilon:.3f} | "
            f"WinRate@20: {win_rate_20:.2f}"
        )

    # 训练结束后再次保存最终模型（覆盖为最后一次参数）。
    agent.save(MODEL_PATH)

    # 绘制奖励曲线并保存图片。
    save_training_plot(episode_rewards)

    # 输出完成信息。
    print(f"Training finished. Model saved to: {MODEL_PATH}")
    print(f"Training curve saved to: {PLOT_PATH}")


def save_training_plot(episode_rewards: List[float]) -> None:
    """绘制并保存训练奖励曲线。"""
    # 若输出目录不存在则自动创建。
    os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)

    # 创建画布并绘制原始奖励曲线。
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, label="Episode Reward", alpha=0.45)

    # 计算并绘制滑动平均曲线提升可读性。
    smooth = moving_average(episode_rewards, window=10)
    if len(smooth) > 0:
        smooth_x = np.arange(len(smooth)) + (10 - 1)
        plt.plot(smooth_x, smooth, label="Moving Avg(10)", linewidth=2)

    # 设置图表标题与坐标轴标签。
    plt.title("DQN Training Reward Curve")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()

    # 保存图像到固定路径。
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close()


if __name__ == "__main__":
    # 直接运行脚本时启动训练。
    train()
