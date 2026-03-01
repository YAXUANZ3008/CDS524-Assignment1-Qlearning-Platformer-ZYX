# CDS524 DQN 平台跳跃竞速作业

本项目实现了一个基于 **Pygame + PyTorch DQN** 的 2D 横版平台跳跃闯关竞速系统，满足以下模式：

- 玩家手动闯关（`human`）
- AI 自主闯关（`ai`）
- 人机同屏竞速（`race`）

## 1. 环境要求

- Python 3.8+
- Windows / Linux / macOS

## 2. 安装依赖

```bash
pip install -r requirements.txt
```

## 3. 训练 DQN

```bash
python train_dqn.py
```

训练输出：

- 模型权重：`checkpoints/dqn_platformer.pth`
- 学习曲线：`outputs/training_curve.png`

## 4. 运行可交互UI

### 4.1 玩家手动模式

```bash
python main.py --mode human
```

控制：

- `A`：左移
- `D`：右移
- `W`：原地跳跃
- `Q`：左跳
- `E`：右跳

### 4.2 AI 自主模式

```bash
python main.py --mode ai --model checkpoints/dqn_platformer.pth
```

### 4.3 人机同屏竞速

```bash
python main.py --mode race --model checkpoints/dqn_platformer.pth
```

## 5. 项目结构

- `config.py`：全局常量、状态/动作定义、DQN固定超参数
- `game_env.py`：平台闯关环境、奖励函数、30维状态编码
- `dqn_agent.py`：Q网络、经验回放、ε-greedy 与训练逻辑
- `train_dqn.py`：训练主程序与学习曲线绘图
- `main.py`：Pygame交互界面（三种模式）

## 6. 说明

- 状态空间固定 30 维归一化向量。
- 动作空间固定 5 个互斥动作：`0=左移、1=右移、2=原地跳跃、3=左跳、4=右跳`。
- 奖励函数为分层正负奖励：进度奖励、金币奖励、通关奖励、时间惩罚、陷阱惩罚等。
