"""项目全局配置与常量定义。"""

# -------------------------------
# 作业强制全局常量（严禁修改）
# -------------------------------
SCREEN_WIDTH = 800  # 游戏窗口宽度
SCREEN_HEIGHT = 400  # 游戏窗口高度
FPS = 60  # 游戏帧率
PLAYER_SIZE = 30  # 玩家/AI角色尺寸（正方形）
MOVE_SPEED = 5  # 水平移动速度（像素/帧）
JUMP_FORCE = -15  # 跳跃初速度（y轴向下为正，负号代表向上）
GRAVITY = 0.8  # 重力加速度

# 颜色定义（纯色矩形，无外部素材）
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)  # 陷阱尖刺
GREEN = (0, 255, 0)  # 平台
YELLOW = (255, 255, 0)  # 金币
BLUE = (0, 0, 255)  # 玩家角色
PURPLE = (128, 0, 128)  # AI角色
ORANGE = (255, 165, 0)  # 终点旗帜

# 状态/动作空间固定定义
STATE_DIM = 30
ACTION_DIM = 5
ACTION_NAMES = {
    0: "LEFT",
    1: "RIGHT",
    2: "JUMP",
    3: "LEFT_JUMP",
    4: "RIGHT_JUMP",
}

# 训练超参数（固定）
SEED = 42
LEARNING_RATE = 1e-3
GAMMA = 0.99
BATCH_SIZE = 64
REPLAY_BUFFER_SIZE = 50000
MIN_REPLAY_SIZE = 1000
TARGET_UPDATE_STEPS = 500
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995
MAX_EPISODES = 300
MAX_STEPS_PER_EPISODE = 1200

# 文件路径
MODEL_PATH = "checkpoints/dqn_platformer.pth"
PLOT_PATH = "outputs/training_curve.png"
