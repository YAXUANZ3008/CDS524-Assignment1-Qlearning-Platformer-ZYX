"""关卡基础对象定义：GameObject及其子类。"""

from __future__ import annotations

import pygame

from config import (
    FPS,
    GREEN,
    ORANGE,
    RED,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    WHITE,
    YELLOW,
)


class GameObject:
    """所有游戏元素的父类。

    该基类只维护两个最核心属性：
    1. rect：使用pygame.Rect表示对象的碰撞盒与绘制区域。
    2. color：使用RGB三元组表示对象的填充颜色。

    子类通过继承该类即可复用基础绘制行为，避免重复代码。
    """

    def __init__(self, rect: pygame.Rect, color: tuple[int, int, int]) -> None:
        """初始化游戏对象。

        Args:
            rect: 对象矩形碰撞盒。
            color: 对象绘制颜色。
        """
        # 保存碰撞盒，后续碰撞检测与绘制都基于该Rect。
        self.rect = rect
        # 保存颜色，用于draw方法填充矩形。
        self.color = color

    def draw(self, screen: pygame.Surface) -> None:
        """在窗口上绘制纯色矩形。

        Args:
            screen: 当前Pygame窗口表面。
        """
        # 使用pygame原生矩形绘制函数渲染对象。
        pygame.draw.rect(screen, self.color, self.rect)


class Platform(GameObject):
    """平台对象：角色可站立的碰撞体。"""

    def __init__(self, rect: pygame.Rect) -> None:
        """创建平台对象，平台统一使用绿色。"""
        # 直接调用父类构造函数初始化基础属性。
        super().__init__(rect=rect, color=GREEN)


class Spike(GameObject):
    """陷阱尖刺对象：角色触碰后直接死亡。"""

    def __init__(self, rect: pygame.Rect) -> None:
        """创建尖刺对象，尖刺统一使用红色。"""
        # 直接调用父类构造函数初始化基础属性。
        super().__init__(rect=rect, color=RED)


class Coin(GameObject):
    """金币对象：角色触碰后收集并增加分数。"""

    def __init__(self, rect: pygame.Rect) -> None:
        """创建金币对象，金币统一使用黄色。"""
        # 直接调用父类构造函数初始化基础属性。
        super().__init__(rect=rect, color=YELLOW)


class Goal(GameObject):
    """终点旗帜对象：角色触碰后闯关成功。"""

    def __init__(self, rect: pygame.Rect) -> None:
        """创建终点对象，终点统一使用橙色。"""
        # 直接调用父类构造函数初始化基础属性。
        super().__init__(rect=rect, color=ORANGE)


class Level(GameObject):
    """关卡管理类：管理固定单关卡内的所有元素。

    该类在初始化时构建固定布局：
    - 至少5个阶梯式平台（并包含完整地面）
    - 中间区域3个陷阱，间隔分布
    - 3个金币分布在平台上
    - 终点旗帜位于最右侧最高平台上

    说明：
    Level继承GameObject以满足“仅继承基类”的要求，但其主要职责是聚合管理。
    """

    def __init__(self) -> None:
        """初始化固定关卡元素。"""
        # 为满足继承关系，给Level设置覆盖全屏的管理Rect与绿色默认色。
        super().__init__(
            rect=pygame.Rect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT),
            color=GREEN,
        )

        # 创建平台列表：包含完整地面 + 5个阶梯平台。
        self.platforms = [
            Platform(pygame.Rect(0, SCREEN_HEIGHT - 40, SCREEN_WIDTH, 40)),
            Platform(pygame.Rect(90, 320, 120, 16)),
            Platform(pygame.Rect(240, 270, 120, 16)),
            Platform(pygame.Rect(390, 220, 120, 16)),
            Platform(pygame.Rect(540, 170, 120, 16)),
            Platform(pygame.Rect(680, 120, 100, 16)),
        ]

        # 创建陷阱列表：3个陷阱在地面中间区域间隔排布。
        self.spikes = [
            Spike(pygame.Rect(200, SCREEN_HEIGHT - 50, 40, 10)),
            Spike(pygame.Rect(380, SCREEN_HEIGHT - 50, 40, 10)),
            Spike(pygame.Rect(560, SCREEN_HEIGHT - 50, 40, 10)),
        ]

        # 创建金币列表：共3个，放置在中高平台上方。
        self.coins = [
            Coin(pygame.Rect(285, 245, 14, 14)),
            Coin(pygame.Rect(435, 195, 14, 14)),
            Coin(pygame.Rect(585, 145, 14, 14)),
        ]

        # 在最右侧最高平台上放置终点旗帜。
        self.goal = Goal(pygame.Rect(740, 65, 20, 55))

    def draw(self, screen: pygame.Surface) -> None:
        """绘制关卡中的所有元素。

        Args:
            screen: 当前Pygame窗口表面。
        """
        # 先绘制所有平台，作为关卡基础结构。
        for platform in self.platforms:
            platform.draw(screen)

        # 再绘制所有陷阱。
        for spike in self.spikes:
            spike.draw(screen)

        # 绘制所有金币。
        for coin in self.coins:
            coin.draw(screen)

        # 最后绘制终点旗帜，确保可见性。
        self.goal.draw(screen)


if __name__ == "__main__":
    # 初始化Pygame，确保该文件可独立运行预览关卡。
    pygame.init()

    # 创建固定尺寸窗口，严格匹配项目全局常量。
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Level Objects Preview")
    clock = pygame.time.Clock()

    # 构建关卡实例。
    level = Level()

    # 主循环：仅用于展示关卡元素绘制效果。
    running = True
    while running:
        # 处理窗口关闭事件。
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # 每帧先清屏，避免残影。
        screen.fill(WHITE)

        # 绘制关卡全部元素。
        level.draw(screen)

        # 刷新显示并控制帧率。
        pygame.display.flip()
        clock.tick(FPS)

    # 退出前释放Pygame资源。
    pygame.quit()
