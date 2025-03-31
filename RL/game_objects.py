import random
import math
import pygame

# 游戏窗口设置
WIDTH, HEIGHT = 800, 600

# 颜色定义
WHITE = (255, 255, 255)
SILVER = (192, 192, 192)  # 飞船统一颜色
GRAY = (128, 128, 128)    # 陨石颜色
YELLOW = (255, 255, 0)

# 飞船类
class Spaceship:
    def __init__(self):
        self.x = WIDTH // 2
        self.y = HEIGHT - 100
        self.speed = 5
        self.width = 15  # 放大船体宽度
        self.height = 30  # 放大船体高度
        
    def draw(self, surface):
        # 绘制主机体（大三角形）
        points = [
            (self.x, self.y),
            (self.x - self.width//2, self.y + self.height),
            (self.x + self.width//2, self.y + self.height)
        ]
        pygame.draw.polygon(surface, SILVER, points)
        
        # 绘制左右发动机使用相同颜色
        engine_color = SILVER  # 统一使用银色
        # 左发动机
        left_engine = [
            (self.x - self.width//2 + 5, self.y + self.height//2),
            (self.x - self.width//2 - 10, self.y + self.height),
            (self.x - self.width//2 + 5, self.y + self.height)
        ]
        pygame.draw.polygon(surface, engine_color, left_engine)
        
        # 右发动机
        right_engine = [
            (self.x + self.width//2 - 5, self.y + self.height//2),
            (self.x + self.width//2 + 10, self.y + self.height),
            (self.x + self.width//2 - 5, self.y + self.height)
        ]
        pygame.draw.polygon(surface, engine_color, right_engine)
        
    def move(self, dx, dy):
        self.x = max(self.width, min(WIDTH - self.width, self.x + dx * self.speed))
        self.y = max(HEIGHT//2, min(HEIGHT - self.height, self.y + dy * self.speed))

# 子弹类
class Bullet:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.speed = 10
        self.radius = 3
        
    def update(self):
        self.y -= self.speed
        
    def draw(self, surface):
        pygame.draw.circle(surface, YELLOW, (self.x, self.y), self.radius)

# 陨石类
class Asteroid:
    # 类级别预生成形状模板
    SHAPE_TEMPLATES = [
        # 五边形（规则凸多边形）
        [(0.5, 0.0), (1.0, 0.4), (0.8, 1.0), (0.2, 1.0), (0.0, 0.4)],
        # 六边形（规则凸多边形）
        [(0.2, 0.2), (0.5, 0.0), (0.8, 0.2), (0.8, 0.8), (0.5, 1.0), (0.2, 0.8)],
        # 四边形（凸形）
        [(0.1, 0.3), (0.9, 0.1), (0.7, 0.9), (0.3, 0.7)],
        # 三角形（凸形）
        [(0.1, 0.1), (0.9, 0.1), (0.5, 0.9)],
        # 七边形（凸形）
        [(0.3,0.1),(0.7,0.1),(0.9,0.3),(0.8,0.7),(0.5,0.9),(0.2,0.7),(0.1,0.3)]
    ]

    def __init__(self):
        self.size = random.choice([60, 80, 100])
        self.x = random.randint(self.size, WIDTH - self.size)
        self.y = -self.size * 2
        self.speed = random.randint(2, 5)
        self.health = math.ceil(self.size / 2)
        
        # 从预定义模板中选择形状并缩放
        template = random.choice(Asteroid.SHAPE_TEMPLATES)
        scale = self.size / 2
        self.relative_points = [(x*scale, y*scale) for (x,y) in template]
        
    def update(self):
        self.y += self.speed
        
    def draw(self, surface):
        # 根据预生成的相对坐标计算绝对位置
        points = [(self.x + dx, self.y + dy) for (dx, dy) in self.relative_points]
        pygame.draw.polygon(surface, GRAY, points)
