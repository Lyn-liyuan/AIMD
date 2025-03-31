
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
from game_objects import Spaceship, Bullet, Asteroid
import os
import pygame

# 检测可用设备（新增代码）#####
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 环境参数调整
WIDTH, HEIGHT = 800, 600
STATE_DIM = 18
ACTION_DIM = 6
BATCH_SIZE = 128
MEMORY_CAPACITY = 20000
LEARNING_RATE = 0.00025
GAMMA = 0.99
EPSILON_DECAY = 0.999
MIN_EPSILON = 0.01

# 游戏环境封装类
class AsteroidEnv:
    def __init__(self, render=False):
        self.render_mode = render
        if self.render_mode:
           self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        else:
           os.environ["SDL_VIDEODRIVER"] = "dummy"
           self.screen = None
        self.reset()

    def reset(self):
        # Pygame初始化（只在需要渲染时执行）
        if self.render_mode and not pygame.get_init():
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("AI太空陨石大战")
        
        self.ship = Spaceship()
        self.bullets = []
        self.asteroids = []
        self.score = 0
        self.game_over = False
        self.step_counter = 0
        return self._get_state()

    def _get_state(self):
        """从环境提取状态特征"""
        # 基础状态：飞船位置、子弹信息（最近3发）
        state = [
            self.ship.x / WIDTH,  # 归一化坐标
            self.ship.y / HEIGHT,
            (self.ship.x - WIDTH//2) / (WIDTH//2),  # 相对屏幕中心
            (self.ship.y - HEIGHT//2) / (HEIGHT//2)
        ]
        
        # 最近的3颗陨石信息（位置+速度+大小）
        asteroids_sorted = sorted(self.asteroids, 
                                key=lambda a: a.y, 
                                reverse=True)[:3]  # 取最接近的3颗
        
        for asteroid in asteroids_sorted:
            state += [
                asteroid.x / WIDTH,
                asteroid.y / HEIGHT,
                asteroid.speed / 5,  # 归一化速度
                asteroid.size / 100  # 归一化大小
            ]
        
        # 填充不足的陨石信息
        while len(state) < STATE_DIM:
            state += [0.0] * 4  # 格式: [x, y, speed, size]

        # 添加子弹信息（最近3发）
        for bullet in self.bullets[:3]:
            state += [bullet.x/WIDTH, bullet.y/HEIGHT]
        while len(state) < STATE_DIM - 8:  # 剩余空间不足时截断
            state += [0.0] * (STATE_DIM - len(state))
        
        return np.array(state[:STATE_DIM], dtype=np.float32)

    def step(self, action):
        """执行动作并返回新状态/奖励/终止标志"""
        self._take_action(action)
        collision_count = self._update_game()
        
        reward = self._calculate_reward(collision_count)
        done = self.game_over
        next_state = self._get_state()
        
        return next_state, reward, done, {}

    def _take_action(self, action):
        """将离散动作转换为控制指令"""
        # 动作解码：0-无操作 1-左 2-右 3-上 4-下 5-开火
        dx = 0
        dy = 0
        fire = False
        
        if action == 1 :
            dx = -1
        elif action == 2 :
            dx = 1
        elif action == 3 :
            dy = -1
        elif action == 4  :
            dy = 1
        elif action == 5 :
            fire = True
        
        # 执行开火
        if fire:
            self.bullets.append(Bullet(self.ship.x, self.ship.y))
        else:
            self.ship.move(dx, dy)

    def _update_game(self):
        """更新游戏逻辑"""
        # 随机生成陨石（降低训练时生成频率）
        if random.random() < 0.02:
            self.asteroids.append(Asteroid())
            
        collision_count = 0

        # 子弹更新
        for bullet in self.bullets[:]:
            bullet.update()
            if bullet.y < 0:
                self.bullets.remove(bullet)

        # 陨石更新和碰撞检测
        for asteroid in self.asteroids[:]:
            asteroid.update()
            
            # 子弹碰撞检测
            collision = False
            for bullet in self.bullets[:]:
                distance = math.hypot(asteroid.x - bullet.x, asteroid.y - bullet.y)
                if distance < asteroid.size//2:
                    asteroid.health -= 10
                    if asteroid.health <= 0:
                        self.score += asteroid.size
                        collision_count += asteroid.size
                        self.asteroids.remove(asteroid)
                    self.bullets.remove(bullet)
                    collision = True
                    break
            if collision:
                continue
                
            # 飞船碰撞检测
            distance = math.hypot(asteroid.x - self.ship.x, 
                                asteroid.y - self.ship.y)
            if distance < asteroid.size//2 + self.ship.width//2:
                self.game_over = True
                
            # 移除离开屏幕的陨石
            if asteroid.y > HEIGHT + asteroid.size:
                self.asteroids.remove(asteroid)

        self.step_counter += 1
        return collision_count

    def _calculate_reward(self,collision_count):
        """设计奖励函数"""
        reward = collision_count
       
        
        # 接近陨石惩罚（动态调整）
        danger_threshold = 100
        nearest_distance = min(
            [math.hypot(a.x-self.ship.x, a.y-self.ship.y) for a in self.asteroids] 
            + [float('inf')])  # 处理无陨石情况
        
        if nearest_distance < danger_threshold:
            reward -= (1 - nearest_distance/danger_threshold) * 0.5
            
        # 碰撞惩罚
        if self.game_over:
            reward -= 50
            
        return reward

    def render(self):
        """可视化当前状态"""
        if not self.render_mode:
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
               running = False
            
        self.screen.fill((0, 0, 0))
        
        # 绘制所有对象
        self.ship.draw(self.screen)
        for bullet in self.bullets:
            bullet.draw(self.screen)
        for asteroid in self.asteroids:
            asteroid.draw(self.screen)
            
        # 显示得分
        font = pygame.font.SysFont(None, 36)
        score_text = font.render(f"Score: {self.score}", True, (255,255,255))
        self.screen.blit(score_text, (10, 10))
        
        pygame.display.flip()
        pygame.time.delay(30)


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, ACTION_DIM)
        )
        self.to(device)  # 初始化时移动到设备 #######

    def forward(self, x):
        # 确保输入数据在GPU #######
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        return self.net(x.to(device))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """存储经验元组 (state, action, reward, next_state, done)"""
        self.buffer.append((
            state.astype(np.float32),  # 确保状态存储为float32
            np.int64(action),          # 动作存储为int64
            float(reward),             # 奖励强制转为浮点数
            next_state.astype(np.float32),
            float(done)                # done转为浮点数用于后续计算
        ))
    
    def sample(self, batch_size):
        """采样并返回批量的张量数据"""
        if len(self.buffer) < batch_size:
            return None
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        samples = [self.buffer[i] for i in indices]
        
        # 高效拆包方式
        states, actions, rewards, next_states, dones = zip(*samples)
        
        # 转换为PyTorch张量并确保设备正确性
        return (
            torch.stack([torch.as_tensor(s) for s in states]).to(device),
            torch.stack([torch.as_tensor(a) for a in actions]).to(device),
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device),
            torch.stack([torch.as_tensor(ns) for ns in next_states]).to(device),
            torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)  # 保持维度一致
        )
    
    def __len__(self):
        """返回当前缓存容量"""
        return len(self.buffer)



class DQNAgent:
    def __init__(self):
        self.policy_net = DQN().to(device)
        self.target_net = DQN().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayBuffer(MEMORY_CAPACITY)
        self.epsilon = 1.0
        
    def select_action(self, state):
        if random.random() > self.epsilon:
            with torch.no_grad():
                # 输入数据转换到GPU #######
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                return self.policy_net(state_tensor).argmax().item()
        else:
            return random.randint(0, ACTION_DIM-1)
    
    def update(self):
        """更新神经网络（错误修复版）"""
        # === 新增空值判断 ===
        batch = self.memory.sample(BATCH_SIZE)
        if batch is None:
            print("Warning: Buffer samples not enough, skip update.")
            return  # 直接跳过此次更新
        states, actions, rewards, next_states, dones = batch
        
        # Q值计算
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (1 - dones) * GAMMA * max_next_q
        
        # 损失计算
        loss = nn.MSELoss()(current_q, target_q)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        
        self.optimizer.step()
        
        self.epsilon = max(MIN_EPSILON, self.epsilon * EPSILON_DECAY)
        
        # 软更新目标网络 #######
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*0.005 + target_net_state_dict[key]*0.995
        self.target_net.load_state_dict(target_net_state_dict)
        
        return loss.item()

def train_dqn(episodes=500):
    
    env = AsteroidEnv(render=True)
    agent = DQNAgent()
    rewards_history = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            env.render()
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.memory.push(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state
            agent.update()
            
        rewards_history.append(total_reward)
        # if episode % 50 == 0 or 
        if True or episode == episodes-1:
            avg_reward = np.mean(rewards_history[-50:]) if episode >=50 else np.mean(rewards_history)
            print(f"Episode: {episode} | Avg Reward: {avg_reward:.2f} | Epsilon: {agent.epsilon:.3f}")

    torch.save(agent.policy_net.state_dict(), "space_dqn_gpu.pth")  # 更新保存文件名 #######
    plt.plot(rewards_history)
    plt.title("Training Progress (GPU)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.show()

def test_model():
    env = AsteroidEnv(render=True)
    policy_net = DQN().to(device)
    # 加载时指定设备 #######
    policy_net.load_state_dict(torch.load("space_dqn_gpu.pth", map_location=device))
    policy_net.eval()
    
    state = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        env.render()
        with torch.no_grad():
            # 输入数据转换到GPU #######
            state_tensor = torch.FloatTensor(state).to(device)
            action = policy_net(state_tensor).argmax().item()
        state, reward, done, _ = env.step(action)
        total_reward += reward
        
    print(f"Test Total Reward: {total_reward}")

if __name__ == "__main__":
    # 优先使用GPU训练 #######
   
    train_dqn() 
    #test_model()
