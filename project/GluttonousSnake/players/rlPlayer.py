import numpy as np
import random
import pickle
import os

class QAgent:
    """基于Q-learning的智能体"""
    
    def __init__(self, env, epsilon=0.1, alpha=0.1, gamma=0.9, load_model=False):
        """
        Args:
            env: 游戏环境
            epsilon: 探索率
            alpha: 学习率
            gamma: 折扣因子
            load_model: 是否加载已有模型
        """
        self.env = env
        self.epsilon = epsilon  # 探索率
        self.alpha = alpha      # 学习率
        self.gamma = gamma      # 折扣因子
        
        # 状态空间大小
        self.state_size = (env.height, env.width, 4)  # 棋盘状态 + 蛇的方向
        
        # 动作空间
        self.action_size = 4  # 上、右、下、左
        
        # 初始化Q表为近似方法
        self.weights = np.zeros((16, 4))  # 16个特征，4个动作
        
        if load_model and os.path.exists('snake_q_agent.pkl'):
            self.load()
    
    def get_features(self, state):
        """从状态提取特征
        
        特征包括:
        1. 食物相对于蛇头的方向 (4个)
        2. 障碍物在蛇头附近的位置 (4个)
        3. 蛇的当前方向 (4个)
        4. 蛇头是否靠近边界 (4个)
        """
        features = np.zeros(16)
        
        head_pos = None
        food_pos = None
        
        # 找到蛇头和食物位置
        for i in range(self.env.height):
            for j in range(self.env.width):
                if state[i, j] == self.env.SNAKE_HEAD:
                    head_pos = (i, j)
                elif state[i, j] == self.env.FOOD:
                    food_pos = (i, j)
        
        if head_pos is None or food_pos is None:
            return features
        
        # 1. 食物相对于蛇头的方向
        if food_pos[0] < head_pos[0]:  # 食物在上方
            features[0] = 1
        if food_pos[1] > head_pos[1]:  # 食物在右侧
            features[1] = 1
        if food_pos[0] > head_pos[0]:  # 食物在下方
            features[2] = 1
        if food_pos[1] < head_pos[1]:  # 食物在左侧
            features[3] = 1
        
        # 2. 障碍物在蛇头附近的位置
        # 上方是障碍物
        if head_pos[0] - 1 < 0 or (head_pos[0] - 1 >= 0 and state[head_pos[0] - 1, head_pos[1]] == self.env.SNAKE_BODY):
            features[4] = 1
        
        # 右侧是障碍物
        if head_pos[1] + 1 >= self.env.width or (head_pos[1] + 1 < self.env.width and state[head_pos[0], head_pos[1] + 1] == self.env.SNAKE_BODY):
            features[5] = 1
        
        # 下方是障碍物
        if head_pos[0] + 1 >= self.env.height or (head_pos[0] + 1 < self.env.height and state[head_pos[0] + 1, head_pos[1]] == self.env.SNAKE_BODY):
            features[6] = 1
        
        # 左侧是障碍物
        if head_pos[1] - 1 < 0 or (head_pos[1] - 1 >= 0 and state[head_pos[0], head_pos[1] - 1] == self.env.SNAKE_BODY):
            features[7] = 1
        
        # 3. 蛇的当前方向
        current_direction = self.env.snake.direction
        features[8 + current_direction] = 1
        
        # 4. 蛇头是否靠近边界
        if head_pos[0] <= 1:  # 靠近上边界
            features[12] = 1
        if head_pos[1] >= self.env.width - 2:  # 靠近右边界
            features[13] = 1
        if head_pos[0] >= self.env.height - 2:  # 靠近下边界
            features[14] = 1
        if head_pos[1] <= 1:  # 靠近左边界
            features[15] = 1
        
        return features
    
    def get_q_value(self, state, action):
        """计算Q值"""
        features = self.get_features(state)
        return np.dot(features, self.weights[:, action])
    
    def get_action(self, state, explore=True):
        """选择动作"""
        if explore and random.random() < self.epsilon:
            return random.choice([0, 1, 2, 3])
        
        q_values = [self.get_q_value(state, a) for a in range(self.action_size)]
        return np.argmax(q_values)
    
    def train(self, state, action, reward, next_state, done):
        """训练智能体"""
        # 计算当前Q值
        current_q = self.get_q_value(state, action)
        
        # 计算下一状态的最大Q值
        next_q = 0
        if not done:
            next_q = max([self.get_q_value(next_state, a) for a in range(self.action_size)])
        
        # 计算目标Q值
        target_q = reward + self.gamma * next_q
        
        # 更新权重
        features = self.get_features(state)
        for i in range(len(features)):
            self.weights[i, action] += self.alpha * features[i] * (target_q - current_q)
    
    def save(self):
        """保存模型"""
        with open('snake_q_agent.pkl', 'wb') as f:
            pickle.dump(self.weights, f)
    
    def load(self):
        """加载模型"""
        with open('snake_q_agent.pkl', 'rb') as f:
            self.weights = pickle.load(f)