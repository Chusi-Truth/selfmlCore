import numpy as np
import random
from snake import Snake, SnakeNode

class Env:
    """贪吃蛇游戏环境"""
    
    # 动作空间
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    
    # 单元格状态
    EMPTY = 0
    FOOD = 1
    SNAKE_BODY = 2
    SNAKE_HEAD = 3
    
    def __init__(self, width=10, height=10):
        self.width = width
        self.height = height
        self.board = np.zeros((height, width), dtype=int)
        self.snake = Snake(initial_length=3)
        self.score = 0
        self.steps = 0
        self.max_steps = 100  # 防止游戏无限进行
        self.food_pos = None
        self.game_over = False
        
        # 初始化蛇的位置
        self._init_snake()
        
        # 初始生成食物
        self.create_food()
    
    def _init_snake(self):
        """初始化蛇的位置在左上方"""
        start_x = self.width // 4
        start_y = self.height // 4
        
        # 清除蛇的身体
        self.snake.body = []
        
        # 添加头部和身体
        self.snake.body.append(SnakeNode(start_y, start_x))
        self.snake.body.append(SnakeNode(start_y, start_x - 1))
        self.snake.body.append(SnakeNode(start_y, start_x - 2))
        
        self.snake.length = 3
        self.snake.direction = self.RIGHT
        
        # 更新棋盘
        self.update_board()
    
    def update_board(self):
        """更新棋盘状态"""
        # 清空棋盘
        self.board.fill(self.EMPTY)
        
        # 放置蛇身
        for i in range(1, len(self.snake.body)):
            node = self.snake.body[i]
            if 0 <= node.x < self.height and 0 <= node.y < self.width:
                self.board[node.x, node.y] = self.SNAKE_BODY
        
        # 放置蛇头
        head = self.snake.body[0]
        if 0 <= head.x < self.height and 0 <= head.y < self.width:
            self.board[head.x, head.y] = self.SNAKE_HEAD
        
        # 放置食物
        if self.food_pos:
            self.board[self.food_pos[0], self.food_pos[1]] = self.FOOD
    
    def create_food(self):
        """随机生成食物"""
        empty_cells = []
        for i in range(self.height):
            for j in range(self.width):
                if self.board[i, j] == self.EMPTY:
                    empty_cells.append((i, j))
        
        if empty_cells:
            self.food_pos = random.choice(empty_cells)
            self.board[self.food_pos[0], self.food_pos[1]] = self.FOOD
            return True
        return False
    
    def step(self, action):
        """环境步进一步
        Args:
            action: 0 (上), 1 (右), 2 (下), 3 (左)
        Returns:
            next_state: 下一个状态
            reward: 奖励
            done: 是否结束
            info: 附加信息
        """
        if self.game_over:
            return self.board.copy(), 0, True, {"score": self.score}
        
        # 更新蛇的方向 (防止直接反向)
        current_direction = self.snake.direction
        if (action == self.UP and current_direction != self.DOWN) or \
           (action == self.RIGHT and current_direction != self.LEFT) or \
           (action == self.DOWN and current_direction != self.UP) or \
           (action == self.LEFT and current_direction != self.RIGHT):
            self.snake.direction = action
        
        # 移动蛇
        head = self.snake.body[0]
        new_head = SnakeNode(head.x, head.y)
        
        # 根据方向移动
        if self.snake.direction == self.UP:
            new_head.x -= 1
        elif self.snake.direction == self.RIGHT:
            new_head.y += 1
        elif self.snake.direction == self.DOWN:
            new_head.x += 1
        elif self.snake.direction == self.LEFT:
            new_head.y -= 1
        
        # 检查是否撞墙
        if (new_head.x < 0 or new_head.x >= self.height or 
            new_head.y < 0 or new_head.y >= self.width):
            self.game_over = True
            return self.board.copy(), -10, True, {"score": self.score}
        
        # 检查是否撞到自己
        for i in range(len(self.snake.body) - 1):
            if new_head.x == self.snake.body[i].x and new_head.y == self.snake.body[i].y:
                self.game_over = True
                return self.board.copy(), -10, True, {"score": self.score}
        
        # 移动蛇头
        self.snake.body.insert(0, new_head)
        
        # 检查是否吃到食物
        reward = -0.01  # 默认小惩罚，鼓励尽快找到食物
        if self.food_pos and new_head.x == self.food_pos[0] and new_head.y == self.food_pos[1]:
            self.score += 1
            reward = 10.0
            self.snake.length += 1
            self.create_food()
        else:
            # 没吃到食物，移除尾部
            self.snake.body.pop()
        
        # 更新棋盘
        self.update_board()
        
        # 更新步数
        self.steps += 1
        if self.steps >= self.max_steps:
            return self.board.copy(), reward, True, {"score": self.score}
        
        return self.board.copy(), reward, self.game_over, {"score": self.score}
    
    def reset(self):
        """重置环境"""
        self.board.fill(self.EMPTY)
        self.score = 0
        self.steps = 0
        self.game_over = False
        self._init_snake()
        self.create_food()
        return self.board.copy()
    
    def render(self):
        """打印棋盘状态 (文本模式)"""
        symbols = {
            self.EMPTY: '.',
            self.FOOD: 'F',
            self.SNAKE_BODY: 'o',
            self.SNAKE_HEAD: 'O'
        }
        
        for i in range(self.height):
            line = []
            for j in range(self.width):
                line.append(symbols[self.board[i, j]])
            print(''.join(line))
        print(f"Score: {self.score}")
    
    def get_state(self):
        """返回当前状态，用于RL算法"""
        return self.board.copy()