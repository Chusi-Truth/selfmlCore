import pygame
import sys
import time

class GameVisualizer:
    """游戏可视化类"""
    
    def __init__(self, env, cell_size=40, fps=5):
        self.env = env
        self.cell_size = cell_size
        self.fps = fps
        self.width = env.width * cell_size
        self.height = env.height * cell_size
        
        # 颜色定义
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.GREEN = (0, 255, 0)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.GRAY = (200, 200, 200)
        
        # 初始化pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('贪吃蛇')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        
        # 状态
        self.paused = False
        
    def draw(self):
        """绘制游戏状态"""
        self.screen.fill(self.WHITE)
        
        # 绘制网格线
        for x in range(0, self.width, self.cell_size):
            pygame.draw.line(self.screen, self.GRAY, (x, 0), (x, self.height))
        for y in range(0, self.height, self.cell_size):
            pygame.draw.line(self.screen, self.GRAY, (0, y), (self.width, y))
        
        # 绘制棋盘内容
        for i in range(self.env.height):
            for j in range(self.env.width):
                rect = pygame.Rect(
                    j * self.cell_size, 
                    i * self.cell_size, 
                    self.cell_size, 
                    self.cell_size
                )
                
                if self.env.board[i, j] == self.env.SNAKE_HEAD:
                    pygame.draw.rect(self.screen, self.BLUE, rect)
                elif self.env.board[i, j] == self.env.SNAKE_BODY:
                    pygame.draw.rect(self.screen, self.GREEN, rect)
                elif self.env.board[i, j] == self.env.FOOD:
                    pygame.draw.rect(self.screen, self.RED, rect)
        
        # 显示分数
        score_text = self.font.render(f'Score: {self.env.score}', True, self.BLACK)
        self.screen.blit(score_text, (10, 10))
        
        # 显示游戏状态
        if self.env.game_over:
            game_over_text = self.font.render('Game Over!', True, self.RED)
            self.screen.blit(game_over_text, (self.width//2 - 80, self.height//2 - 18))
        
        if self.paused:
            pause_text = self.font.render('Paused', True, self.BLACK)
            self.screen.blit(pause_text, (self.width//2 - 50, self.height//2 + 30))
        
        pygame.display.flip()
    
    def handle_human_input(self):
        """处理人类玩家的键盘输入"""
        action = None
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action = self.env.UP
                elif event.key == pygame.K_RIGHT:
                    action = self.env.RIGHT
                elif event.key == pygame.K_DOWN:
                    action = self.env.DOWN
                elif event.key == pygame.K_LEFT:
                    action = self.env.LEFT
                elif event.key == pygame.K_p:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self.env.reset()
                    
        return action
    
    def run_human_player(self):
        """运行人类玩家的游戏循环"""
        self.env.reset()
        
        while True:
            action = self.handle_human_input()
            
            if not self.paused and action is not None:
                _, reward, done, info = self.env.step(action)
                
                if done:
                    time.sleep(2)
                    self.env.reset()
            
            self.draw()
            self.clock.tick(self.fps)
            
    def run_ai_player(self, agent, render_every=1, train_mode=True):
        """运行AI玩家的游戏循环
        
        Args:
            agent: 强化学习智能体
            render_every: 每隔多少步渲染一次
            train_mode: 是否处于训练模式
        """
        self.env.reset()
        episode = 0
        step_counter = 0
        
        while True:
            # 处理退出事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_p:
                        self.paused = not self.paused
            
            if self.paused:
                self.draw()
                self.clock.tick(self.fps)
                continue
            
            # 获取当前状态
            state = self.env.get_state()
            
            # 智能体选择动作
            action = agent.get_action(state, train_mode)
            
            # 执行动作
            next_state, reward, done, info = self.env.step(action)
            
            # 训练智能体
            if train_mode:
                agent.train(state, action, reward, next_state, done)
            
            step_counter += 1
            
            # 渲染
            if step_counter % render_every == 0:
                self.draw()
                self.clock.tick(self.fps if not train_mode else 60)  # 训练模式下更快
            
            if done:
                episode += 1
                if train_mode:
                    print(f"Episode: {episode}, Score: {info['score']}")
                self.draw()
                time.sleep(1)
                self.env.reset()