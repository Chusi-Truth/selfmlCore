class SnakeNode:
    """蛇的一个节点"""
    def __init__(self, x, y):
        self.x = x  # 行
        self.y = y  # 列

class Snake:
    """蛇类"""
    def __init__(self, initial_length=3):
        self.length = initial_length
        self.body = []  # 蛇身，第一个元素是头部
        self.direction = 1  # 0: 上, 1: 右, 2: 下, 3: 左