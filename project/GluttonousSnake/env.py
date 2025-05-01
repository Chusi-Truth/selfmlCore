from snake import snake,snakeNode
class env():
    def __init__(self, width=10, height=10):
        self.width = width
        self.height = height
        self.board = [[0 for _ in range(width)] for _ in range(height)]
        self.snake = snake(int(max(self.width/3,3)))
        # init snake position
        for i in range(self.snake.length):
            self.snake.body.append(snakeNode(0, i))
    
    
    def print_board(self):
        for row in self.board:
            print(" ".join(str(cell) for cell in row))
        print()
            
        