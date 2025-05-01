# Direction and action constants
TAWARD_UP=100001
TAWARD_DOWN=100002
TAWARD_LEFT=100003
TAWARD_RIGHT=100004
TURN_UP=100005
TURN_DOWN=100006
TURN_LEFT=100007
TURN_RIGHT=100008
KEEP_GOING=100009

class snakeNode():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.ishead = False

    def __str__(self):
        return f"({self.x}, {self.y})"




class snake():
    def __init__(self, initLength):
        self.length = initLength
        self.body = []
        self.toward=TAWARD_RIGHT
        self.nowDirection=KEEP_GOING
        self.isAlive=True