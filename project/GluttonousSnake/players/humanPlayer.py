class HumanPlayer:
    """人类玩家类"""
    
    def __init__(self, env):
        self.env = env
    
    def play(self, visualizer):
        """开始游戏"""
        visualizer.run_human_player()