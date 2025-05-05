import argparse
import env
import visualize
import players.humanPlayer as human_player
import players.rlPlayer as rl_agent

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='贪吃蛇游戏')
    parser.add_argument('--mode', type=str, default='human', choices=['human', 'ai-play', 'ai-train'],
                      help='游戏模式: human(人类玩家), ai-play(AI演示), ai-train(AI训练)')
    parser.add_argument('--width', type=int, default=10, help='游戏宽度')
    parser.add_argument('--height', type=int, default=10, help='游戏高度')
    parser.add_argument('--fps', type=int, default=5, help='游戏帧率')
    args = parser.parse_args()
    
    # 创建游戏环境
    game = env.Env(args.width, args.height)
    
    # 创建可视化器
    viz = visualize.GameVisualizer(game, fps=args.fps)
    
    if args.mode == 'human':
        # 人类玩家模式
        player = human_player.HumanPlayer(game)
        player.play(viz)
    elif args.mode == 'ai-play':
        # AI演示模式
        agent = rl_agent.QAgent(game, epsilon=0.01, load_model=True)
        viz.run_ai_player(agent, train_mode=False)
    else:  # ai-train
        # AI训练模式
        agent = rl_agent.QAgent(game, epsilon=0.1)
        try:
            viz.run_ai_player(agent, render_every=100, train_mode=True)
        except KeyboardInterrupt:
            print("训练中断，正在保存模型...")
            agent.save()
            print("模型已保存!")

if __name__ == "__main__":
    main()