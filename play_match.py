from dnb_env import DotsAndBoxesEnv
from rl_agent import DQNAgent
from human import Human
from mcts_agent import MCTSAgent

# Initialize environment
game = DotsAndBoxesEnv(visualize=True)

rl_agent_1000 = DQNAgent(env=game)
rl_agent_1000.load_agent(path="agent_checkpoint_1000_episodes.pth")

rl_agent_100_rewards = DQNAgent(game)
rl_agent_100_rewards.load_agent(path="agent_checkpoint_100_rewards_episodes.pth")

human_player = Human(env=game)

mcts_agent = MCTSAgent(env=game, simulations=10)

game.play_game(player1=human_player, player2=rl_agent_100_rewards)