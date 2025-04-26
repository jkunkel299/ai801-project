from dnb_env import DotsAndBoxesEnv
from rl_agent import DQNAgent
from mcts_agent import MCTSAgent
import pandas as pd

''' The idea here is to repeat games x times (x=10) to collect data on how 
    the RL agent performs. Then load an agent that has undergone more training,
    and repeat to compare performance as learning increases. 
    Comparisons are made between the initial RL model (self-trained) and MCTS,
    the improved RL model (trained against MCTS) and MCTS, and an RL model that
    is trained against MCTS using a modified reward structure.'''

# Initialize environment
game = DotsAndBoxesEnv(visualize=False)

# Initialize RL agents with increasing levels of training
rl_agent_10 = DQNAgent(game)
rl_agent_10.load_agent(path="agent_checkpoint_10_episodes.pth")
rl_agent_100 = DQNAgent(game)
rl_agent_100.load_agent(path="agent_checkpoint_100_episodes.pth")
rl_agent_500 = DQNAgent(game)
rl_agent_500.load_agent(path="agent_checkpoint_500_episodes.pth")
rl_agent_1000 = DQNAgent(game)
rl_agent_1000.load_agent(path="agent_checkpoint_1000_episodes.pth")

# Initialize RL agents that were trained against MCTS
rl_agent_10_mcts = DQNAgent(game)
rl_agent_10_mcts.load_agent(path="agent_checkpoint_10_mcts_episodes.pth")
rl_agent_100_mcts = DQNAgent(game)
rl_agent_100_mcts.load_agent(path="agent_checkpoint_100_mcts_episodes.pth")

# Initialize RL agents that were trained with improved reward structure
rl_agent_10_rewards = DQNAgent(game)
rl_agent_10_rewards.load_agent(path="agent_checkpoint_10_rewards_episodes.pth")
rl_agent_100_rewards = DQNAgent(game)
rl_agent_100_rewards.load_agent(path="agent_checkpoint_100_rewards_episodes.pth")

# Initialize MCTS agent
mcts_agent = MCTSAgent(env=game, simulations=10)

player1_arr = []
player2_arr = []
player1_score_arr = []
player2_score_arr = []
winner_arr = []

epsilons = {
    "RL_10": rl_agent_10.epsilon,
    "RL_100": rl_agent_100.epsilon,
    "RL_500": rl_agent_500.epsilon,
    "RL_1000": rl_agent_1000.epsilon,
    "RL_10_MCTS": rl_agent_10_mcts.epsilon,
    "RL_100_MCTS": rl_agent_100_mcts.epsilon,
    "RL_10_Rewards": rl_agent_10_rewards.epsilon,
    "RL_100_Rewards": rl_agent_100_rewards.epsilon
}

# RL Agents vs MCTS, RL as Player 1
for i in range (1, 51):
    scores, players, winner = game.play_game(player1 = rl_agent_10, 
                                             player2 = mcts_agent)
    player1_arr.append('RL Agent, 10')
    player2_arr.append('MCTS')
    player1_score_arr.append(scores[0])
    player2_score_arr.append(scores[1])
    winner_arr.append(winner)
    
    game.reset()

    scores, players, winner = game.play_game(player1 = rl_agent_100, 
                                             player2 = mcts_agent)
    player1_arr.append('RL Agent, 100')
    player2_arr.append('MCTS')
    player1_score_arr.append(scores[0])
    player2_score_arr.append(scores[1])
    winner_arr.append(winner)

    game.reset()

    scores, players, winner = game.play_game(player1 = rl_agent_500, 
                                             player2 = mcts_agent)
    player1_arr.append('RL Agent, 500')
    player2_arr.append('MCTS')
    player1_score_arr.append(scores[0])
    player2_score_arr.append(scores[1])
    winner_arr.append(winner)

    game.reset()

    scores, players, winner = game.play_game(player1 = rl_agent_1000, 
                                             player2 = mcts_agent)
    player1_arr.append('RL Agent, 1000')
    player2_arr.append('MCTS')
    player1_score_arr.append(scores[0])
    player2_score_arr.append(scores[1])
    winner_arr.append(winner)

    game.reset()

# RL Agents vs MCTS, RL as Player 2
for i in range (1, 51):
    scores, players, winner = game.play_game(player1 = mcts_agent, 
                                             player2 = rl_agent_10)
    player1_arr.append('MCTS')
    player2_arr.append('RL Agent, 10')
    player1_score_arr.append(scores[0])
    player2_score_arr.append(scores[1])
    winner_arr.append(winner)

    game.reset()

    scores, players, winner = game.play_game(player1 = mcts_agent, 
                                             player2 = rl_agent_100)
    player1_arr.append('MCTS')
    player2_arr.append('RL Agent, 100')
    player1_score_arr.append(scores[0])
    player2_score_arr.append(scores[1])
    winner_arr.append(winner)

    game.reset()

    scores, players, winner = game.play_game(player1 = mcts_agent, 
                                             player2 = rl_agent_500)
    player1_arr.append('MCTS')
    player2_arr.append('RL Agent, 500')
    player1_score_arr.append(scores[0])
    player2_score_arr.append(scores[1])
    winner_arr.append(winner)

    game.reset()

    scores, players, winner = game.play_game(player1 = mcts_agent, 
                                             player2 = rl_agent_1000)
    player1_arr.append('MCTS')
    player2_arr.append('RL Agent, 1000')
    player1_score_arr.append(scores[0])
    player2_score_arr.append(scores[1])
    winner_arr.append(winner)

    game.reset()

# Improved RL agent vs MCTS, RL as player 1
for i in range (1, 51):
    scores, players, winner = game.play_game(player1 = rl_agent_10_mcts, 
                                             player2 = mcts_agent)
    player1_arr.append('RL Agent 2.0, 10')
    player2_arr.append('MCTS')
    player1_score_arr.append(scores[0])
    player2_score_arr.append(scores[1])
    winner_arr.append(winner)

    game.reset()

    scores, players, winner = game.play_game(player1 = rl_agent_100_mcts, 
                                             player2 = mcts_agent)
    player1_arr.append('RL Agent 2.0, 100')
    player2_arr.append('MCTS')
    player1_score_arr.append(scores[0])
    player2_score_arr.append(scores[1])
    winner_arr.append(winner)

    game.reset()

# Improved RL agent vs MCTS, RL as player 2
for i in range (1, 51):

    scores, players, winner = game.play_game(player1 = mcts_agent, 
                                             player2 = rl_agent_10_mcts)
    player1_arr.append('MCTS')
    player2_arr.append('RL Agent 2.0, 10')
    player1_score_arr.append(scores[0])
    player2_score_arr.append(scores[1])
    winner_arr.append(winner)

    game.reset()
    scores, players, winner = game.play_game(player1 = mcts_agent, 
                                             player2 = rl_agent_100_mcts)
    player1_arr.append('MCTS')
    player2_arr.append('RL Agent 2.0, 100')
    player1_score_arr.append(scores[0])
    player2_score_arr.append(scores[1]) 
    winner_arr.append(winner)

    game.reset()

# Improved RL agent vs MCTS, RL as player 1
for i in range (1, 51):
    scores, players, winner = game.play_game(player1 = rl_agent_10_rewards, 
                                             player2 = mcts_agent)
    player1_arr.append('RL Agent rewards, 10')
    player2_arr.append('MCTS')
    player1_score_arr.append(scores[0])
    player2_score_arr.append(scores[1])
    winner_arr.append(winner)

    game.reset()

    scores, players, winner = game.play_game(player1 = rl_agent_100_rewards, 
                                             player2 = mcts_agent)
    player1_arr.append('RL Agent rewards, 100')
    player2_arr.append('MCTS')
    player1_score_arr.append(scores[0])
    player2_score_arr.append(scores[1])
    winner_arr.append(winner)

    game.reset()

# Improved RL agent vs MCTS, RL as player 2
for i in range (1, 51):

    scores, players, winner = game.play_game(player1 = mcts_agent, 
                                             player2 = rl_agent_10_rewards)
    player1_arr.append('MCTS')
    player2_arr.append('RL Agent rewards, 10')
    player1_score_arr.append(scores[0])
    player2_score_arr.append(scores[1])
    winner_arr.append(winner)

    game.reset()
    scores, players, winner = game.play_game(player1 = mcts_agent, 
                                             player2 = rl_agent_100_rewards)
    player1_arr.append('MCTS')
    player2_arr.append('RL Agent rewards, 100')
    player1_score_arr.append(scores[0])
    player2_score_arr.append(scores[1]) 
    winner_arr.append(winner)

    game.reset()

# Anaylitics as the agent is trained, using pandas
results = {'Player 1': player1_arr,
           'Player 2': player2_arr,
           'Player 1 score': player1_score_arr,
           'Player 2 score': player2_score_arr,
           'Winner': winner_arr}

df = pd.DataFrame(results)
df.to_csv('results.csv', index=False)

ep_df = pd.DataFrame(epsilons, index=[0])
ep_df.to_csv('epsilons.csv', index=False)