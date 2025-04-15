from dnb_env import DotsAndBoxesEnv
from rl_agent import DQNAgent
# will also import the Monte Carlo Tree Search AI Agent
import pandas as pd

# Initialize environment
game = DotsAndBoxesEnv(visualize=False)
# Initialize agent
rl_agent_10 = DQNAgent(game)
rl_agent_10.load_agent(path="agent_checkpoint_10_episodes.pth")
rl_agent_100 = DQNAgent(game)
rl_agent_100.load_agent(path="agent_checkpoint_10_episodes.pth")
rl_agent_500 = DQNAgent(game)
rl_agent_500.load_agent(path="agent_checkpoint_10_episodes.pth")
rl_agent_1000 = DQNAgent(game)
rl_agent_1000.load_agent(path="agent_checkpoint_10_episodes.pth")

player1_arr = []
player2_arr = []
player1_score_arr = []
player2_score_arr = []
winner_arr = []

# RL with 10 episodes of training vs Random
for i in range (1, 11):
    scores, players, winner = game.play_game(player1 = rl_agent_10)
    player1_arr.append('RL Agent, 10')
    player2_arr.append('Random')
    player1_score_arr.append(scores[0])
    player2_score_arr.append(scores[1])
    winner_arr.append(winner)
    print(f"in for-loop, i={i}")

for i in range (1, 11):
    scores, players, winner = game.play_game(player2 = rl_agent_10)
    player1_arr.append('Random')
    player2_arr.append('RL Agent, 10')
    player1_score_arr.append(scores[0])
    player2_score_arr.append(scores[1])
    winner_arr.append(winner)
    print(f"in for-loop, i={i}")

# RL with 100 episodes of training vs Random
for i in range (1, 11):
    scores, players, winner = game.play_game(player1 = rl_agent_100)
    player1_arr.append('RL Agent, 100')
    player2_arr.append('Random')
    player1_score_arr.append(scores[0])
    player2_score_arr.append(scores[1])
    winner_arr.append(winner)
    print(f"in for-loop, i={i}")

for i in range (1, 11):
    scores, players, winner = game.play_game(player2 = rl_agent_100)
    player1_arr.append('Random')
    player2_arr.append('RL Agent, 100')
    player1_score_arr.append(scores[0])
    player2_score_arr.append(scores[1])
    winner_arr.append(winner)
    print(f"in for-loop, i={i}")

# RL with 500 episodes of training vs Random
for i in range (1, 11):
    scores, players, winner = game.play_game(player1 = rl_agent_500)
    player1_arr.append('RL Agent, 500')
    player2_arr.append('Random')
    player1_score_arr.append(scores[0])
    player2_score_arr.append(scores[1])
    winner_arr.append(winner)
    print(f"in for-loop, i={i}")

for i in range (1, 11):
    scores, players, winner = game.play_game(player2 = rl_agent_500)
    player1_arr.append('Random')
    player2_arr.append('RL Agent, 500')
    player1_score_arr.append(scores[0])
    player2_score_arr.append(scores[1])
    winner_arr.append(winner)
    print(f"in for-loop, i={i}")

# RL with 1000 episodes of training vs Random
for i in range (1, 11):
    scores, players, winner = game.play_game(player1 = rl_agent_1000)
    player1_arr.append('RL Agent, 1000')
    player2_arr.append('Random')
    player1_score_arr.append(scores[0])
    player2_score_arr.append(scores[1])
    winner_arr.append(winner)
    print(f"in for-loop, i={i}")

for i in range (1, 11):
    scores, players, winner = game.play_game(player2 = rl_agent_1000)
    player1_arr.append('Random')
    player2_arr.append('RL Agent, 1000')
    player1_score_arr.append(scores[0])
    player2_score_arr.append(scores[1])
    winner_arr.append(winner)
    print(f"in for-loop, i={i}")

# Anaylitics as the agent is trained, using pandas
''' The idea here is to repeat games x times (x=10?) to collect data on how 
    the RL agent performs. Then load an agent that has undergone more training,
    and repeat to compare performance as learning increases. Right now it's 
    playing against a random player, but this process would be repeated for the 
    Monte Carlo Tree Search AI agent, and even a human player'''
results = {'Player 1': player1_arr,
           'Player 2': player2_arr,
           'Player 1 score': player1_score_arr,
           'Player 2 score': player2_score_arr,
           'Winner': winner_arr}

df = pd.DataFrame(results)

df.to_csv('results.csv', index=False)



'''filename format: results_player1_player2
    if either player is the RL agent: RLepisodes
    if either player is the Monte Carlo Tree Search agent: MCTS
    if either player is random: rand
    i.e., If player 1 is RL agent with 100 training episodes and player 2 is
    random, filename = results_RL100_rand.csv'''