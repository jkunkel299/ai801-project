from dnb_env import DotsAndBoxesEnv
from rl_agent import DQNAgent
# will also import the Markov Decision AI Agent
import pandas as pd

# Initialize environment
game = DotsAndBoxesEnv(visualize=True)
# Initialize agent
rl_agent = DQNAgent(game)
rl_agent.load_agent()

scores, players, winner = game.play_game(player1 = rl_agent)

#TODO
# Anaylitics as the agent is trained, using pandas
''' The idea here is to repeat games x times (x=10?) to collect data on how 
    the RL agent performs. Then load an agent that has undergone more training,
    and repeat to compare performance as learning increases. Right now it's 
    playing against a random player, but this process would be repeated for the 
    Monte Carlo Tree Search AI agent, and even a human player'''
# results = {'Player 1': players[0],
#            'Player 2': players[1],
#            'Player 1 score': scores[0],
#            'Player 2 score': scores[1],
#            'Winner': winner}

# df = pd.DataFrame(results)
# df.to_csv('results.csv', index=False)
