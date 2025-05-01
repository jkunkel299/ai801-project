# Group 9 - Dots and Boxes game with AI players
### Jessica Kunkel, Manish Kumar
#### AI-801, Spring 2025

## Running the Game
To run the program, first ensure all necessary packages are installed.

The main game file is play_match.py. For Human vs MCTS gameplay, no changes need to be made to the file. Line 19 can be edited to change which agents play the game: change the parameter value for player1, player2, or both to one of the four agents initialized in the file:
  1. rl_agent_1000
  2. rl_agent_100_rewards
  3. human_player
  4. mcts_agent

## Summary
Dots-and-Boxes is a classic pen-and-paper game in which two players take turns drawing lines between dots in a grid, aiming to maximize the number of boxes they create while minimizing the opponent’s boxes. Dots-and-Boxes is a perfect information, deterministic game in which the state of the game board is fully observable by all players and there is no randomness affecting future states. Although its rules are simple, the Dots-and-Boxes game has a high level of complexity due to the size of its search space (a 5x5 grid composed of a total of 60 potential moves, with many potential states), posing a challenge for AI. 

This project explores how two types of Artificial Intelligence agents can learn and perform effectively in a Dots-and-Boxes game environment. One agent is based on Monte Carlo Tree Search (MCTS) and the other uses Reinforcement Learning (RL) to compete against each other and human players in Dots-and-Boxes. The Dots-and-Boxes game provides a contained yet non-trivial setting for testing decision-making under uncertainty, reward-based learning, and adversarial strategies.

This project aims to showcase how traditional AI algorithms can be effectively applied to interactive, real-time game environments by simulating human-like reasoning and long-term planning through game-playing search and reinforcement learning techniques.
