from rl_agent import DQNAgent
from dnb_env import DotsAndBoxesEnv

# Initialize environment
local_env = DotsAndBoxesEnv(visualize=False)
# Initialize agent
agent_10 = DQNAgent(local_env)
agent_100 = DQNAgent(local_env)
agent_500 = DQNAgent(local_env)
agent_1000 = DQNAgent(local_env)

# train agent
# agent_10.train_agent(num_episodes=10)
# agent_10.save_agent("agent_checkpoint_10_episodes.pth")
# agent_10.load_agent("agent_checkpoint_10_episodes.pth")
# print(f"Epsilon after 10 episodes = {agent_10.epsilon}")

agent_100.train_agent(num_episodes=100)
agent_100.save_agent("agent_checkpoint_100_episodes.pth")
print(f"Epsilon after 100 episodes = {agent_100.epsilon}")

agent_500.train_agent(num_episodes=500)
agent_500.save_agent("agent_checkpoint_500_episodes.pth")
print(f"Epsilon after 500 episodes = {agent_500.epsilon}")

agent_1000.train_agent(num_episodes=1000)
agent_1000.save_agent("agent_checkpoint_1000_episodes.pth")
print(f"Epsilon after 1000 episodes = {agent_1000.epsilon}")
