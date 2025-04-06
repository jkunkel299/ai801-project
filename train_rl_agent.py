from rl_agent import DQNAgent
from dnb_env import DotsAndBoxesEnv

# Initialize environment
local_env = DotsAndBoxesEnv(visualize=False)
# Initialize agent
agent = DQNAgent(local_env)
# train agent
agent.train_agent(num_episodes=10)
agent.save_agent("agent_checkpoint_10_episodes.pth")
print(f"Epsilon after 10 episodes = {agent.epsilon}")

# agent.load_agent()
# agent.train_agent(num_episodes=10)
# print(f"Epsilon after loading agent, running 10 additional episodes = {agent.epsilon}")
# agent.save_agent()