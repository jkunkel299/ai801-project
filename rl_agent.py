import numpy as np
import random
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt
from dnb_env import DotsAndBoxesEnv
import pickle


class CNN_DQN(nn.Module):
    def __init__(self, input_channels, output_dim):
        super(CNN_DQN, self).__init__()
        # Convolutional layers to extract spatial features of the dots-and-boxes board
        # each convolutional layer applies a 2D convolution over an inpyt image compsed of input planes
        # in this case the image will be the dots-and-boxes board
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        # Fully connected layers to map features to Q-values
        self.fc1 = nn.Linear(128 * 11 * 11, 512) # flatten before passing to dense layer, assuming 5 x 5 board
            # a 5x5 box board is actually and 11x11 board space, since it includes both dots and edges
        self.fc2 = nn.Linear(512, output_dim) # output layer for Q-values

    def forward(self, x):
        # manually unsqueezing x to alleviate runtime errors
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        # ReLU activation applies the rectified linear unit function element-wise
        x = F.relu(self.conv1(x)) # First convolutional layer + ReLU activation
        x = F.relu(self.conv2(x)) # Second convolutional layer + ReLU activation
        x = F.relu(self.conv3(x)) # Third convolutional layer + ReLU activation
        
        x = x.view(x.size(0), -1) # flatten feature maps before feeding into fully connected layer
        x = F.relu(self.fc1(x)) # fully connected layer with ReLU activation
        x = self.fc2(x) # output Q-values for all actions

        return x
    
class DQNAgent:
    def __init__(self, env):
        self.env = env
        input_channels = 1
        output_dim = env.action_space.n

        # visualization
        self.q_value_history = []
        self.reward_history = []

        self.model = CNN_DQN(input_channels, output_dim)
        self.target_model = CNN_DQN(input_channels, output_dim)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.gamma = 0.99

    def get_state(self, board):
        """Convert board state to sensor format for convolutional neural network"""
        return torch.tensor(board[np.newaxis, :, :], dtype=torch.float32) # Shape: (1, height, width)

    def choose_action(self, state):
        """Epsilon-greedy action selection"""
        #state_tensor = torch.tensor(state[np.newaxis, np.newaxis, :, :], dtype=torch.float32)
        state_tensor = self.get_state(self.env.board)
        valid_actions = self.env.get_valid_actions()
        # print("valid actions in choose: ", valid_actions)
        if np.random.rand() < self.epsilon:
            action = random.choice(valid_actions)
        else:      
            # Exploitation: Choose the best action from available actions    
            with torch.no_grad():
                q_values = self.model(state_tensor).squeeze(0) # predict Q-values
                q_values = q_values.cpu().numpy()# Convert to NumPy for indexing

                # Select the best action from valid actions
                valid_q_values = {a: q_values[a] for a in valid_actions} # Filter by available actions
                action = max(valid_q_values, key=valid_q_values.get)# Choose action with highest Q-value
        # print("action: ",action)
        return action
        
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience tuple in memory"""
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        """Train the CNN-DQN using experiences from the replay buffer"""
        if len(self.memory) < self.batch_size:
            self.decay_epsilon()
            return # Wait until enough experience is collected
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.array(states) # convert list to single numpy.ndarray to improve performance
        actions = np.array(actions) # convert list to single numpy.ndarray to improve performance
        rewards = np.array(rewards) # convert list to single numpy.ndarray to improve performance
        next_states = np.array(next_states) # convert list to single numpy.ndarray to improve performance
        dones = np.array(dones) # convert list to single numpy.ndarray to improve performance

        # convert to tensors and reshape for CNN
        states_tensor = torch.tensor(states, dtype=torch.float32).unsqueeze(1)
        actions_tensor = torch.tensor(actions, dtype=torch.float32).unsqueeze(1)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states_tensor = torch.tensor(next_states, dtype=torch.float32)
        dones_tensor = torch.tensor(dones, dtype=torch.float32)

        # get current Q-values
        q_values = self.model(states_tensor).gather(1, actions_tensor.long()).squeeze(1)
        # get target Q-values using the target network
        with torch.no_grad():
            # reshaping next_q_values and dones_tensor to align with the shape of rewards_tensor
            next_q_values = self.target_model(next_states_tensor).max(1)[0].unsqueeze(1)
            dones_tensor = dones_tensor.unsqueeze(1)
            target_q_values = rewards_tensor + (self.gamma * next_q_values * (1 - dones_tensor))
        
        # compute loss and backpropagate
        loss = F.mse_loss(q_values.unsqueeze(1), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # epsilon decay for exploration-exploitation balance
        self.decay_epsilon()

    def update_target_network(self):
        """Update the target network weights to match the main model"""
        self.target_model.load_state_dict(self.model.state_dict())

    def decay_epsilon(self):
        """Decay the epsilon value for epsilon-greedy action selection"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train_agent(self, num_episodes=1000):
        """Train the convolutional neural network DQN agent on the Dots and Boxes game"""
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            valid_actions = True
            total_q = 0
            moves = 0
            total_reward = 0

            while not done:
                valid_actions = self.env.get_valid_actions()
                # print(valid_actions)
                if not valid_actions:
                    done = True
                    break
                action = self.choose_action(state)
                next_state, reward, _, done = self.env.step(action)

                total_reward += reward

                q_values = self.model(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
                total_q += q_values.max().item()
                moves += 1

                self.store_experience(state, action, reward, next_state, done)
                self.train()

                state = next_state
                # print("moves: ", moves)

            self.reward_history.append(total_reward)

            avg_q_val = total_q / moves if moves > 0 else 0
            self.q_value_history.append(avg_q_val)
            # update the target network periodically
            if episode % 10 == 0:
                self.update_target_network()

            # Decay epsilon for better exploitation over time
            self.decay_epsilon()
            if episode % 100 == 0:
                print(f"Episode {episode+1}: Epsilon = {self.epsilon:.4f}")
            print("Episode: ", episode+1)
        print(f"Episode {episode+1}: Epsilon = {self.epsilon:.4f}")
        #self.plot_q_values()
        # self.plot_rewards()

    # def plot_rewards(self):
    #     plt.plot(self.reward_history)
    #     plt.xlabel("Episode")
    #     plt.ylabel("Total Reward")
    #     plt.title("Training Reward Progression")
    #     plt.show()

    def plot_q_values(self):
        plt.plot(self.q_value_history)
        plt.xlabel("Episode")
        plt.ylabel("Average Q-Value")
        plt.title("Q-Value Progression During Training")
        plt.show()

    def save_agent(self, path="agent_checkpoint.pth"):
        """Save entire agent: model, target model, memory, and epsilon
            For use across instances of gameplay"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_state_dict': self.target_model.state_dict(),
            'memory': self.memory,
            'epsilon': self.epsilon
        }, path)
        print(f"[INFO] Checkpoint Agent saved to {path}")
    
    def load_agent(self, path="agent_checkpoint.pth"):
        """Load entire agent: model, target model, memory, and epsilon
            When using the agent across instances of gameplay"""
        checkpoint = torch.load(path, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_state_dict'])
        self.memory = checkpoint['memory']
        self.epsilon = checkpoint['epsilon']
        print(f"[INFO] Checkpoint Agent loaded from {path}")


# # Initialize environment
# local_env = DotsAndBoxesEnv(visualize=False)
# # Initialize agent
# agent = DQNAgent(local_env)
# # train agent
# agent.train_agent(num_episodes=10)