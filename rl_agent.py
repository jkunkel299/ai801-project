import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt
from dnb_env import DotsAndBoxesEnv
from mcts_agent import MCTSAgent

class CNN_DQN(nn.Module):
    """Convolutional neural network"""
    def __init__(self, input_channels, output_dim):
        super(CNN_DQN, self).__init__()
        '''Convolutional layers to extract spatial features of the 
        dots-and-boxes board each convolutional layer applies a 2D convolution 
        over an input image compsed of input planes in this case the image will
         be the dots-and-boxes board'''
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, 
                               kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, 
                               kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, 
                               kernel_size=3, stride=1, padding=1)

        # Fully connected layers to map features to Q-values
        # flatten before passing to dense layer, assuming 5 x 5 board
        self.fc1 = nn.Linear(128 * 11 * 11, 512) 
        ''' a 5x5 box board is actually and 11x11 board space, 
            since it includes both dots and edges'''
        self.fc2 = nn.Linear(512, output_dim) # output layer for Q-values

    def forward(self, x):
        # manually unsqueezing x to alleviate runtime errors
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        ''' ReLU activation applies the rectified linear unit function 
            element-wise'''
        x = F.relu(self.conv1(x)) # First convolutional layer + ReLU activation
        x = F.relu(self.conv2(x)) # 2nd convolutional layer + ReLU activation
        x = F.relu(self.conv3(x)) # 3rd convolutional layer + ReLU activation
        
        # flatten feature maps before feeding into fully connected layer
        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x)) # fully connected layer with ReLU activation
        x = self.fc2(x) # output Q-values for all actions

        return x
    
class DQNAgent:
    """Deep Q-learning agent for use with convolutional neural network"""
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
        self.epsilon_min = 0.0001
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.gamma = 0.99

    def get_state(self, board):
        """Convert board state to sensor format for convolutional neural 
            network"""
        # Shape: (1, height, width)
        return torch.tensor(board[np.newaxis, :, :], dtype=torch.float32) 
    
    def choose_action(self):
        """Epsilon-greedy action selection"""
        state_tensor = self.get_state(self.env.board)
        valid_actions = self.env.get_valid_actions()
        
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

        return action if action in valid_actions else self.choose_action()
        
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

    def check_risky(self, row, col, env):
        """check adjacent positions to see if the current (row, col) move
            gives the opponent the opportunity to make a box. A negative
            reward (negative reinforcement) is returned if a box could be 
            formed in the next move."""
        up_2 = row-2, col
        down_2 = row+2, col
        up_left = row-1, col-1
        up_right = row-1,col+1
        down_left = row+1, col-1
        down_right = row+1, col+1
        left_2 = row, col-2
        right_2 = row, col+2
        
        if row % 2 == 0: # horizontal line is drawn this turn
            if row == 0: # if the line is on the top edge
                if ((env.board[down_left] and env.board[down_right]) 
                    or (env.board[down_2] and env.board[down_left])
                    or (env.board[down_2] and env.board[down_right])):
                    reward = -1
                else:
                    reward = 0
            elif row == env.rows*2: # if the line is on the bottom edge
                if ((env.board[up_left] and env.board[up_right]) 
                    or (env.board[up_2] and env.board[up_left])
                    or (env.board[up_2] and env.board[up_right])):
                    reward = -1
                else:
                    reward = 0
            else:
                if ((env.board[down_left] and env.board[down_right]) 
                    or (env.board[down_2] and env.board[down_left])
                    or (env.board[down_2] and env.board[down_right])):
                    reward = -1
                elif ((env.board[up_left] and env.board[up_right]) 
                    or (env.board[up_2] and env.board[up_left])
                    or (env.board[up_2] and env.board[up_right])):
                    reward = -1
                else:
                    reward = 0
        else: # vertical line is drawn this turn
            if col == 0: # if the line is on the left edge
                if ((env.board[right_2] and env.board[up_right])
                    or (env.board[right_2] and env.board[down_right])
                    or (env.board[down_right] and env.board[up_right])):
                    reward = -1
                else:
                    reward = 0
            elif col == env.cols*2: # if the line is on the right edge
                if ((env.board[left_2] and env.board[up_left])
                    or (env.board[left_2] and env.board[down_left])
                    or (env.board[down_left] and env.board[up_left])):
                    reward = -1
                else:
                    reward = 0
            else:
                if ((env.board[right_2] and env.board[up_right])
                    or (env.board[right_2] and env.board[down_right])
                    or (env.board[down_right] and env.board[up_right])):
                    reward = -1
                elif ((env.board[left_2] and env.board[up_left])
                    or (env.board[left_2
                                  ] and env.board[down_left])
                    or (env.board[down_left] and env.board[up_left])):
                    reward = -1
                else:
                    reward = 0

        return reward

    def train_agent(self, num_episodes=1000):
        """Train the convolutional neural network DQN agent on the Dots and 
            Boxes game"""
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
                action = self.choose_action()
                next_state, reward, _, done = self.env.step(action)

                total_reward += reward

                q_values = self.model(torch
                                      .tensor(state, dtype=torch.float32)
                                      .unsqueeze(0))
                total_q += q_values.max().item()
                moves += 1

                self.store_experience(state, action, reward, next_state, done)
                self.train()

                state = next_state

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

    def train_against_mcts(self, num_episodes=1000):
        """Improving the training algorithm to train against an MCTS agent
            instead of through self-play."""
        mcts_agent = MCTSAgent(self.env, simulations=5)

        for episode in range(num_episodes):
            self.env.reset()
            done = False
            valid_actions = True
            total_q = 0
            moves = 0
            total_reward = 0

            while not done:
                valid_actions = self.env.get_valid_actions()
                if not valid_actions:
                    done = True
                    break

                current_player = self.env.current_player

                if current_player == 1:
                    state = self.env.board
                    action = self.choose_action()
                else:
                    action = mcts_agent.choose_action()
                
                next_state, reward, _, done = self.env.step(action)

                total_reward += reward

                if current_player == 1:
                    q_values = self.model(torch
                                          .tensor(state, dtype=torch.float32)
                                          .unsqueeze(0))
                    total_q += q_values.max().item()

                    self.store_experience(state, action, reward, next_state, done)
                    self.train()

                moves += 1
                
                current_player = (3 - current_player if 
                                  reward == 0 else current_player)

                state = next_state
            
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

    def train_mcts_reward(self, num_episodes=1000):
        """Improving the training algorithm to train against an MCTS agent
            with an improved reward structure, including negative reinforcement
            for giving the opponent the opportunity to form a box, positive 
            reinforcement for taking moves that allow for a continued turn, and
            larger positive and negative rewards for winning or losing the 
            game."""
        mcts_agent = MCTSAgent(self.env, simulations=5)
        scores = [0, 0]

        for episode in range(num_episodes):
            self.env.reset()
            done = False
            valid_actions = True
            total_q = 0
            moves = 0
            total_reward = 0

            while not done:
                valid_actions = self.env.get_valid_actions()
                if not valid_actions:
                    done = True
                    break

                current_player = self.env.current_player
                previous_player = current_player

                if current_player == 1:
                    state = self.env.board
                    action = self.choose_action()
                else:
                    action = mcts_agent.choose_action()
                
                next_state, reward, _, done = self.env.step(action)
                
                # add initial box-forming reward to scores array
                scores[current_player-1] += reward

                total_reward += reward

                if current_player == 1:
                    # penalize for setting up opponent to make a box
                    row, col = self.env._action_to_index(action)
                    reward += self.check_risky(row, col, self.env)

                    q_values = self.model(torch
                                          .tensor(state, dtype=torch.float32)
                                          .unsqueeze(0))
                    total_q += q_values.max().item()

                    self.store_experience(state, action, reward, next_state, done)
                    self.train()

                    last_rl_state = state
                    last_rl_action = action

                moves += 1
                
                current_player = (3 - current_player if 
                                  reward == 0 else current_player)
                
                if current_player == previous_player and previous_player == 1:
                    reward += 0.5 # bonus for getting another turn

                state = next_state

            if done:
                # Endgame bonus, +10 points for win, -10 points for loss
                if scores[0] > scores[1]:
                    reward += 10
                else:
                    reward -= 10
            
                self.store_experience(last_rl_state, last_rl_action, reward, 
                                      next_state, done)
                self.train()
            
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