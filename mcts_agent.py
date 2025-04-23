import numpy as np
import random
import math
from dnb_env import DotsAndBoxesEnv

class GameState:
    def __init__(self, board, rows, cols, current_player):
        self.board = np.copy(board)
        self.rows = rows
        self.cols = cols
        self.current_player = current_player

    def get_valid_actions(self):
        available = []
        moves = ((self.rows + 1) * self.cols + 
                 (self.cols + 1) * self.rows)
        for i in range(moves):
            row, col = self._action_to_index(i)
            if self.board[row, col]==0:
                available.append(i)
        return available
    
    def step(self, action):
        row, col = self._action_to_index(action)
        self.board[row, col] = self.current_player
        reward, _, _ = self._check_box_static(row, col)
        done = np.all(self.board != 0)
        if reward == 0:
            self.current_player = 3 - self.current_player
        return self.board, reward, self.current_player, done
    
    def _action_to_index(self, action):
        """Convert the action number (which edge is selected for a move) to 
        board row and column coordinates"""
        total_horizontal_edges = self.rows * (self.cols + 1)
        if action < total_horizontal_edges:
            row = (action // self.cols) * 2
            col = (action % self.cols) * 2 + 1
        else:
            action -= total_horizontal_edges
            row = (action // (self.cols + 1)) * 2 + 1
            col = (action % (self.cols + 1)) * 2
        return row, col
    
    def _check_box_static(self, row, col):
        """determine if a box is made with the current move
        Inputs are the row and column position of the action taken,
        outputs are the reward (increase in score) and the box(es) made by the 
        move"""
        # initialize variables to describe edges relative to current position
        up_by_2 = row-2, col
        down_by_2 = row+2, col
        up_left = row-1, col-1
        up_right = row-1,col+1
        down_left = row+1, col-1
        down_right = row+1, col+1
        left_by_2 = row, col-2
        right_by_2 = row, col+2

        boxA = None
        boxB = None
        
        if row % 2 == 0: # horizontal line is drawn this turn
            if row == 0: # if the line is on the far left edge
                if (self.board[down_left] and self.board[down_right] 
                    and self.board[down_by_2]):
                    # box below
                    boxA = [row, col-1]
                    reward = 1
                else:
                    reward = 0
            elif row == self.rows*2: # if the line is on the far right edge
                if (self.board[up_left] and self.board[up_right] 
                    and self.board[up_by_2]):
                    # box above
                    boxA = [row-2, col-1]
                    reward = 1
                else:
                    reward = 0
            else:
                if (self.board[up_by_2] and self.board[up_left] 
                    and self.board[up_right] and self.board[down_by_2] 
                    and self.board[down_left] and self.board[down_right]):
                    # box above and below
                    boxA = [row-2, col-1]
                    boxB = [row, col-1]
                    reward = 2 # two boxes made at once
                elif (self.board[down_by_2] and self.board[down_left] 
                      and self.board[down_right]): 
                    # box below
                    boxA = [row, col-1]
                    reward = 1
                elif (self.board[up_by_2] and self.board[up_left] 
                      and self.board[up_right]): 
                    # box above
                    boxA = [row-2, col-1]
                    reward = 1
                else:
                    reward = 0
        else: # vertical line is drawn this turn
            if col == 0: # if the line is on the top edge
                if (self.board[up_right] and self.board[down_right] 
                    and self.board[right_by_2]):
                    # box to the right
                    boxA = [row-1, col]
                    reward = 1
                else:
                    reward = 0
            elif col == self.cols*2: # if the line is on the bottom edge
                if (self.board[up_left] and self.board[down_left] 
                    and self.board[left_by_2]):
                    # box to the left
                    boxA = [row-1, col-2]
                    reward = 1
                else:
                    reward = 0
            else:
                if (self.board[up_left] and self.board[down_left] 
                    and self.board[left_by_2] and self.board[up_right] 
                    and self.board[down_right] and self.board[right_by_2]):
                    # box to the left and right
                    boxA = [row-1, col]
                    boxB = [row-1, col-2]
                    reward = 2 # two boxes made at once
                elif (self.board[up_right] and self.board[down_right] 
                      and self.board[right_by_2]): 
                    # box right
                    boxA = [row-1, col]
                    reward = 1
                elif (self.board[up_left] and self.board[down_left] 
                      and self.board[left_by_2]): 
                    # box left
                    boxA = [row-1, col-2]
                    reward = 1               
                else:
                    reward = 0

        return reward, boxA, boxB

class TreeNode:
    '''The MCTS Tree Node Class represents a node in the Monte Carlo Tree used 
        for MCTS (Monte Carlo Tree Search). Each node tracks its parent, 
        children, visit count, total value, and the action that led to it. It 
        supports expansion (adding child nodes), selection of the best child 
        based on UCB1(Upper Confidence Bound 1), and backpropagation of 
        simulation rewards to update statistics.'''
    def __init__(self, parent=None, action=None):
        self.parent = parent
        self.children = {}
        self.action = action
        self.visits = 0
        self.value = 0.0
    '''Checks if all valid actions have been explored from this node.
        A node is considered fully expanded if all valid actions have 
        corresponding child nodes.'''
    def is_fully_expanded(self, valid_actions):
        return set(valid_actions).issubset(set(self.children.keys()))
        
    '''Selects and returns the best child node using the UCT (Upper Confidence 
        Bound for Trees) formula. Balances exploration and exploitation by 
        considering both the average value of each child and the uncertainty (based
        on visit count). The c_param controls the level of exploration.'''
    def best_child(self, c_param=1.4):
        return max(self.children.values(), key=
                   lambda node: node.value / (node.visits + 1e-4) + c_param * 
                   math.sqrt(math.log(self.visits + 1) / (node.visits + 1e-4)))

    '''Expands the current node by selecting a random untried action from the 
        list of valid actions. A new child node is created for the selected 
        action and added to the current node's children. Returns the newly-
        created child node.'''
    def expand(self, valid_actions):
        untried = [a for a in valid_actions if a not in self.children]
        action = random.choice(untried)
        self.children[action] = TreeNode(parent=self, action=action)
        return self.children[action]
        
    '''Recursively update the visit count and value of the current node and its
        ancestors. The reward is added to the current node's value, and the 
        visit count is incremented. If the node has a parent, the reward is 
        negated (assuming a two-player zero-sum game) and propagated up the 
        tree to reflect the opponent's perspective.'''
    def backpropagate(self, reward):
        self.visits += 1
        self.value += reward
        if self.parent:
            self.parent.backpropagate(-reward)


class MCTSAgent:
    '''The Monte Carlo Tree Search (MCTS) Agent class implements the core logic
        of the MCTS algorithm to make decisions in the game. The agent performs
        multiple simulations from the current game state to estimate the best 
        action.
        Each simulation involves:
            1. Selection: Traverses the tree using UCB1 (Upper Confidence Bound 
                1) to select promising nodes.
            2. Expansion: Adds a new child node if unexplored valid actions 
                exist.
            3. Simulation: Plays out the game using a heuristic policy to 
                estimate outcome.
            4. Backpropagation: Propagates the result back up the tree to 
                update node statistics.
        The action with the highest visit count after all simulations is 
        selected.'''
    def __init__(self, env, simulations=100):
        self.simulations = simulations
        self.env = env

    def choose_action(self):
        root = TreeNode()
        max_siumulation_depth = 10
        valid_actions = self.env.get_valid_actions()
        if not valid_actions:
            return None

        # Run multiple simulations
        for _ in range(self.simulations):
            sim_state = GameState(self.env.board, self.env.rows, 
                                  self.env.cols, self.env.current_player)
            node = root

            # Selection: traverse the tree until a leaf or terminal state
            while (node.is_fully_expanded(sim_state.get_valid_actions()) 
                   and node.children):
                node = node.best_child()
                _, _, _, done = sim_state.step(node.action)
                if done:
                    break

            if not sim_state.get_valid_actions():
                continue

            if not node.is_fully_expanded(sim_state.get_valid_actions()):
                node = node.expand(sim_state.get_valid_actions())
                _, reward, _, done = sim_state.step(node.action)
            else:
                reward = 0

            # Simulation: rollout using a heuristic policy
            for _ in range(max_siumulation_depth): 
                ''' max_siumulation_depth implemented to improve performance, 
                rather than playing simulations through to the end of the 
                game'''
                possible_actions = sim_state.get_valid_actions()
                if not possible_actions:
                    break
                sim_action = self.heuristic_policy(possible_actions, sim_state)
                _, sim_reward, _, done = sim_state.step(sim_action)
                reward += sim_reward

            node.backpropagate(reward)

        best_move = max(root.children.items(), key=lambda item: item[1].visits)[0]
        return best_move

    def heuristic_policy(self, actions, env):
        """
        Selects an action during MCTS rollouts using a simple heuristic.
        This heuristic prioritizes moves that immediately complete a box,
        as they score points and grant an extra turn. If such moves exist,
        one is selected at random.
        """
        
        # Prefer actions that complete a box
        best_actions = []
        for action in actions:
            row, col = env._action_to_index(action)
            reward, _, _ = env._check_box_static(row, col)
            if reward > 0:
                best_actions.append(action)

        if best_actions:
            return random.choice(best_actions)

        # Otherwise, avoid risky actions near nearly completed boxes
        safe_actions = []
        for action in actions:
            row, col = env._action_to_index(action)
            # Temporarily mark the action
            original_value = env.board[row, col]
            env.board[row, col] = env.current_player
            risky = False

            # Check all boxes that could be adjacent to this move
            for dr in [-1, 1]:
                for dc in [-1, 1]:
                    r, c = row + dr, col + dc
                    if (0 <= r < env.board.shape[0] 
                        and 0 <= c < env.board.shape[1]):
                        if env.board[r, c] == 0:
                            _, box_posA, box_posB = env._check_box_static(r, c)
                            if box_posA or box_posB:
                                risky = True
            env.board[row, col] = original_value  # Restore

            if not risky:
                safe_actions.append(action)

        return (random.choice(safe_actions) if safe_actions 
                else random.choice(actions))