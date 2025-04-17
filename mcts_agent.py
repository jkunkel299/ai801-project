import numpy as np
import random
import math
from dnb_env import DotsAndBoxesEnv

# ==== MCTS Tree Node Class ====
# Represents a node in the Monte Carlo Tree used for MCTS (Monte Carlo Tree Search).
# Each node tracks its parent, children, visit count, total value, and the action that led to it.
# It supports expansion (adding child nodes), selection of the best child based on UCB1(Upper Confidence Bound 1),
# and backpropagation of simulation rewards to update statistics.
class TreeNode:
    def __init__(self, parent=None, action=None):
        self.parent = parent
        self.children = {}
        self.action = action
        self.visits = 0
        self.value = 0.0
# Checks if all valid actions have been explored from this node
# A node is considered fully expanded if all valid actions have corresponding child nodes.
    def is_fully_expanded(self, valid_actions):
        return set(valid_actions).issubset(set(self.children.keys()))
        
# Selects and returns the best child node using the UCT (Upper Confidence Bound for Trees) formula.
# Balances exploration and exploitation by considering both the average value of each child
# and the uncertainty (based on visit count). The c_param controls the level of exploration.
    def best_child(self, c_param=1.4):
        return max(self.children.values(), key=lambda node: node.value / (node.visits + 1e-4) + c_param * math.sqrt(math.log(self.visits + 1) / (node.visits + 1e-4)))

# Expands the current node by selecting a random untried action from the list of valid actions.
# A new child node is created for the selected action and added to the current node's children.
# Returns the newly created child node.
    def expand(self, env, valid_actions):
        untried = [a for a in valid_actions if a not in self.children]
        action = random.choice(untried)
        self.children[action] = TreeNode(parent=self, action=action)
        return self.children[action]
        
# Recursively update the visit count and value of the current node and its ancestors.
# The reward is added to the current node's value, and the visit count is incremented.
# If the node has a parent, the reward is negated (assuming a two-player zero-sum game)
# and propagated up the tree to reflect the opponent's perspective.
    def backpropagate(self, reward):
        self.visits += 1
        self.value += reward
        if self.parent:
            self.parent.backpropagate(-reward)

# ===== Monte Carlo Tree Search (MCTS) Agent =====
# This class implements the core logic of the MCTS algorithm to make decisions in the game.
# The agent performs multiple simulations from the current game state to estimate the best action.
# Each simulation involves:
#   1. Selection: Traverses the tree using UCB1 (Upper Confidence Bound 1) to select promising nodes.
#   2. Expansion: Adds a new child node if unexplored valid actions exist.
#   3. Simulation: Plays out the game using a heuristic policy to estimate outcome.
#   4. Backpropagation: Propagates the result back up the tree to update node statistics.
# The action with the highest visit count after all simulations is selected.
class MCTSAgent:
    def __init__(self, env, simulations=100):
        self.simulations = simulations
        self.env = env

    def choose_action(self):
        root = TreeNode()
        valid_actions = self.env.get_valid_actions()
        if not valid_actions:
            return None

        # Run multiple simulations
        for _ in range(self.simulations):
            sim_env = self.copy_env(self.env)
            node = root

            # Selection: traverse the tree until a leaf or terminal state
            while node.is_fully_expanded(sim_env.get_valid_actions()) and node.children:
                node = node.best_child()
                _, _, _, done = sim_env.step(node.action)
                if done:
                    break

            if not sim_env.get_valid_actions():
                continue

            if not node.is_fully_expanded(sim_env.get_valid_actions()):
                node = node.expand(sim_env, sim_env.get_valid_actions())
                _, reward, _, done = sim_env.step(node.action)
            else:
                reward = 0

            # Simulation: rollout using a heuristic policy
            while not done:
                possible_actions = sim_env.get_valid_actions()
                if not possible_actions:
                    break
                sim_action = self.heuristic_policy(possible_actions, sim_env)
                _, sim_reward, _, done = sim_env.step(sim_action)
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
            reward, _, _ = env._check_box(row, col)
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
                    if 0 <= r < env.board.shape[0] and 0 <= c < env.board.shape[1]:
                        if env.board[r, c] == 0:
                            _, box_posA, box_posB = env._check_box(r, c)
                            if box_posA or box_posB:
                                risky = True
            env.board[row, col] = original_value  # Restore

            if not risky:
                safe_actions.append(action)

        return random.choice(safe_actions) if safe_actions else random.choice(actions)
        
    # Creates and returns a deep copy of the given DotsAndBoxesEnv environment.
    # This is useful for simulations (during MCTS) where we want to explore future states
    # without modifying the original environment. The board and current player are copied over.

    def copy_env(self, env):
        new_env = DotsAndBoxesEnv(grid_size=env.size, visualize=False)
        new_env.board = np.copy(env.board)
        new_env.current_player = env.current_player
        return new_env
