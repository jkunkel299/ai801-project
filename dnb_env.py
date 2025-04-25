import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
import pygame
import sys

spaces = gym.spaces

PLAYER_1 = "AI-1"
PLAYER_2 = "AI-2"
class DotsAndBoxesEnv(gym.Env):
    """ Dots-and-Boxes environment using OpenAI Gym"""
    def __init__(self, grid_size=(5,5), visualize=True):
        """
        __init__ initializes the gameplay environment
        Input parameters:
            grid_size: sets the size of the game board. Default is 5x5 boxes
                (6x6 dots)
            visualize: sets pygame visualization. Default is True (game 
                visible), False will make the gameplay automated in the 
                background without visualization
        Attributes are established for size of the board, action space 
            (boundaries of potential actions), observation space, and current
            player.
        Additional attributes are established for the purpose of visualizing
            gameplay (cell_size, padding, screen_size) using pygame.
        """
        super(DotsAndBoxesEnv, self).__init__
        self.visualize = visualize
        
        self.size = grid_size
        self.rows, self.cols = grid_size
        # Create a board representation where edges are stored 
        #   (this is a square board with grid_size rows and columns)
        self.board = np.zeros((2 * self.rows + 1, 2 * self.cols + 1), dtype=int)

        # Define the action space: The number of possible moves (horizontal 
        #   + vertical edges)
        self.action_space = spaces.Discrete((self.rows + 1) * self.cols + 
                                            (self.cols + 1) * self.rows) 

        # Define the observation space: The state of the board
        self.observation_space = spaces.Box(0, 2, shape=self.board.shape, 
                                            dtype=int)

        # Initialize first player
        self.current_player = 1

        # Pygame setup
        self.cell_size = 80
        self.padding = 20
        self.screen_size = (self.cols * self.cell_size + 2 * self.padding, 
                            self.rows * self.cell_size + 2 * self.padding)
        # Initialize the pygame environment if visualize = True
        if visualize:
            pygame.init()
            self.screen = pygame.display.set_mode(self.screen_size)
            self.screen.fill((255, 255, 255))
            pygame.display.set_caption("Dots and Boxes")

    def reset(self):
        """reset the board and set the first player"""
        self.board.fill(0)
        self.current_player = 1
        if self.visualize:
            self.screen.fill((255, 255, 255))
            self.render()
        return self.board
    
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
    
    def _check_box(self, row, col):
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
            if row == 0: # if the line is on the top edge
                if (self.board[down_left] and self.board[down_right] 
                    and self.board[down_by_2]):
                    # box below
                    boxA = [row, col-1]
                    reward = 1
                else:
                    reward = 0
            elif row == self.rows*2: # if the line is on the bottom edge
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
            if col == 0: # if the line is on the left edge
                if (self.board[up_right] and self.board[down_right] 
                    and self.board[right_by_2]):
                    # box to the right
                    boxA = [row-1, col]
                    reward = 1
                else:
                    reward = 0
            elif col == self.cols*2: # if the line is on the right edge
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
        
    def _calculate_reward(self, row, col):
        """Determine if a move results in the completion of a box
        Inputs are the row and column of the move.
        Outputs are the calculated total reward and boxes filled.
        Calls the check_box function to determine output values."""
        reward = 0
        check_box = self._check_box(row, col)
        boxes_filled = [check_box[1], check_box[2]]
        reward += check_box[0]
        return reward, boxes_filled
    
    def get_valid_actions(self):
        """Return a list of actions that are available on the board to ensure that
        the players cannot select moves that have already been taken (i.e., 
        edges that are already occupied)"""
        available = []
        moves = self.action_space.n
        for i in range(0,moves):
            row, col = self._action_to_index(i)
            if self.board[row, col]==0:
                available.append(i)
        return available

    def step(self, action):
        """Logic for taking an action. The function converts the action value 
        into the row, col position, calculates the boxes filled and reward as
        applicable, recomputes the state of the board, and switches to the next 
        player.
        Input is an action value (numerical position on the game board).
        Output is the board state, reward for boxes filled in this turn, any 
        boxes that are filled in this turn, and the "done" tracking condition"""
        # Convert action index into board coordinates
        row, col = self._action_to_index(action)
        
        # Mark the move for the current player
        self.board[row, col] = self.current_player

        # Calculate the reward for the move
        reward, boxes_filled = self._calculate_reward(row, col)

        board_state = self.board.tolist()
        board_state_no_zero = [x for s in board_state for x in s if x != 0]
        done = not all(board_state_no_zero)
        
        '''If no box is completed, switch the player. 
        (3 - self.current_player) allows the check to appropriately set player
        1 or 2'''
        self.current_player = 3 - self.current_player if reward == 0 else self.current_player

        return self.board, reward, boxes_filled, done

    def render(self):        
        """pygame rendering for better gameplay visualization"""
        # set the line and dot color
        p1_line = (0, 0, 255) # blue for Player 1
        p2_line = (255, 0, 0) # red for Player 2
        dot_color = (0, 0, 0) # black

        # Draw grid of dots
        for i in range(self.rows + 1):
            for j in range (self.cols + 1):
                pygame.draw.circle(self.screen, dot_color, 
                                   (self.padding + j * self.cell_size, 
                                    self.padding + i * self.cell_size), 5)

        # Draw edges based on board state
        for i in range(self.board.shape[0]):
            for j in range(self.board.shape[1]):
                if self.board[i, j] == 1: # Player 1 edge
                    x = (j // 2) * self.cell_size
                    y = (i // 2) * self.cell_size

                    if i % 2 == 0: # Horizontal line
                        pygame.draw.line(self.screen, p1_line, 
                                         (self.padding + x, self.padding + y), 
                                         (self.padding + x + self.cell_size, 
                                          self.padding + y), 5)
                    else: # Vertial line
                        pygame.draw.line(self.screen, p1_line, 
                                         (self.padding + x, self.padding + y), 
                                         (self.padding + x, self.padding + y + 
                                          self.cell_size), 5)
                elif self.board[i, j] == 2: # Player 2 edge
                    x = (j // 2) * self.cell_size
                    y = (i // 2) * self.cell_size

                    if i % 2 == 0: # Horizontal line
                        pygame.draw.line(self.screen, p2_line, 
                                         (self.padding + x, self.padding + y), 
                                         (self.padding + x + self.cell_size, 
                                          self.padding + y), 5)
                    else: # Vertical line
                        pygame.draw.line(self.screen, p2_line, 
                                         (self.padding + x, self.padding + y), 
                                         (self.padding + x, self.padding + y + 
                                          self.cell_size), 5)
        # update display            
        pygame.display.update()
    
    def fill_box(self, row, col):
        """Logic to fill a completed box with the appropriate player's color"""
        x = col * self.cell_size / 2
        y = row * self.cell_size / 2
        if row == 2 * self.rows:
            y = (self.rows - 1) * self.cell_size
        if col == 2 * self.cols:
            x = (self.cols - 1) * self.cell_size

        if self.current_player == 1:
            box_fill = (150, 148, 255) # light blue fill
        else:
            box_fill = (255, 148, 150) # light red fill
        # draw a rectangle in the appropriate position with the appropriate color
        pygame.draw.rect(self.screen, box_fill, (self.padding + x, self.padding + y, 
                                                 self.cell_size, self.cell_size))
        # update pygame display
        pygame.display.update()

    def close(self):
        # close the pygame display
        pygame.quit()
        sys.exit()

    def play_game(env, player1='random_moves', player2='random_moves'):
        """Logic to play the dots and boxes game.
        Inputs are the game envrionment, Player1, and Player2. When the 
        play_game function is called the default for the players is to move
        randomly. The players can also be specified as an AI agent or human.
        Each non-random player must have an appropriate choose_action method, 
        which is called in the gameplay logic.
        Outputs are the scores, players, and winner of the game
        """
        running = True
        done = False
        while running:
            # initialize gameplay variables for state, score, turn
            #state = env.reset()
            scores = [0, 0]
            turn = 0  
            players = [player1, player2]         
            
            while not done:
                if env.visualize:
                    env.render()
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                            env.close()
                    pygame.event.pump()
                    pygame.time.delay(500)        
                valid_actions = env.get_valid_actions()             
                
                if not running:
                    break

                if not valid_actions:
                    #done = True
                    break
                '''logic for implementing two-player gameplay. 
                   Each AI or human player required to have choose_action method 
                   that is called by the gameplay logic.'''
                if env.current_player % 2 != 0:
                    if player1 != 'random_moves':
                        player1_action = player1.choose_action()
                    elif player1 == 'random_moves':
                        player1_action = random.choice(valid_actions)
                    action = player1_action # Player 1's turn
                else:
                    if player2 != 'random_moves':
                        player2_action = player2.choose_action()
                    elif player2 == 'random_moves':
                        player2_action = random.choice(valid_actions)
                    action = player2_action # Player 2's turn

                # if action:
                _, reward, boxes_filled, done = env.step(action)      
                boxA = boxes_filled[0]
                boxB = boxes_filled[1]
                scores[env.current_player - 1] += reward

                '''only call fill_box if visualization is True for the game 
                environment'''
                if env.visualize:
                    if boxA is not None:
                        env.fill_box(boxA[0], boxA[1])
                    if boxB is not None:
                        env.fill_box(boxB[0], boxB[1])
                    env.render()
                turn += 1 # alternate turns
                
            #TODO refine gameplay visualization of scores in pygame
            # pygame.display.set_caption(f"Game over! Final Score: Player 1 = {scores[0]}, Player 2 = {scores[1]}")
            # if scores[0] > scores[1]:
            #     pygame.display.set_caption("Player 1 wins!")
            # elif scores[0] < scores[1]:
            #     pygame.display.set_caption("Player 2 wins!")
            # elif scores[0] == scores[1]:
            #     pygame.display.set_caption("It's a draw!")
            print(f"Game over! Final Score: Player 1 = {scores[0]}, Player 2 = {scores[1]}")
            if scores[0] > scores[1]:
                print("Player 1 wins!")
                winner = "Player 1"
            elif scores[0] < scores[1]:
                print("Player 2 wins!")
                winner = "Player 2"
            elif scores[0] == scores[1]:
                print("It's a draw!")
            return scores, players, winner
        env.close()

if __name__ == "__main__":
    game=DotsAndBoxesEnv(visualize=True)
    game.play_game()