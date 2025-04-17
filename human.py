import pygame

class Human():
    def __init__(self, env):
        self.env = env
    # Human mouse click action index
    def mouse_click_to_action(self, pos):
        """
        Converts a mouse click position into a valid game action.
        Checks whether the clicked location corresponds to a drawable edge
        (horizontal or vertical line) on the Dots and Boxes board.

        Parameters:
        pos (tuple): (x, y) pixel coordinates of the mouse click.
        env (DotsAndBoxesEnv): The current game environment instance.

        Returns:
        int or None: The action index if the click maps to a valid edge, otherwise None.
        """

        x, y = pos
        x -= self.env.padding
        y -= self.env.padding

        if x < 0 or y < 0:
            return None

        row_pos = y / (self.env.cell_size / 2)
        col_pos = x / (self.env.cell_size / 2)

        row = round(row_pos)
        col = round(col_pos)

        if row < 0 or col < 0 or row >= self.env.board.shape[0] or col >= self.env.board.shape[1]:
            return None

        if (row % 2 == 0 and col % 2 == 1) or (row % 2 == 1 and col % 2 == 0):
            for action in self.env.get_valid_actions():
                try:
                    r, c = self.env._action_to_index(action)
                    if r == row and c == col:
                        return action
                except IndexError:
                    continue
        return None

    def choose_action(self):
        waiting_for_input = True
        while waiting_for_input:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    self.close()
                    return
                if event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    action = self.mouse_click_to_action(pos)
                    if action is not None and action in self.env.get_valid_actions():
                        return action