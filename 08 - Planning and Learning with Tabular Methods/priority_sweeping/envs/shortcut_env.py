import numpy as np

class ShortcutEnv():
    def __init__(self):
        self.name = 'Shortcut Maze'

        self.width = 9
        self.height = 6

        self.start_pos = np.array([3, 0])
        self.goal = np.array([self.width-1, self.height-1])

        self.old_obstacles = [[1, 2], [2, 2], [3, 2], [4, 2], [5, 2], [6, 2], [7, 2], [8, 2]]
        self.new_obstacles = [[1, 2], [2, 2], [3, 2], [4, 2], [5, 2], [6, 2], [7, 2]]
        self.obstacles = self.old_obstacles

        self.actions = [[0, 1], [1, 0], [0, -1], [-1, 0]]
        self.actions = [np.array(a) for a in self.actions]

        self.t = 0

    def reset_t(self):
        self.t = 0
        self.obstacles = self.old_obstacles

    def reset(self):
        self.pos = self.start_pos
        return self.pos

    def at_goal(self):
        return tuple(self.pos) == tuple(self.goal)

    def step(self, a):
        self.t += 1
        if self.t == 3000:
            self.obstacles = self.new_obstacles
        next_pos = np.clip(self.pos + self.actions[a], 0, [self.width-1, self.height-1])
        if next_pos.tolist() not in self.obstacles:
            self.pos = next_pos
        if self.at_goal():
            return self.pos, 1, True
        return self.pos, 0, False
