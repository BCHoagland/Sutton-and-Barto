import numpy as np

class Env():
    def __init__(self):
        self.width = 9
        self.height = 6

        self.start_pos = np.array([0, 3])
        self.goal = np.array([self.width-1, self.height-1])

        self.obstacles = [[2, 2], [2, 3], [2, 4], [5, 1], [7, 3], [7, 4], [7, 5]]

        self.actions = [[0, 1], [1, 0], [0, -1], [-1, 0]]
        self.actions = [np.array(a) for a in self.actions]

    def reset(self):
        self.pos = self.start_pos
        return self.pos

    def at_goal(self):
        return tuple(self.pos) == tuple(self.goal)

    def step(self, a):
        next_pos = np.clip(self.pos + self.actions[a], 0, [self.width-1, self.height-1])
        if next_pos.tolist() not in self.obstacles:
            self.pos = next_pos
        if self.at_goal():
            return self.pos, 1, True
        return self.pos, 0, False
