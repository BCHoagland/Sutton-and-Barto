import numpy as np

class Env():
    def __init__(self, actions, stochastic_wind):
        self.width = 10
        self.height = 7

        self.wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
        self.stochastic_wind = stochastic_wind

        self.actions = [np.array(dir) for dir in actions]

        self.start_pos = np.array([0, 3])
        self.goal = np.array([7, 3])

    def reset(self):
        self.pos = self.start_pos
        return self.pos

    def at_goal(self):
        return tuple(self.pos) == tuple(self.goal)

    def step(self, a_ind):
        extra_wind = np.random.randint(-1, 2) if self.stochastic_wind else 0
        wind_component = np.array([0, self.wind[self.pos[0]]]) + extra_wind
        self.pos = np.clip(self.pos + wind_component + self.actions[a_ind], 0, [self.width-1, self.height-1])
        done = self.at_goal()
        return self.pos, -1, done
