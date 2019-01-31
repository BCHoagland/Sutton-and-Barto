import random
import numpy as np

from args import *

class Agent():
    def __init__(self, env):
        self.env_width = env.width
        self.env_height = env.height
        self.reset()

    def reset(self):
        self.Q = np.zeros((self.env_width, self.env_height, 4))
        self.model = np.zeros((self.env_width, self.env_height, 4, 2), dtype=object)
        self.old_s = []
        self.old_a = []

    def update_Q(self, s, a, r, s2):
        s_a = tuple(s) + tuple([a])
        s2_max_a = tuple(s2) + tuple([np.argmax(self.Q[tuple(s2)])])
        self.Q[s_a] += α * (r + (γ * self.Q[s2_max_a]) - self.Q[s_a])

    def update_model(self, s, a, r, s2):
        s_a = tuple(s) + tuple([a])
        self.model[s_a][0] = r
        self.model[s_a][1] = s2

    def get_action(self, s):
        if random.random() < ε:
            a = random.randint(0, 3)
        else:
            possible_a = self.Q[tuple(s)]
            all_a = np.argwhere(possible_a == np.amax(possible_a)).flatten()
            a = np.random.choice(all_a)

        self.old_s.append(s)
        self.old_a.append(a)
        return a

    def sample_past(self):
        ind = random.randint(0, len(self.old_s)-1)
        return self.old_s[ind], self.old_a[ind]

    def predict_transition(self, s, a):
        s_a = tuple(s) + tuple([a])
        return self.model[s_a][0], self.model[s_a][1]
