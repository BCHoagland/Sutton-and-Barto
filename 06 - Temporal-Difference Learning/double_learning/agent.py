import random
import numpy as np

from args import *

class Q_Agent():
    def reset(self):
        self.Q = np.zeros((4, 2))

    def get_action(self, s):
        if random.random() < eps:
            return random.randint(0, 1)
        else:
            return np.argmax(self.Q[tuple([s])])

    def update_Q(self, s, a, r, s2):
        s = tuple([s]); s2 = tuple([s2]); a = tuple([a])
        s_a = s + a
        self.Q[s_a] += alpha * (r + (gamma * np.max(self.Q[tuple(s2)])) - self.Q[s_a])

class Double_Q_Agent():
    def reset(self):
        self.Q1 = np.zeros((4, 2))
        self.Q2 = np.zeros((4, 2))

    def get_action(self, s):
        if random.random() < eps:
            return random.randint(0, 1)
        else:
            return np.argmax(self.Q1[tuple([s])] + self.Q2[tuple([s])])

    def update_Q(self, s, a, r, s2):
        s = tuple([s]); s2 = tuple([s2]); a = tuple([a])
        s_a = s + a

        if random.random() < 0.5:
            s2_max_a = s2 + tuple([np.argmax(self.Q1[s2])])
            self.Q1[s_a] += alpha * (r + (gamma * self.Q2[s2_max_a]) - self.Q1[s_a])
        else:
            s2_max_a = s2 + tuple([np.argmax(self.Q2[s2])])
            self.Q2[s_a] += alpha * (r + (gamma * self.Q1[s2_max_a]) - self.Q2[s_a])
