import random
import numpy as np

from args import *

class Agent():
    def __init__(self, env):
        self.env = env
        self.reset()

    def reset(self):
        self.Q = np.zeros((self.env.width, self.env.height, 4))

    def get_action(self, s):
        if random.random() < eps:
            return random.randint(0, 3)
        else:
            return np.argmax(self.Q[tuple(s)])

class Q_Learning_Agent(Agent):
    def __init__(self, env):
        Agent.__init__(self, env)

    def update_Q(self, s, a, r, s2, a2):                            # a2 isn't used, but it's a parameter so both agents have the same method signatures
        s_a = tuple(s) + tuple([a])
        self.Q[s_a] += alpha * (r + (gamma * np.max(self.Q[tuple(s2)])) - self.Q[s_a])

class SARSA_Agent(Agent):
    def __init__(self, env):
        Agent.__init__(self, env)

    def update_Q(self, s, a, r, s2, a2):
        s_a = tuple(s) + tuple([a])
        s2_a2 = tuple(s2) + tuple([a2])
        self.Q[s_a] += alpha * (r + (gamma * self.Q[s2_a2]) - self.Q[s_a])

class Expected_SARSA_Agent(Agent):
    def __init__(self, env):
        Agent.__init__(self, env)

    def expected_next_Q(self, s2):
        expectation = 0
        for a2 in range(4):
            s2_a2 = tuple(s2) + tuple([a2])
            prob = 1 - eps + (eps / 4) if a2 == np.argmax(self.Q[tuple(s2)]) else eps / 4
            expectation += prob * self.Q[s2_a2]
        return expectation

    def update_Q(self, s, a, r, s2, a2):
        s_a = tuple(s) + tuple([a])
        s2_a2 = tuple(s2) + tuple([a2])
        self.Q[s_a] += alpha * (r + (gamma * self.expected_next_Q(s2)) - self.Q[s_a])
