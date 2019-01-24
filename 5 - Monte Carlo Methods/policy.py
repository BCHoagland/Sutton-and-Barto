import random
import numpy as np

def get_max_a(s, Q):
    s=tuple(s)

    max_a = (0, 0)
    max_q = Q[s + max_a]

    for x_vel in range(-1, 2):
        for y_vel in range(-1, 2):
            a = (x_vel, y_vel)
            q = Q[s + a]
            if q > max_q:
                max_q = q
                max_a = a

    return np.array(max_a)

class TargetPolicy():
    def __init__(self, env):
        track = env.track
        self.P = np.zeros((len(track[0]), len(track), 5, 5, 2)).astype(int)

    def update_policy(self, s, Q):
        self.P[s] = get_max_a(s, Q)

    def get_action(self, s):
        return self.P[tuple(s)]

class BehaviorPolicy():
    def __init__(self, eps):
        self.eps = eps

    def get_action(self, s, Q):
        if random.random() < self.eps:
            return np.random.randint(low=-1, high=2, size=2)
        return get_max_a(s, Q)

    def get_prob(self, s, a, Q, eps):
        if tuple(get_max_a(np.array(s), Q)) == a:
            return 1 - eps + (eps / 9)
        return eps / 9
