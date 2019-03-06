import random

class Env():
    def __init__(self):
        self.reset()

    def reset(self):
        self.s = 500
        return self.s

    def step(self):
        s_plus = random.randint(1, 100)
        s_plus *= -1 if random.random() < 0.5 else 1
        self.s += s_plus
        done = True if (self.s < 1 or self.s > 1000) else False
        if done:
            r = -1 if self.s < 1 else 1
        else:
            r = 0

        return self.s, r, done
