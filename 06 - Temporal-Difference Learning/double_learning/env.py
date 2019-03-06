import numpy as np

# state 0 = left terminal
# state 1 = B
# state 2 = A
# state 3 = right terminal

# action 0 = left
# action 1 = right

class Env():
    def reset(self):
        self.pos = 2
        return self.pos

    def step(self, a):
        if self.pos == 1:                                               # any action in B leads to left terminal state
            self.pos = 0
            reward = np.random.normal(-0.1, 1.0)
            return self.pos, reward, True
        elif self.pos == 2:
            if a == 0:
                self.pos = 1
                done = False
            else:
                self.pos = 3
                done = True
            return self.pos, 0, done
        else:
            print('State error')
            quit()
