import random
import numpy as np

from args import *
from visualize import *

class DynaQAgent():
    def __init__(self, env):
        self.name = 'Dyna-Q'
        self.env = env
        self.env_name = env.name
        self.env_width = env.width
        self.env_height = env.height
        self.reset()

    def reset(self):
        self.Q = np.zeros((self.env_width, self.env_height, 4))
        self.model = np.zeros((self.env_width, self.env_height, 4, 2), dtype=object)
        self.old_s = []
        self.old_a = []
        self.env.reset()

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

    def train(self, num_timesteps):
        print('Training %s agent on %s...' % (self.name, self.env_name), end='', flush=True)

        updates = 0

        all_r = [0] * num_timesteps
        env = self.env

        # run through trials
        for trial in range(num_trials):
            self.reset()
            env.reset_t()
            total_r = 0

            # run trial for set number of timesteps
            s = env.reset()
            for step in range(num_timesteps):

                # one-step tabular q-learning
                a = self.get_action(s)
                s2, r, done = env.step(a)
                self.update_Q(s, a, r, s2)

                # record mean cumulative reward for the current timestep
                total_r += r
                all_r[step] += (total_r - all_r[step]) / (trial + 1)

                # n-step planning
                self.update_model(s, a, r, s2)
                for _ in range(n):
                    old_s, old_a = self.sample_past()
                    pred_r, pred_s2 = self.predict_transition(old_s, old_a)
                    self.update_Q(old_s, old_a, pred_r, pred_s2)
                    updates += 1

                # move to next timestep
                s = env.reset() if done else s2

            # plot cumulative reward occasionally
            if trial == 0 or trial % vis_iter == vis_iter-1:
                new = True if trial == 0 else False
                plot(all_r, self, new)
        print('DONE\t\ttotal updates: %d' % (updates))
