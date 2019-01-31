import random
import numpy as np
from heapq import heappush, heappop
import itertools

from args import *
from visualize import *

# priority queue implementation from Python heapq documentation: https://docs.python.org/2/library/heapq.html
class PQueue():
    def __init__(self):
        self.pq = []
        self.entry_finder = {}
        self.REMOVED = '<removed-task>'
        self.counter = itertools.count()

    def push(self, task, priority=0):
        if task in self.entry_finder:
            self.remove_task(task)
        count = next(self.counter)
        entry = [priority, count, task]
        self.entry_finder[task] = entry
        heappush(self.pq, entry)

    def remove_task(self, task):
        entry = self.entry_finder.pop(task)
        entry[-1] = self.REMOVED

    def pop(self):
        while self.pq:
            priority, count, task = heappop(self.pq)
            if task is not self.REMOVED:
                del self.entry_finder[task]
                return task
        raise KeyError('pop from an empty priority queue')

    def is_empty(self):
        return not self.entry_finder

class PSAgent():
    def __init__(self, env):
        self.name = 'Priority Sweeping'
        self.env = env
        self.env_name = env.name
        self.env_width = env.width
        self.env_height = env.height
        self.reset()

    def reset(self):
        self.Q = np.zeros((self.env_width, self.env_height, 4))
        self.model = np.zeros((self.env_width, self.env_height, 4, 2), dtype=object)
        self.preceeding = np.zeros((self.env_width, self.env_height), dtype=object)
        self.p_queue = PQueue()
        self.env.reset()

    def update_Q(self, s, a, r, s2):
        s_a = tuple(s) + tuple([a])
        max_next_Q = np.max(self.Q[tuple(s2)])
        self.Q[s_a] += α * (r + (γ * max_next_Q) - self.Q[s_a])

    def update_model(self, s, a, r, s2):
        s_a = tuple(s) + tuple([a])
        self.model[s_a] = [r, s2]

    def get_action(self, s):
        if random.random() < ε:
            a = random.randint(0, 3)
        else:
            possible_a = self.Q[tuple(s)]
            all_a = np.argwhere(possible_a == np.amax(possible_a)).flatten()
            a = np.random.choice(all_a)
        return a

    def predict_transition(self, s, a):
        s_a = tuple(s) + tuple([a])
        return self.model[s_a][0], self.model[s_a][1]

    def get_P(self, s, a, r, s2):
        s_a = tuple(s) + tuple([a])
        max_next_Q = np.max(self.Q[tuple(s2)])
        return abs(r + (γ * max_next_Q) - self.Q[s_a])

    def add_preceeding(self, s, a, s2):
        if type(self.preceeding[tuple(s2)]) == int:
            self.preceeding[tuple(s2)] = [(tuple(s), a)]
        elif (tuple(s), a) not in self.preceeding[tuple(s2)]:
                self.preceeding[tuple(s2)].append((tuple(s), a))

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

                self.add_preceeding(s, a, s2)

                # record mean cumulative reward for the current timestep
                total_r += r
                all_r[step] += (total_r - all_r[step]) / (trial + 1)

                # n-step planning
                self.update_model(s, a, r, s2)

                p = self.get_P(s, a, r, s2)
                if p > θ:
                    self.p_queue.push((tuple(s), a), -p)

                for _ in range(n):
                    if self.p_queue.is_empty():
                        break
                    old_s, old_a = self.p_queue.pop()
                    pred_r, pred_s2 = self.predict_transition(old_s, old_a)
                    self.update_Q(old_s, old_a, pred_r, pred_s2)
                    updates += 1

                    pre_s = self.preceeding[tuple(s)]
                    if type(pre_s) != int:
                        for (s_hat, a_hat) in pre_s:
                            r_hat, _ = self.predict_transition(s_hat, a_hat)
                            p = self.get_P(s_hat, a_hat, r_hat, s)
                            if p > θ:
                                self.p_queue.push((tuple(s_hat), a_hat), -p)

                # move to next timestep
                s = env.reset() if done else s2

            # plot cumulative reward occasionally
            if trial == 0 or trial % vis_iter == vis_iter-1:
                new = True if trial == 0 else False
                plot(all_r, self, new)
        print('DONE\t\ttotal updates: %d' % (updates))
