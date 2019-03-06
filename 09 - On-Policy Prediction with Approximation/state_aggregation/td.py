import numpy as np

from visualize import *

def train_td(env, γ, α, num_episodes):
    w = [0] * 10

    def value(s):
        ind = (s-1) // 100
        return 0 if (s < 1 or s > 1000) else w[ind]

    print('Training with TD(0) |----------|\n                    |', end='', flush=True)
    for ep in range(num_episodes):

        s = env.reset()
        while True:
            s2, r, done = env.step()

            ind = (s-1) // 100
            w[ind] += α * (r + (γ * value(s2)) - value(s))

            s = s2
            if done:
                break

        if ep % (num_episodes / 10) == (num_episodes / 10) - 1:
            print('-', end='', flush=True)

    plot(w, 'TD(0) State Aggregation')
    print('|')
