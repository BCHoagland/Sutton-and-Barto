import numpy as np

from visualize import *

def train_mc(env, γ, α, num_episodes):
    w = [0] * 10

    print('Training with MC |----------|\n                 |', end='', flush=True)
    for ep in range(num_episodes):
        S = []
        R = []
        s = env.reset()
        while True:
            S.append(s)
            s, r, done = env.step()
            R.append(r)
            if done:
                for i in reversed(range(len(R)-1)):
                    R[i] += γ * R[i+1]
                for i in range(len(R)):
                    ind = (S[i]-1) // 100
                    w[ind] += α * (R[i] - w[ind])
                break

        if ep % (num_episodes / 10) == (num_episodes / 10) - 1:
            print('-', end='', flush=True)

    plot(w, 'MC State Aggregation')
    print('|')
