import random
import numpy as np
import time

from env import Env
from visualize import *

# hyperparameters
max_episodes = 2000
eps = 0.1
alpha = 0.5
gamma = 1

# action sets
four_a = [[0, 1], [0, -1], [1, 0], [-1, 0]]
eight_a = [[0, 1], [0, -1], [1, 0], [-1, 0], [-1, -1], [-1, 1], [1, 1], [1, -1]]
nine_a = [[0, 1], [0, -1], [1, 0], [-1, 0], [-1, -1], [-1, 1], [1, 1], [1, -1], [0, 0]]

# number of actions used during training and whether or not wind is stochastic
actions = four_a
stochastic_wind = False

# create windy gridworld environment
env = Env(actions, stochastic_wind)

# initialize all Q values to 0
Q = np.zeros((env.width, env.height, len(actions)))

# ε-greedy policy derived from Q
def policy(s):
    if random.random() < eps:
        return random.randint(0, len(actions)-1)
    else:
        return np.argmax(Q[tuple(s)])

# SARSA update to the stored action-value estimates
def update_Q(s, a, r, s2, a2):
    s_a = tuple(s) + tuple([a])
    s2_a2 = tuple(s2) + tuple([a2])
    Q[s_a] += alpha * (r + (gamma * Q[s2_a2]) - Q[s_a])

# train a policy while using SARSA to create state-value estimates
def train():
    for ep in range(max_episodes):
        s = env.reset()
        a = policy(s)

        t = 0
        while True:
            t += 1
            s2, r, done = env.step(a)
            a2 = policy(s2)

            update_Q(s, a, r, s2, a2)

            s = s2
            a = a2

            if done:
                break

        plot_t(ep, t, len(actions))

# display an agent going through the gridworld with a deterministic greedy policy derived from Q
def demo():
    s = env.reset()
    a = np.argmax(Q[tuple(s)])

    while True:
        s2, r, done = env.step(a)
        a2 = np.argmax(Q[tuple(s2)])
        s = s2
        a = a2

        map(env)
        time.sleep(0.2)

        if done:
            break

if __name__ == '__main__':
    train()
    demo()
