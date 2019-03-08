import gym
import numpy as np
import torch
import torch.optim as optim

from model import *
from visualize import *

# hyperparameters
max_episodes = 1000
γ = 0.99
α = 0.001

# make environment
env = gym.make('CartPole-v0')
n_obs = env.observation_space.shape[0]
n_acts = env.action_space.n

def train():
    clear_vis()

    # make policy
    π = Policy(n_obs, n_acts)
    opt_π = optim.SGD(π.parameters(), lr=α)

    # make value function
    v = Value(n_obs, n_acts)
    opt_v = optim.SGD(v.parameters(), lr=α)

    # run through training episodes
    for ep in range(max_episodes):
        ep_r = 0
        s = env.reset()
        I = 1

        while True:
            # take a step according to policy
            a, log_p = π(s)
            v_s = v(s)
            s2, r, done, _ = env.step(a.numpy())
            ep_r += r

            # determine TD error
            δ = r + (γ * v(s2)) - v_s if not done else r - v_s
            δ = δ.detach()

            # optimize value function
            opt_v.zero_grad()
            (-δ * v_s).backward()
            opt_v.step()

            # optimize policy
            opt_π.zero_grad()
            (-δ * I * log_p).backward()
            opt_π.step()

            # plot cumulative reward if episode has finished
            if done:
                if ep % 5 == 4:
                    plot_r(ep_r)
                break

            # move to next state
            I *= γ
            s = s2

train()
