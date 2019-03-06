import gym
import numpy as np
import torch
import torch.optim as optim

from agent import Policy
from storage import *
from visualize import *

max_episodes = 1000
γ = 0.99
α = 0.001

env = gym.make('CartPole-v0')
n_obs = env.observation_space.shape[0]
n_acts = env.action_space.n

def calc_returns():
    for i in reversed(range(len(rewards) - 1)):
        rewards[i] += γ * rewards[i + 1]
    mean = np.mean(rewards)
    std = np.std(rewards)
    for i in range(len(rewards)):
        rewards[i] = (rewards[i] - mean) / std

def train():
    π = Policy(n_obs, n_acts)
    opt = optim.SGD(π.parameters(), lr=α)

    for ep in range(max_episodes):

        # generate trajectory
        s = env.reset()
        clear_history()
        ep_r = 0
        while True:
            a = π(s)
            s2, r, done, _ = env.step(a)
            ep_r += r

            if done:
                if ep % 5 == 4:
                    plot_r(ep_r)
                break
            else:
                store(s, a, r)
                s = s2

        # calculate discounted returns
        calc_returns()

        # update policy for each state-action pair in the trajectory
        s, a, r = get_history()
        opt.zero_grad()
        (-torch.dot(π.get_log_p(s, a), r)).backward()
        opt.step()

train()
