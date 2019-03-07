import gym
import numpy as np
import torch
import torch.optim as optim

from model import Policy, Value
from storage import *
from visualize import *

# hyperparameters
max_episodes = 1000
γ = 0.99
α = 0.001

# make environment
env = gym.make('CartPole-v0')
n_obs = env.observation_space.shape[0]
n_acts = env.action_space.n

def calc_returns():
    # calculate discounted returns
    for i in reversed(range(len(rewards) - 1)):
        rewards[i] += γ * rewards[i + 1]

    # normalize returns for faster convergence
    mean = np.mean(rewards)
    std = np.std(rewards)
    for i in range(len(rewards)):
        rewards[i] = (rewards[i] - mean) / std

def train(use_baseline):
    clear_vis()

    # make policy
    π = Policy(n_obs, n_acts)
    opt_π = optim.SGD(π.parameters(), lr=α)

    # make value function
    v = Value(n_obs)
    opt_v = optim.SGD(v.parameters(), lr=α)

    # run through training episodes
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
                    plot_r(ep_r, use_baseline)
                break
            else:
                store(s, a, r)
                s = s2

        # calculate discounted returns and get stored observations
        calc_returns()
        s, a, r = get_history()

        # if using a baseline, update the value function
        if use_baseline:
            # modify returns with the baseline
            b = v(s).squeeze()
            δ = r - b

            # update value function
            opt_v.zero_grad()
            (-torch.dot(δ, b)).backward(retain_graph=True)
            opt_v.step()

        # if not using a baseline, just proceed with the normal returns
        else:
            δ = r

        # update policy for each state-action pair in the trajectory
        opt_π.zero_grad()
        (-torch.dot(π.get_log_p(s, a), δ)).backward()
        opt_π.step()

# train without baseline
train(False)
# train with baseline
train(True)
