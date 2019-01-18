from math import sqrt, log
import random
import numpy as np
from bandit import Bandit
from visualize import *

# number of possible actions in the k-armed bandit problem
k = 10
num_agents = 1000

# softmax function
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# policy for choosing and evaluating actions
class Policy():
    def __init__(self, k):
        self.probs = [0] * k

    # sample from the actions
    def get_action(self, H):
        self.probs = softmax(H)
        return np.random.choice(k, 1, p=self.probs)[0]

    # get the probability of selecting a given action
    def get_prob(self, a):
        return self.probs[a]

# train an agent with the given ε hyperparameter
def train(alpha):

    # initialize random k-armed bandit scenarios for each agent running in parallel
    bandit = [Bandit(k) for _ in range(num_agents)]

    # keep track of each agent's optimal actions (1) and non-optimal actions (0)
    successes = np.array([0] * num_agents)

    # initialize action-value estimates and number of times each action has been taken to 0
    H = [[0] * k for _ in range(num_agents)]

    # total reward; this will be used to calculate the average reward
    r_total = [0 for _ in range(num_agents)]

    # create a policy
    policy = [Policy(k) for _ in range(num_agents)]

    for step in range(1000):
        for agent in range(num_agents):

            # take action by sampling from policy
            a = policy[agent].get_action(H[agent])

            # get a reward by acting with the chosen action
            r = bandit[agent].act(a)
            r_total[agent] += r

            # update preferences based on the gradient of the expected reward
            for i in range(k):
                if i == a:
                    H[agent][i] += alpha * (r - (r_total[agent] / (step+1))) * (1 - policy[agent].get_prob(i))
                else:
                    H[agent][i] -= alpha * (r - (r_total[agent] / (step+1))) * policy[agent].get_prob(i)

            # store if the agent acted optimally or not
            if a == bandit[agent].optimal_a:
                successes[agent] = 1
            else:
                successes[agent] = 0

        # plot the mean progress of all agents
        new = True if step == 0 else False
        plot(step, successes.mean(), '% Optimal Action - Gradient Bandit', '⍺: ' + str(alpha), new)
