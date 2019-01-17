import random
import numpy as np
from bandit import Bandit
from visualize import *

# number of possible actions in the k-armed bandit problem
k = 10
num_agents = 1000

# train an agent with the given epsilon hyperparameter
def train(eps):

    # initialize random k-armed bandit scenarios for each agent running in parallel
    bandit = [Bandit(k) for _ in range(num_agents)]

    # keep track of each agent's optimal actions (1) and non-optimal actions (0)
    successes = np.array([0] * num_agents)

    # initialize action-value estimates and number of times each action has been taken to 0
    Q = [[0] * k for _ in range(num_agents)]
    N = [[0] * k for _ in range(num_agents)]

    for step in range(1000):
        for agent in range(num_agents):

            # take random action with probability eps
            # otherwise take action that maximizes the stored action-value estimates
            if random.random() < eps:
                a = random.randint(0, k - 1)
            else:
                a = Q[agent].index(max(Q[agent]))

            # get a reward by acting with the chosen action
            r = bandit[agent].act(a)

            # update stored values
            N[agent][a] += 1
            Q[agent][a] += (1 / N[agent][a]) * (r - Q[agent][a])             # incremental running average of rewards for that action

            # plot an averaged version of the the agent's progress
            if a == bandit[agent].optimal_a:
                successes[agent] = 1
            else:
                successes[agent] = 0

        new = True if step == 0 else False
        plot(step, successes.mean(), 'ε-Greedy Bandit - Percent Optimal Action', str(eps), new)

# train agents with different epsilon values
epsilons = [0, 0.01, 0.1]
for eps in epsilons:
    train(eps)
