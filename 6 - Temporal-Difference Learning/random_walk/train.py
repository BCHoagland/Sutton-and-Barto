from math import sqrt
from copy import deepcopy

from env import *
from visualize import *

# true values
true_values = [1/6, 1/3, 1/2, 2/3, 5/6, 0]

# return a newly initialized array of estimated state values
def init_V():
    V = [0.5] * len(positions)
    V.append(0)                                             # V(term) = 0; V[-1] and V[len(positions)] will both access this
    return V

# return the RMS error of the given estimated state values and the true state values
def RMS(values):
    n = len(values[:-1])
    error = 0
    for i in range(n):
        error += (true_values[i] - values[i])**2
    return sqrt(error / n)

def train_V():
    V = init_V()

    # all_Vs starts out with the true state values and the starting state value estimates
    all_V = [true_values, deepcopy(V)]
    plot_V(all_V)

    # run 100 episodes
    for ep in range(1, 101):

        # step through environment and perform TD(0) updates to state value estimates
        s = reset()
        while True:
            s2, r, done = step(s)
            V[s] += 0.1 * (r + V[s2] - V[s])
            s = s2

            if done:
                break

        # plot state values on certain episodes
        if ep == 1 or ep == 10 or ep == 100:
            all_V.append(deepcopy(V))
            plot_V(all_V)

def train_RMS(alpha):
    all_RMS = []

    # run 100 trials
    for agent in range(100):
        V = init_V()

        agent_RMS = [RMS(V)]

        # run 100 episodes per trial
        for ep in range(100):

            # step through environment and perform TD(0) updates to state value estimates
            s = reset()
            while True:
                s2, r, done = step(s)
                V[s] += alpha * (r + V[s2] - V[s])
                s = s2

                if done:
                    break

            agent_RMS.append(RMS(V))

        all_RMS.append(agent_RMS)

    # plot average RMS error once all trials have completed
    means = np.array(all_RMS).mean(axis=0)
    plot_RMS(means, alpha)

if __name__ == '__main__':
    train_V()

    train_RMS(0.15)
    train_RMS(0.1)
    train_RMS(0.05)
