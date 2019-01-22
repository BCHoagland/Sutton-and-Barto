from math import *
import numpy as np
from visualize import *

# probability of getting heads (and winning the stake)
p_h = 0.4

# convergence factor for value iteration
theta = 1e-10

# discount factor
gamma = 1

def init_S():
    return [i for i in range(1, 100)]

def init_V():
    V = [0] * 101
    V[100] = 1
    return V

def expected_return(s, a, V):
    ret = 0.0

    # there are only two possible next states: winning or losing the state
    # the reward is 0 at every state except in one of the terminal states, so we can just use [Ɣ * V(s')] instead of [r + Ɣ * V(s')]
    ret += p_h * (gamma * V[s + a])
    ret += (1 - p_h) * (gamma * V[s - a])
    return ret

def value_iteration(S, V):
    while True:
        delta = 0

        for s in S:
            v = V[s]

            A = [i for i in range(min(s, 100-s) + 1)]

            # set value of given state equal to the highest expected return from all possible actions in that state
            max_ret = 0
            for a in A:
                cur_ret = expected_return(s, a, V)
                max_ret = cur_ret if cur_ret > max_ret else max_ret
            V[s] = max_ret

            # update the maximum change in value for this iteration
            delta = max(delta, abs(v - V[s]))

        # stop updating the value function once the maximum change in value is very small
        if delta < theta:
            plot_V(S, V)
            return V

def make_policy(S, V):
    P = [0] * (len(S) + 2)

    # set the policy to maximize the value at each state
    # in this scenario, there are many optimal policies due to many ties in value
    for s in S:
        A = [i for i in range(min(s, 100-s) + 1)]

        max_ret = 0
        for a in A:
            cur_ret = expected_return(s, a, V)
            if cur_ret > max_ret:
                max_ret = cur_ret
                P[s] = a

    plot_P(S, P)
    return P

if __name__ == '__main__':
    S = init_S()
    V = init_V()
    V = value_iteration(S, V)
    P = make_policy(S, V)
