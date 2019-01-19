from math import *
import numpy as np
from visualize import *

dim = 20

def poisson(lam, n):
    return pow(lam, n) * np.exp(-lam) / factorial(n)

def expected_return(s, a, V):
    ret = 0.0
    ret -= 2 * abs(a)

    for num_rentals_1 in range(11):
        for num_rentals_2 in range(11):

            p_rentals_1 = poisson(3, num_rentals_1)
            p_rentals_2 = poisson(4, num_rentals_2)

            num_cars_1 = np.clip(s[0] - a, 0, dim)
            num_cars_2 = np.clip(s[1] + a, 0, dim)

            actual_rentals_1 = min(num_cars_1, num_rentals_1)
            actual_rentals_2 = min(num_cars_2, num_rentals_2)

            reward = (actual_rentals_1 + actual_rentals_2) * 10

            p_total = p_rentals_1 * p_rentals_2
            num_cars_1 = np.clip(num_cars_1 - actual_rentals_1 + 3, 0, dim)
            num_cars_2 = np.clip(num_cars_2 - actual_rentals_2 + 2, 0, dim)
            ret += p_total * (reward + (0.9 * V[num_cars_1][num_cars_2]))

            # for num_returns_1 in range(11):
            #     p_returns_1 = poisson(3, num_returns_1)
            #
            #     for num_returns_2 in range(11):
            #         p_returns_2 = poisson(2, num_returns_2)
            #
            #         p_total = p_rentals_1 * p_rentals_2 * p_returns_1 * p_returns_2
            #
            #         num_cars_1 = np.clip(num_cars_1 - actual_rentals_1 + num_returns_1, 0, dim)
            #         num_cars_2 = np.clip(num_cars_2 - actual_rentals_2 + num_returns_2, 0, dim)
            #
            #         ret += p_total * (reward + (0.9 * V[num_cars_2][num_cars_1]))

    return ret

def policy_eval(states, P, V):

    theta = 1

    while True:
        delta = 0

        for x,y in states:
            v = V[x][y]

            a = P[x][y]
            V[x][y] = expected_return((x,y), a, V)

            delta = max(delta, abs(v - V[x][y]))

        plot_V(V)
        if delta < theta:
            return V

def policy_iter(states, P, V):

    stable = True

    for x,y in states:
        old_a = P[x][y]

        max_a = old_a
        max_return = expected_return((x, y), old_a, V)
        for a in range(-5, 6):
            if (a >= 0 and x >= a) or (a < 0 and y >= abs(a)):
                cur_return = expected_return((x, y), a, V)
                if max_return < cur_return:
                    max_a = a
                    max_return = cur_return

        P[x][y] = max_a
        plot_P(P)

        if old_a != P[x][y]:
            stable = False

    return P, stable

if __name__ == '__main__':
    states = []
    for x in range(dim+1):
        for y in range(dim+1):
            states.append((x, y))

    P = [[0] * (dim+1) for _ in range(dim+1)]
    V = [[0] * (dim+1) for _ in range(dim+1)]

    epoch = 0
    while True:
        epoch += 1

        print('Epoch %d: Policy Evaluation...' % epoch, end='', flush=True)
        V = policy_eval(states, P, V)

        print('Policy Iteration...', end='', flush=True)
        P, stable = policy_iter(states, P, V)
        print('DONE')

        if stable:
            break
