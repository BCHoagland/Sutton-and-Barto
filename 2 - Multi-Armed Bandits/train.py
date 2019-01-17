from algos import eps_greedy, optimistic_init, ucb, gradient_bandit

# train ε-greedy agents with different ε values
epsilons = [0, 0.01, 0.1]
for eps in epsilons:
    eps_greedy.train(eps)

# train ε-greedy with constant step size and optimistic initial values
optimistic_init.train(eps=0, init_value=5)
optimistic_init.train(0.1, 0)
