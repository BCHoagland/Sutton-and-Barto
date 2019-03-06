from algos import eps_greedy, optimistic_init, ucb, gradient_bandit

# train ε-greedy agents with different ε values
epsilons = [0, 0.01, 0.1]
for eps in epsilons:
    eps_greedy.train(eps)

# train greedy agent with optimistic initial value estimates and compare it to an ε-greedy agent
optimistic_init.train(eps=0, init_value=5)
optimistic_init.train(0.1, 0)

# train UCB agents with different c values
cs = [1, 2, 5]
for c in cs:
    ucb.train(c)

# train gradient bandit agent with different alpha values
alphas = [0.01, 0.1, 0.4]
for alpha in alphas:
    gradient_bandit.train(alpha)
