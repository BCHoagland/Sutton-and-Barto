from algos import eps_greedy

# train ε-greedy agents with different ε values
epsilons = [0, 0.01, 0.1]
for eps in epsilons:
    eps_greedy.train(eps)
