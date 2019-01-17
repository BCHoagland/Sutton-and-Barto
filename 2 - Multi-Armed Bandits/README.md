# 2 - Multi-Armed Bandits
The various algorithms in this section train agents on the k-armed bandits problem.

- [x] ε-Greedy
- [ ] ε-Greedy with Optimistic Initialization
- [ ] Upper-Confidence-Bound (UCB) Action Selection
- [ ] Gradient Bandit


## ε-Greedy
Take a random action with probability ε, otherwise take the action with the highest estimated value.
Estimated values are updated with an incremental running average.

It can be seen that lower ε corresponds with slower but steadier convergence to an optimal solution. In the case of ε = 0, the agent never explores and therefore can only learn optimal strategies by random chance.
