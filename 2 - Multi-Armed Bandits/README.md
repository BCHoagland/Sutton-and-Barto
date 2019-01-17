# 2 - Multi-Armed Bandits
The various algorithms in this section train agents on the k-armed bandits problem.


## ε-Greedy Bandit
Take a random action with probability ε, otherwise take the action with the highest estimated value.
Estimated values are updated with an incremental running average.

It can be seen that lower ε corresponds with slower but steadier convergence to an optimal solution. In the case of ε = 0, the agent either acts optimally every time by random chance or never acts optimally.
