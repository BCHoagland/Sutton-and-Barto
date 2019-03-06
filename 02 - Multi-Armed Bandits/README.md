# 2 - Multi-Armed Bandits
The various algorithms in this section train agents on the k-armed bandits problem.

- [x] ε-Greedy
- [x] ε-Greedy with Optimistic Initialization
- [x] Upper-Confidence-Bound (UCB) Action Selection
- [x] Gradient Bandit


## ε-Greedy
Take a random action with probability ε, otherwise take the action with the highest estimated value.
Estimated values are updated with an incremental running average.

It can be seen that lower ε corresponds with slower but steadier convergence to an optimal solution. In the case of ε = 0, the agent never explores and therefore can only learn optimal strategies by random chance.


## ε-Greedy with Optimistic Initialization
The initial value estimations of each action start at some value larger than what their real value probably is. As the agent tries greedy actions and is disappointed, those actions' values drop while the other actions' value estimates stay high. This kickstarts exploration at the beginning of training, even if ε = 0.

This implementation also uses a constant step size ⍺ instead of a 1/N step size like the normal ε-greedy implementation does.


## Upper-Confidence-Bound (UCB) Action Selection
UCB includes an exploration term in the value estimate, ensuring non-random exploration (unlike ε-greedy) without optimistic initialization.

This implementation uses a constant step size, and adds 1 to both terms under the radical in the maximization function as an easy way of avoiding log(0) or division by 0.


## Gradient Bandit
Store preference values for each action relative to the other actions. The probability of choosing an action is based on the value of its preference put through the softmax function along with the other actions' preferences.

The preferences are updated based on the gradient of the expected reward.
