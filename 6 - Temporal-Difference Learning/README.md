# Temporal-Difference Learning

## Random Walk
Two separate experiments are conducted using TD(0) on a small linear Markov Reward Process (MRP):

<img src="./random_walk/img/walk-diagram.png">
All episodes begin in state C

1. Convergence of state value estimates to the true state values over 100 episodes
<img src="./random_walk/img/state-values.svg">

2. RMS error over 100 episodes using different ⍺ values, each averaged over 100 trials
<img src="./random_walk/img/alphas.svg">
