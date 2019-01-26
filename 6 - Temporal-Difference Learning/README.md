# Temporal-Difference Learning

## Random Walk
Two separate experiments are conducted using TD(0) on a small linear Markov Reward Process (MRP):

<img src="./random_walk/img/walk-diagram.png">
All episodes begin in state C

1. Convergence of state value estimates to the true state values over 100 episodes
<img src="./random_walk/img/state-values.svg">

2. RMS error over 100 episodes using different ⍺ values, each averaged over 100 trials
<img src="./random_walk/img/alphas.svg">


## Windy Gridworld
An agent attempts to move from one position in a grid to another, but some columns in the grid have constant amounts of "wind" that pushes the agent upward during every move.

# Gridworld Map
Map of the windy gridworld. The agent (yellow) must reach the goal (green). Purple areas of the grid have no wind, blue areas move the agent an additional 1 tile upward every move, and the teal areas move the agent an additional 2 tiles upward.
<img src="./windy_gridworld/img/map.svg">

# Agents
SARSA was used to train three agents that each had access to a different action space:

1. Four actions: up, down, left, right
<img src="./windy_gridworld/four/map.svg">

2. Eight actions: all straight and diagonal directions
<img src="./windy_gridworld/eight/map.svg">

3. Nine actions: all straight and diagonal directions, plus an action to stand still (wind still has effect)
<img src="./windy_gridworld/nine/map.svg">
