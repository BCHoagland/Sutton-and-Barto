# Planning and Learning with Tabular Methods

## Dyna-Q
An agent must navigate around obstacles to the top right corner of a grid, with the grid and all four possible agent actions shown below:

<img src="./dyna_q/img/map.png">

The agent used Tabular Dyna-Q (a combination of one-step tabular Q-learning with model learning and planning) to efficiently determine the quickest route through the grid. The below graph shows how quickly the agents navigated through the grid each episode, where n denotes how many planning steps were taken every timestep. The higher the n value, the faster the agent was able to determine an optimal path. Each episode length is averaged over 100 trials.

<img src="./dyna_q/img/ep_lens.svg">


## Priority Sweeping

Dyna-Q (while planning, states are randomly sampled) and Priority Sweeping (while planning, samples states according to which change estimated values the most) were compared on three mazes: the maze from the original Dyna-Q testing, as well as two mazes whose layouts changed after 1000 and 3000 timesteps, respectively:

### Blocking Maze
<img src="./priority_sweeping/img/blocking_map.png">

### Shortcut Maze
<img src="./priority_sweeping/img/shortcut_map.png">

### Results
The cumulative reward from training is displayed below, with the x-axis denoting the number of total timesteps passed during training. The results are averaged over 20 trials. Note that since the graphs display cumulative reward, finding an optimal policy results in the reward curve becoming linear.

<img src="./priority_sweeping/img/basic.svg">
<img src="./priority_sweeping/img/blocking.svg">
<img src="./priority_sweeping/img/shortcut.svg">

Note that the final linear reward curve of the Priority Sweeping agent on the shortcut environment has higher slope than the Dyna-Q agent's. Upon further inspection it can be seen that the Priority Sweeping agent consistently discovers the shortcut while the Dyna-Q agent does not.

Although Dyna-Q outperformed Priority Sweeping in terms of timestep efficiency, priority sweeping can be seen to significantly outperform Dyna-Q in terms of number of updates per training session. Dyna-Q must perform the maximum number of updates, and as such performs about an order of magnitude more updates per fixed set of timesteps as priority sweeping does on the three training tasks. The following graphs show the two agents' cumulative reward as a function of parameter updates, with each agent collecting this data over a fixed number of timesteps. These graphs correspond to the same cumulative reward graphs as above.

<img src="./priority_sweeping/img/basic_param.svg">
<img src="./priority_sweeping/img/blocking_param.svg">
<img src="./priority_sweeping/img/shortcut_param.svg">
