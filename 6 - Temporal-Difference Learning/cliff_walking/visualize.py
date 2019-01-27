import numpy as np
from visdom import Visdom

viz = Visdom()

all_r = []

legend = ['SARSA', 'Q-Learning']

def plot(r, new):
    means = np.mean(r, axis=0)
    if new:
        all_r.append(means)
    else:
        all_r[len(all_r)-1] = means

    title = 'Episode Returns'

    viz.line(
        Y=np.array(all_r).transpose(),
        win=title,
        opts=dict(
            title=title,
            legend=legend[:len(all_r)]
        )
    )

# def map(env):
#     grid = np.zeros((env.width, env.height))
#     for i in range(len(env.wind)):
#         grid[i,:] = env.wind[i]
#     grid[tuple(env.start_pos)] = 3
#     grid[tuple(env.goal)] = 3
#     grid[tuple(env.pos)] = 4
#     grid = grid.transpose()
#
#     title = 'Windy Gridworld'
#
#     viz.heatmap(
#         X=grid,
#         win=title,
#         opts=dict(
#             title=title
#         )
#     )
