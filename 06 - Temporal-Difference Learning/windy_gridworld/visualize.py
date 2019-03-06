import numpy as np
from visdom import Visdom

viz = Visdom()

all_ep = []
all_t = []

def plot_t(ep, t, num_a):
    all_ep.append(ep)
    all_t.append(t)

    title = 'Episode Length - ' + str(num_a) + ' Actions'

    viz.line(
        X=np.array(all_ep),
        Y=np.array(all_t),
        win=title,
        opts=dict(
            title=title
        )
    )

def map(env):
    grid = np.zeros((env.width, env.height))
    for i in range(len(env.wind)):
        grid[i,:] = env.wind[i]
    grid[tuple(env.start_pos)] = 3
    grid[tuple(env.goal)] = 3
    grid[tuple(env.pos)] = 4
    grid = grid.transpose()

    title = 'Windy Gridworld'

    viz.heatmap(
        X=grid,
        win=title,
        opts=dict(
            title=title
        )
    )
