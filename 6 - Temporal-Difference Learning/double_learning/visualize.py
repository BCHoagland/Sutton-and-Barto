import numpy as np
from visdom import Visdom

viz = Visdom()

all_r = []

legend = ['Q-Learning', 'Double Q-Learning']

def plot(r, new):
    means = np.mean(r, axis=0)
    if new:
        all_r.append(means)
    else:
        all_r[len(all_r)-1] = means

    title = '% Left Actions from A'

    viz.line(
        X=np.arange(len(all_r[0])),
        Y=np.array(all_r).transpose(),
        win=title,
        opts=dict(
            title=title,
            legend=legend[:len(all_r)]
        )
    )
