import numpy as np
from visdom import Visdom

viz = Visdom()

all_t = []
legend = []

def plot(t, n, new):
    means = np.mean(t, axis=0)
    if new:
        all_t.append(means)
    else:
        all_t[len(all_t)-1] = means

    if 'n = ' + str(n) not in legend:
        legend.append('n = ' + str(n))

    title = 'Episode Length'

    viz.line(
        X=np.arange(1, means.shape[0]+1),
        Y=np.array(all_t).transpose(),
        win=title,
        opts=dict(
            title=title,
            legend=legend
        )
    )
