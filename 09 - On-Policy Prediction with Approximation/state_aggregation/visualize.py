import numpy as np
from visdom import Visdom

viz = Visdom()

def plot(w, title):
    y = []
    for val in w:
        for _ in range(10):
            y.append(val)

    viz.line(
        X=np.arange(1,len(y)+1),
        Y=np.array(y),
        win=title+'s_aggregate',
        opts=dict(
            title=title,
            xlabel='State',
            ylabel='Estimated Value',
            ytickmin=-1,
            ytickmax=1,
            ytickstep=0.2
        )
    )
