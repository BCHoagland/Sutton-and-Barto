from visdom import Visdom
import numpy as np

viz = Visdom()

all_ep = []
all_t = []
title = 'Ep Length - CartPole'

def plot(ep, t):
    all_ep.append(ep)
    all_t.append(t)

    viz.line(
        X=np.array(all_ep),
        Y=np.array(all_t),
        win=title,
        opts=dict(
            title=title
        )
    )
