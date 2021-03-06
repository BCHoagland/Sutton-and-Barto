import numpy as np
from visdom import Visdom

viz = Visdom()

all_r = []

def clear_vis():
    del all_r[:]

def plot_r(r, use_baseline):
    title = 'REINFORCE with Baseline - CartPole-v0' if use_baseline else 'REINFORCE - CartPole-v0'
    all_r.append(r)
    viz.line(
        X=np.arange(1, len(all_r) * 5 + 1, 5),
        Y=np.array(all_r),
        win=title,
        opts=dict(
            title=title,
            xlabel='Episode',
            ylabel='Episode Reward'
        )
    )
