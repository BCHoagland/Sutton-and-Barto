import numpy as np
from visdom import Visdom

viz = Visdom()

def contour(S, data, title):
    viz.line(
        X=np.array(S),
        Y=np.array(data),
        win=title,
        opts=dict(
            title=title
        )
    )

def plot_V(S, V):
    contour(S, V[1:-1], 'Values - Gambler')

def plot_P(S, P):
    contour(S, P[1:-1], 'Policy - Gambler')
