import numpy as np
from visdom import Visdom

viz = Visdom()

def contour(data, title):
    viz.contour(
        X=np.array(data),
        win=title,
        opts=dict(
            title=title
        )
    )

def plot_V(V):
    contour(V, 'Values')


def plot_P(P):
    contour(P, 'Policy')
