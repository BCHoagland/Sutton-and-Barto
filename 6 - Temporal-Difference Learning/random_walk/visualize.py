import numpy as np
from visdom import Visdom

viz = Visdom()

def plot_V(values):
    ep_nums = [0, 1, 10, 100]
    legend = ['true values'] + ['Ep ' + str(ep_nums[i]) for i in range(len(values)-1)]
    dash = np.array(['dash'] + ['solid'] * (len(values)-1))

    title = 'Random Walk State Values'

    values = np.array(values).transpose()
    values = values[:-1]                                    # the last value is V(term), so we don't plot it
    viz.line(
        X=np.arange(len(values)),
        Y=np.array(values),
        win=title,
        opts=dict(
            title=title,
            ytickmin=0,
            ytickmax=1,
            legend=legend,
            dash=dash
        )
    )

rms_win = None
rms_legend = []

def plot_RMS(errors, alpha):
    global rms_win, rms_legend

    title = 'RMS Error'
    rms_legend.append('⍺ = ' + str(alpha))

    if rms_win is None:
        rms_win = viz.line(
            X=np.arange(len(errors)),
            Y=errors,
            win=title,
            name='⍺ = ' + str(alpha),
            opts=dict(
                title=title,
                legend=rms_legend
            )
        )
    else:
        rms_win = viz.line(
            X=np.arange(len(errors)),
            Y=errors,
            win=rms_win,
            name='⍺ = ' + str(alpha),
            update='append',
            opts=dict(
                title=title,
                legend=rms_legend
            )
        )
