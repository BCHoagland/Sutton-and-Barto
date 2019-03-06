import numpy as np
from visdom import Visdom

viz = Visdom()

u_dict = {}
r_dict = {}
legend = []
envs = []

def plot(updates, r, agent, new):
    if agent.env_name not in envs:
        envs.append(agent.env_name)
        u_dict[agent.env_name] = []
        r_dict[agent.env_name] = []
    all_u = u_dict[agent.env_name]
    all_r = r_dict[agent.env_name]

    if new:
        all_u.append(updates)
        all_r.append(r)
    else:
        all_u[len(all_u)-1] = updates
        all_r[len(all_r)-1] = r

    if agent.name not in legend:
        legend.append(agent.name)

    title = 'Cumulative Reward - ' + agent.env_name

    viz.line(
        X=np.array(all_u).transpose(),
        Y=np.array(all_r).transpose(),
        win=title,
        opts=dict(
            title=title,
            legend=legend,
            xlabel='Parameter Updates'
        )
    )
