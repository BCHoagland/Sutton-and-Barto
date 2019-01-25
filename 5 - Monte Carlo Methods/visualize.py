import numpy as np
from visdom import Visdom

viz = Visdom()

# converts the ascii track into numbers and flips the track vertically so it gets plotted correctly
def track_to_nums(track):
    nums = [[] for _ in range(len(track))]
    for i in range(len(track)):
        row = track[i]
        for char in row:
            if char == 'X':
                nums[len(track)-1-i].append(0)
            elif char == '.':
                nums[len(track)-1-i].append(1)
            elif char == '|':
                nums[len(track)-1-i].append(2)
            else:
                nums[len(track)-1-i].append(3)
    return nums

def map(track, pos):
    track = track_to_nums(track)
    track[pos[1]][pos[0]] = 4                           # the track is inverted on the line above, so the inverted y-coord can be used directly

    viz.heatmap(
        X=np.array(track),
        win='Racetrack',
        opts=dict(
            title='Racetrack'
        )
    )

def frequency_map(visited):
    viz.heatmap(
        X=visited.transpose(),
        win='Visited',
        opts=dict(
            title='Number of Times Visited',
            colormap='Hot'
        )
    )

def plot_ep_len(ep_lens):
    viz.line(
        X=np.arange(len(ep_lens)),
        Y=np.array(ep_lens),
        win='Episode Lengths',
        opts=dict(
            title='Episode Lengths'
        )
    )
