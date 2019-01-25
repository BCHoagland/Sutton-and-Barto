from collections import deque
import time

from env import *
from policy import *
from visualize import *

# hyperparameters
eps = 0.1
gamma = 0.9

track_num = 2
max_episodes = 40000
vis_iter = 500

# create environment and policies
env = Env(track_num)
behavior_policy = BehaviorPolicy(eps)
target_policy = TargetPolicy(env)

# create tables for Q function, Cumulative weights, and number of times each state has been visited
Q = np.zeros((len(env.track[0]), len(env.track), 5, 5, 3, 3))
Q.fill(-200)                                                                # setting values to low numbers helps get good results quickly for this env
C = np.zeros((len(env.track[0]), len(env.track), 5, 5, 3, 3))
visited = np.zeros((len(env.track[0]), len(env.track)))

# runs an episode with the given policy and returns all states, actions, and rewards
def run_episode(policy, display=False):
    S = []
    A = []
    R = []

    s = env.reset()
    while True:
        S.append(tuple(s))

        if display:
            a = policy.get_action(s)                                        # if we're displaying, we're using the target policy
        else:
            a = policy.get_action(s, Q)                                     # if we're not displaying, we're using the behavior policy
        s, r, done = env.step(a)

        A.append(tuple(a))
        R.append(r)

        visited[tuple(env.pos)] += 1

        if display:
            time.sleep(0.2)
            map(env.track, env.pos)

        if done:
            break

    return S, A, R

# begin training
print('Training...', end='', flush=True)
ep_lens = []
for ep in range(max_episodes):
    # generate episode using the behavior policy
    S, A, R = run_episode(behavior_policy)

    # store the length of the episode, updating the plot of episode times and state visits occasionally
    ep_lens.append(len(R))
    if ep % vis_iter == vis_iter - 1:
        plot_ep_len(ep_lens)
        frequency_map(visited)

    # update Q values and target policy using Off-Policy Monte Carlo Control
    G = 0
    W = 1
    for t in reversed(range(len(R))):
        s_a = S[t] + A[t]
        G = gamma * G + R[t]
        C[s_a] += W
        Q[s_a] += (W / C[s_a]) * (G - Q[s_a])
        target_policy.update_policy(S[t], Q)
        if A[t] != tuple(target_policy.get_action(np.array(S[t]))):
            break
        W *= 1 / behavior_policy.get_prob(S[t], A[t], Q, eps)
print('DONE')

# demo target policy
for _ in range(10):
    run_episode(target_policy, display=True)
