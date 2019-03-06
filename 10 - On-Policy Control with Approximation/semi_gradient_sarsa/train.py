import gym
import random
import torch
import torch.optim as optim

from value import Q
from visualize import *

max_episodes = 2000
ε = 0.2
γ = 0.99
α = 1e-2
vis_iter = 20

env = gym.make('CartPole-v0')

q = Q()
optimizer = optim.SGD(q.parameters(), lr=α)

# return action that maximizes current q(s, a) approximation
def max_a(s):
    max_q = q(s, 0).item()
    max_a = 0
    for a in range(1, 2):
        if max_q < q(s, a).item():
            max_q = q(s, a).item()
            max_a = a
    return max_a

# return ε-greedy action
def greedy_a(s):
    if random.random() < ε:
        return env.action_space.sample()
    return max_a(s)

# update q function approximation
def update_q(target, s, a):
    error = torch.pow(target - q(s, a), 2)
    optimizer.zero_grad()
    error.backward()
    optimizer.step()

# run for set number of episodes
for ep in range(max_episodes):

    # reset environment
    s = env.reset()
    a = greedy_a(s)

    # step through environment
    steps = 0
    while True:
        steps += 1
        s2, r, done, _ = env.step(a)

        # when episode finishes, the update target is just 'R'
        if done:
            if ep % vis_iter == vis_iter - 1:
                plot(ep, steps)
            update_q(r, s, a)
            break

        # if the episode hasn't finished, the episode target is 'R + γ q(S', A')'
        a2 = greedy_a(s2)
        target = r + (γ * q(s2, a2))
        update_q(target, s, a)

        # move to next state-action pair
        s = s2
        a = a2

# show off policy after training
s = env.reset()
while True:
    env.render()
    a = max_a(s)
    s, _, done, _ = env.step(a)
    if done:
        break
env.close()
