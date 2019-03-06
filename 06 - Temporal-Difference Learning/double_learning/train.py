import numpy as np

from env import Env
from agent import *
from visualize import *

# training hyperparameters
max_episodes = 300
num_trials = 4000
vis_iter = 100

# create environment and agents
env = Env()
q_agent = Q_Agent()
double_q_agent = Double_Q_Agent()

def train(agent):
    # store the number of times each agent went left from state A
    all_left = []

    for trial in range(num_trials):
        agent.reset()
        all_left.append([])

        # run agent in environment
        for ep in range(max_episodes):
            s = env.reset()
            while True:
                a = agent.get_action(s)
                s2, r, done = env.step(a)
                agent.update_Q(s, a, r, s2)

                # record if the agent went left in state A
                if s == 2:
                    all_left[trial].append(1-a)

                s = s2
                if done:
                    break

        # plot occasionally
        if trial == 0 or trial % vis_iter == vis_iter - 1:
            new = True if trial == 0 else False
            plot(all_left, new)

# train both agents
train(q_agent)
train(double_q_agent)
