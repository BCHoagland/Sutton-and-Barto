from env import Env
from agents import SARSA_Agent, Q_Learning_Agent, Expected_SARSA_Agent
from visualize import *

num_trials = 100
max_episodes = 500

env = Env()

sarsa_agent = SARSA_Agent(env)
q_learning_agent = Q_Learning_Agent(env)
expected_sarsa_agent = Expected_SARSA_Agent(env)

def train(agent):

    all_rewards = []

    for trial in range(num_trials):
        all_rewards.append([])

        agent.reset()

        for ep in range(max_episodes):
            ep_r = 0
            s = env.reset()

            while True:
                a = agent.get_action(s)
                s2, r, done = env.step(a)
                ep_r += r
                a2 = agent.get_action(s2)

                agent.update_Q(s, a, r, s2, a2)

                s = s2

                if done:
                    break

            all_rewards[trial].append(ep_r)

        new = True if trial == 0 else False
        plot(all_rewards, new)

train(sarsa_agent)
train(q_learning_agent)
train(expected_sarsa_agent)
