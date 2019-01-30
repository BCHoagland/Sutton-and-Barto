from env import Env
from agent import Agent
from visualize import *

num_episodes = 50
num_trials = 100
vis_iter = 10

env = Env()

agent = Agent(env)

def train(n):
    print('Training %d-step planning agent...' % n, end='', flush=True)

    # run trials
    all_t = []
    for trial in range(num_trials):
        all_t.append([])
        agent.reset()

        # run episodes for each trial
        for ep in range(num_episodes):
            s = env.reset()
            t = 0
            while True:
                t += 1

                # one-step tabular Q-learning
                a = agent.get_action(s)
                s2, r, done = env.step(a)
                agent.update_Q(s, a, r, s2)

                # planning
                agent.update_model(s, a, r, s2)
                for _ in range(n):
                    old_s, old_a = agent.sample_past()
                    pred_r, pred_s2 = agent.predict_transition(old_s, old_a)
                    agent.update_Q(old_s, old_a, pred_r, pred_s2)

                # record how long it took to finish the episode
                if done:
                    s = env.reset()
                    all_t[trial].append(t)
                    break
                else:
                    s = s2

        # update plot occasionally
        if trial == 0 or trial % vis_iter == vis_iter-1:
            new = True if trial == 0 else False
            plot(all_t, n, new)
    print('DONE')

# train with different levels of planning
train(0)
train(5)
train(50)
