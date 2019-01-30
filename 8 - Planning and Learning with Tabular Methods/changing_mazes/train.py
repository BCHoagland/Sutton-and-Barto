from envs.blocking_env import BlockingEnv
from envs.shortcut_env import ShortcutEnv
from agents.dyna_q import DynaQAgent
from visualize import *
from args import *

def train(agent, num_timesteps):
    print('Training %d-step planning agent...' % n, end='', flush=True)

    all_r = [0] * num_timesteps
    env = agent.env

    # run through trials
    for trial in range(num_trials):
        agent.reset()
        env.reset_t()
        total_r = 0

        # run trial for set number of timesteps
        s = env.reset()
        for step in range(num_timesteps):

            # one-step tabular q-learning
            a = agent.get_action(s)
            s2, r, done = env.step(a)
            agent.update_Q(s, a, r, s2)

            # record mean cumulative reward for the current timestep
            total_r += r
            all_r[step] += (total_r - all_r[step]) / (trial + 1)

            # n-step planning
            agent.update_model(s, a, r, s2)
            for _ in range(n):
                old_s, old_a = agent.sample_past()
                pred_r, pred_s2 = agent.predict_transition(old_s, old_a)
                agent.update_Q(old_s, old_a, pred_r, pred_s2)

            # move to next timestep
            s = env.reset() if done else s2

        # plot cumulative reward occasionally
        if trial == 0 or trial % vis_iter == vis_iter-1:
            new = True if trial == 0 else False
            plot(all_r, agent, new)
    print('DONE')

# train with Dyna-Q and Dyna-Q+ agents on both mazes
if __name__ == '__main__':
    env = BlockingEnv()
    train(DynaQAgent(env), blocking_timesteps)

    env = ShortcutEnv()
    train(DynaQAgent(env), shortcut_timesteps)
