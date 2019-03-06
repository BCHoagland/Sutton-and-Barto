from envs.basic_env import BasicEnv
from envs.blocking_env import BlockingEnv
from envs.shortcut_env import ShortcutEnv
from agents.dyna_q import DynaQAgent
from agents.priority_sweeping import PSAgent

from args import *

# train with Dyna-Q and Prioritized Sweeping agents on all mazes
if __name__ == '__main__':
    env = BasicEnv()
    DynaQAgent(env).train(basic_timesteps)
    PSAgent(env).train(basic_timesteps)

    env = BlockingEnv()
    DynaQAgent(env).train(blocking_timesteps)
    PSAgent(env).train(blocking_timesteps)

    env = ShortcutEnv()
    DynaQAgent(env).train(shortcut_timesteps)
    PSAgent(env).train(shortcut_timesteps)
