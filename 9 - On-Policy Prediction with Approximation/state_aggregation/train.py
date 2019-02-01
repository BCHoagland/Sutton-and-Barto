from mc import train_mc
from td import train_td

from env import Env

γ = 1
α = 2e-5
num_episodes = 100000

env = Env()

train_mc(env, γ, α, num_episodes)
train_td(env, γ, α, num_episodes)
