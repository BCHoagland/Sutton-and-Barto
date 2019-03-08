import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np

class Policy(nn.Module):
    def __init__(self, n_obs, n_acts):
        super(Policy, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(n_obs, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_acts)
        )

    def forward(self, s):
        s = torch.FloatTensor(s)
        dist = Categorical(logits=self.actor(s))
        a = dist.sample()
        log_p = dist.log_prob(a)
        return a, log_p


class Value(nn.Module):
    def __init__(self, n_obs, n_acts):
        super(Value, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(n_obs, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, s):
        return self.critic(torch.FloatTensor(s))
