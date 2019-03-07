import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np

class Policy(nn.Module):
    def __init__(self, n_obs, n_acts):
        super(Policy, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(n_obs, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_acts)
        )

    def forward(self, s):
        dist = Categorical(logits=self.main(torch.FloatTensor(s)))
        return dist.sample().numpy()

    def get_log_p(self, s, a):
        dist = Categorical(logits=self.main(s))
        return dist.log_prob(a)
