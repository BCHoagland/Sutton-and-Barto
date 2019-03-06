import torch
import torch.nn as nn
from torch.distributions import Categorical

class Q(nn.Module):
    def __init__(self):
        super(Q, self).__init__()
        self.value = nn.Sequential(
            nn.Linear(5, 10),
            nn.ELU(),
            nn.Linear(10, 1)
        )

    def forward(self, s, a):
        s_a = torch.FloatTensor([s[0], s[1], s[2], s[3], a])
        return self.value(s_a)
