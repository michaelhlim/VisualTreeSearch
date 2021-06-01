import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import *


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()
        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)
        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        xt = torch.cat([state, action], 1)
        x1 = F.relu(self.linear1(xu), inplace=False)
        x11 = F.relu(self.linear2(x1), inplace=False)
        x111 = self.linear3(x11)
        x2 = F.relu(self.linear4(xt), inplace=False)
        x22 = F.relu(self.linear5(x2), inplace=False)
        x21 = self.linear6(x22)
        return x111, x21