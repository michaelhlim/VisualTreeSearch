import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam
from utils.utils import *
from experiments.configs import *


#########################
# Transition model
class DynamicNetwork(nn.Module):
    def __init__(self):
        super(DynamicNetwork, self).__init__()
        self.t_enc = nn.Sequential(
            nn.Linear(DIM_STATE + DIM_ACTION, DIM_HIDDEN),
            nn.ReLU(),
            nn.Linear(DIM_HIDDEN, DIM_HIDDEN),
            nn.ReLU(),
            nn.Linear(DIM_HIDDEN, DIM_HIDDEN),
            nn.ReLU(),
            nn.Linear(DIM_HIDDEN, DIM_STATE * 2)
        )

    def t_model(self, state, action):
        if len(action.shape) == 1:
            action = action.unsqueeze(0).repeat(state.shape[0], 1)
        x = torch.cat([state, action], -1)
        x = self.t_enc(x)
        mean = x[:, :DIM_STATE]
        std = x[:, DIM_STATE:].exp()
        delta = torch.randn_like(state) * std + mean
        next_state = state + action + delta
        return next_state