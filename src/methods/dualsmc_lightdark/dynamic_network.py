import torch
import torch.nn as nn
from utils.utils import *

# Configs for floor and no LSTM dual smc
from configs.environments.stanford import *
from configs.solver.dualsmc_lightdark import *

dlp = DualSMC_LightDark_Params()
sep = Stanford_Environment_Params()


#########################
# Transition model
class DynamicNetwork(nn.Module):
    def __init__(self):
        super(DynamicNetwork, self).__init__()
        self.t_enc = nn.Sequential(
            nn.Linear(sep.dim_state + sep.dim_action, dlp.dim_hidden),
            nn.ReLU(),
            nn.Linear(dlp.dim_hidden, dlp.dim_hidden),
            nn.ReLU(),
            nn.Linear(dlp.dim_hidden, dlp.dim_hidden),
            nn.ReLU(),
            nn.Linear(dlp.dim_hidden, sep.dim_state * 2)
        )

    def t_model(self, state, action):
        if len(action.shape) == 1:
            action = action.unsqueeze(0).repeat(state.shape[0], 1)
        x = torch.cat([state, action], -1)
        x = self.t_enc(x)
        mean = x[:, :sep.dim_state]
        std = x[:, sep.dim_state:].exp()
        delta = torch.randn_like(state) * std + mean
        next_state = state + action + delta
        return next_state