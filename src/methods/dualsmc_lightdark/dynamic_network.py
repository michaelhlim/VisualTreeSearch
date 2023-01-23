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
        self.action_noise_enc = nn.Sequential(
            nn.Linear(1, dlp.mlp_hunits),
            nn.ReLU(),
            nn.Linear(dlp.mlp_hunits, dlp.mlp_hunits),
            nn.ReLU(),
            nn.Linear(dlp.mlp_hunits, 1),
            nn.ReLU()
        )

        self.t_enc = nn.Sequential(
            nn.Linear(sep.dim_state + sep.dim_action, dlp.mlp_hunits),
            nn.ReLU(),
            nn.Linear(dlp.mlp_hunits, dlp.mlp_hunits),
            nn.ReLU(),
            nn.Linear(dlp.mlp_hunits, dlp.mlp_hunits),
            nn.ReLU(),
            nn.Linear(dlp.mlp_hunits, sep.dim_state)
        )

    def t_model(self, state, action): 
        if len(action.shape) == 1:
            action = action.unsqueeze(0).repeat(state.shape[0], 1)
        action_noise = torch.randn_like(action) 
        e = self.action_noise_enc(action_noise)
        action = action + e
        x = torch.cat([state, action], -1)
        delta = self.t_enc(x)
        next_state = state + delta 
        
        return next_state