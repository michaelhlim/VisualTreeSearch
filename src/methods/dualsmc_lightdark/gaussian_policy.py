import torch
from torch.distributions import Normal
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import *

from configs.solver.dualsmc_lightdark import *

dlp = DualSMC_LightDark_Params()


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(GaussianPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)
        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=dlp.log_sig_min, max=dlp.log_sig_max)
        return mean, log_std

    def sample(self, mean_state, par_states):
        state = torch.cat((mean_state, par_states), -1)
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(1 - action.pow(2) + dlp.const)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, torch.tanh(mean)

    def get_action(self, mean_state, par_states):
        a, log_prob, _ = self.sample(mean_state, par_states)
        return a, log_prob[0]