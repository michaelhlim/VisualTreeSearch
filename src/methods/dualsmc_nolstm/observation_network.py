import random
import torch
import ctypes
ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll')
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam
from utils.utils import *
from configs.solver.dualsmc import *


class MeasureNetwork(nn.Module):
    def __init__(self):
        super(MeasureNetwork, self).__init__()
        self.dim_m = 16
        self.obs_encode = nn.Sequential(
            nn.Linear(DIM_OBS, DIM_HIDDEN),
            nn.ReLU(),
            nn.Linear(DIM_HIDDEN, DIM_HIDDEN),
            nn.ReLU(),
            nn.Linear(DIM_HIDDEN, DIM_HIDDEN),
            nn.ReLU()
        )
        self.lstm_replace = nn.Sequential(
            nn.Linear(DIM_HIDDEN, DIM_LSTM_HIDDEN),
            nn.ReLU()
            # nn.Linear(192, 128),
            # nn.Sigmoid()
        )
        self.lstm_out = nn.Sequential(
            nn.Linear(DIM_LSTM_HIDDEN, self.dim_m),
            nn.ReLU()
        )
        self.m_net = nn.Sequential(
            nn.Linear(self.dim_m + DIM_STATE, DIM_HIDDEN),
            nn.ReLU(),
            nn.Linear(DIM_HIDDEN, DIM_HIDDEN),
            nn.ReLU(),
            nn.Linear(DIM_HIDDEN, 1),
            nn.Sigmoid()
        )

    def m_model(self, state, obs, hidden, cell, num_par=NUM_PAR_PF):
        # state: (B * K, dim_s)
        # obs: (B, dim_s)
        obs_enc = self.obs_encode(obs)  # (batch, dim_hidden)
        x = self.lstm_replace(obs_enc) # (batch, dim_lstm_hidden)
        x = self.lstm_out(x)  # (batch, dim_m)
        x = x.repeat(num_par, 1)  # (batch * num_par, dim_m)
        x = torch.cat((x, state), -1)  # (batch * num_par, dim_m + 2)
        lik = self.m_net(x).view(-1, num_par)  # (batch, num_par)
        return lik, 0, 0


class ProposerNetwork(nn.Module):
    def __init__(self):
        super(ProposerNetwork, self).__init__()
        self.dim = 64
        self.obs_encode = nn.Sequential(
            nn.Linear(DIM_OBS, DIM_HIDDEN),
            nn.ReLU(),
            nn.Linear(DIM_HIDDEN, DIM_HIDDEN),
            nn.ReLU(),
            nn.Linear(DIM_HIDDEN, self.dim),
            nn.ReLU()
        )
        self.p_net = nn.Sequential(
            nn.Linear(self.dim * 2, DIM_HIDDEN),
            nn.ReLU(),
            nn.Linear(DIM_HIDDEN, DIM_HIDDEN),
            nn.ReLU(),
            nn.Linear(DIM_HIDDEN, DIM_HIDDEN),
            nn.ReLU(),
            nn.Linear(DIM_HIDDEN, 2)
        )

    def forward(self, obs, num_par=NUM_PAR_PF):
        obs_enc = self.obs_encode(obs)  # (B, C)
        obs_enc = obs_enc.repeat(num_par, 1)  # (B * num_par, C)
        z = torch.randn_like(obs_enc)  # (B * num_par, C)
        x = torch.cat([obs_enc, z], -1)  # (B * num_par, 2C)
        proposal = self.p_net(x)  # [B * num_par, 2]
        return proposal

