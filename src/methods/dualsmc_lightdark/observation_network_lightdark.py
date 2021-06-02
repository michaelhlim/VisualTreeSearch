import torch
import torch.nn as nn

from configs.environments.stanford import *
from configs.solver.dualsmc_lightdark import *
from src.methods.dualsmc_lightdark.observation_encoder_lightdark import *
from utils.utils import *

sep = Stanford_Environment_Params()
dlp = DualSMC_LightDark_Params()


# "Branching" behavior
class GAN_Proposer_Measure(nn.Module):
    def __init__(self):
        super(GAN_Proposer_Measure, self).__init__()
        self.observation_encoder = ObservationEncoder()



class MeasureNetwork(GAN_Proposer_Measure):
    def __init__(self):
        super(MeasureNetwork, self).__init__()
        self.dim_m = 64 #16
        self.latent_dim = dlp.latent_dim
        self.dim_hidden = dlp.dim_hidden
        self.dim_lstm_hidden = dlp.dim_lstm_hidden
        self.num_lstm_layer = dlp.num_lstm_layer
        self.dim_state = sep.dim_state

        self.first_layer = nn.Sequential(
            nn.Linear(self.latent_dim, self.dim_hidden),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(self.dim_hidden, self.dim_lstm_hidden, self.num_lstm_layer)
        self.lstm_out = nn.Sequential(
            nn.Linear(self.dim_lstm_hidden, self.dim_m),
            nn.ReLU()
        )

        mlp_hunits = dualsmc_lightdark_params.mlp_hunits
        self.mlp = nn.Sequential(
                nn.Linear(self.dim_m + self.dim_state, mlp_hunits),
                nn.LeakyReLU(),
                nn.Linear(mlp_hunits, mlp_hunits),
                nn.LeakyReLU(),
                nn.Linear(mlp_hunits, mlp_hunits),
                nn.LeakyReLU(),
                nn.Linear(mlp_hunits, 1),
                nn.Sigmoid()
            )

    def m_model(self, state, obs, hidden, cell, num_par=dlp.num_par_pf):
        # state [batch_size * num_par, dim_state]
        # obs [batch_size, in_channels, img_size, img_size]
        enc_obs = self.observation_encoder(state, obs)
        result = self.first_layer(enc_obs) # (batch, dim_hidden)
        x = result.unsqueeze(0)
        x, (h, c) = self.lstm(x, (hidden, cell))
        x = self.lstm_out(x[0])
        x = x.repeat(num_par, 1)  # (batch * num_par, dim_m)        
        x = torch.cat((x, state), -1)  # (batch * num_par, dim_m + dim_state)
        lik = self.mlp(x).view(-1, num_par)
        return lik, h, c



class ProposerNetwork(GAN_Proposer_Measure):
    def __init__(self):
        super(ProposerNetwork, self).__init__()
        self.dim = 64

        self.latent_dim = dualsmc_lightdark_params.latent_dim
        self.dim_hidden = dualsmc_lightdark_params.dim_hidden
        self.dim_state = stanford_environment_params.dim_state

        self.first_layer = nn.Sequential(
            nn.Linear(self.latent_dim, self.dim_hidden),
            nn.ReLU(),
            nn.Linear(self.dim_hidden, self.dim_hidden),
            nn.ReLU(),
            nn.Linear(self.dim_hidden, self.dim),
            nn.ReLU()
        )

        mlp_hunits = dualsmc_lightdark_params.mlp_hunits
        self.mlp = nn.Sequential(
                nn.Linear(self.dim * 2, mlp_hunits),
                nn.LeakyReLU(),
                nn.Linear(mlp_hunits, mlp_hunits),
                nn.LeakyReLU(),
                nn.Linear(mlp_hunits, mlp_hunits),
                nn.LeakyReLU(),
                nn.Linear(mlp_hunits, self.dim_state),
                nn.Sigmoid()
            )


    def forward(self, state, obs, num_par=dualsmc_lightdark_params.num_par_pf):
        # state [batch_size, dim_state]
        # obs [batch_size, in_channels, img_size, img_size]
        _, _, enc_obs = self.observation_encoder(state, obs)  # enc_obs [batch_size, latent_dim]
        result = self.first_layer(enc_obs)  # [batch_size, self.dim]
        result = result.repeat(num_par, 1)  # [batch_size * num_par, self.dim]
        z = torch.randn_like(result)  # [batch_size * num_par, self.dim]
        x = torch.cat([result, z], -1)  # [batch_size * num_par, self.dim * 2]
        proposal = self.mlp(x)  # [batch_size * num_par, dim_state]

        return proposal

