import torch
import torch.nn as nn

from configs.environments.stanford import *
from configs.solver.dualsmc_lightdark import *
from src.methods.vts_lightdark.observation_encoder_lightdark import *
from utils.utils import *

sep = Stanford_Environment_Params()
vlp = VTS_LightDark_Params()



class MeasureNetwork(nn.Module):
    def __init__(self):
        super(MeasureNetwork, self).__init__()
        self.dim_m = 64 #16
        self.obs_encode_out = vlp.obs_encode_out
        self.dim_first_layer = vlp.dim_first_layer
        self.dim_lstm_hidden = vlp.dim_lstm_hidden
        self.num_lstm_layer = vlp.num_lstm_layer
        self.dim_state = sep.dim_state

        self.observation_encoder = ObservationEncoder()

        self.first_layer = nn.Sequential(
            nn.Linear(self.obs_encode_out, self.dim_first_layer),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(self.dim_first_layer, self.dim_lstm_hidden, self.num_lstm_layer)
        self.lstm_out = nn.Sequential(
            nn.Linear(self.dim_lstm_hidden, self.dim_m),
            nn.ReLU()
        )

        mlp_hunits = vlp.mlp_hunits
        self.mlp = nn.Sequential(
                nn.Linear(self.dim_m + self.dim_state + 1, mlp_hunits),
                nn.LeakyReLU(),
                nn.Linear(mlp_hunits, mlp_hunits),
                nn.LeakyReLU(),
                nn.Linear(mlp_hunits, mlp_hunits),
                nn.LeakyReLU(),
                nn.Linear(mlp_hunits, 1),
                nn.Sigmoid()
            )

    def m_model(self, state, orientation, obs, hidden, cell, num_par=vlp.num_par_pf):
        # state [batch_size * num_par, dim_state]
        # obs [batch_size, in_channels, img_size, img_size]
        # orientation [batch_size * num_par, 1]
        enc_obs = self.observation_encoder(obs)  # [batch_size, obs_enc_out]
        result = self.first_layer(enc_obs) # [batch_size, dim_first_layer]
        x = result.unsqueeze(0)  # [1, batch_size, dim_first_layer]
        x, (h, c) = self.lstm(x, (hidden, cell))  # x: [1, batch_size, dim_lstm_hidden]  # h and c same size as hidden, cell
        x = self.lstm_out(x[0])  # [batch_size, dim_m]
        x = x.repeat(num_par, 1)  # [batch_size * num_par, dim_m]        
        x = torch.cat((x, state, orientation), -1)  # [batch_size * num_par, dim_m + dim_state + 1]
        lik = self.mlp(x).view(-1, num_par)  # [batch_size, num_par]
        return lik, h, c



class ProposerNetwork(nn.Module):
    def __init__(self):
        super(ProposerNetwork, self).__init__()
        #self.dim = 64

        self.obs_encode_out = vlp.obs_encode_out
        self.dim_first_layer = vlp.dim_first_layer
        self.dim_state = sep.dim_state

        self.observation_encoder = ObservationEncoder()

        self.first_layer = nn.Sequential(
            nn.Linear(self.obs_encode_out, self.dim_first_layer),
            nn.ReLU()
        )

        mlp_hunits = vlp.mlp_hunits
        self.mlp = nn.Sequential(
                nn.Linear(self.dim_first_layer * 2 + 1, mlp_hunits),
                nn.LeakyReLU(),
                nn.Linear(mlp_hunits, mlp_hunits),
                nn.LeakyReLU(),
                nn.Linear(mlp_hunits, self.dim_state),
                nn.Sigmoid()
            )


    def forward(self, obs, orientation, num_par=vlp.num_par_pf):
        # obs [batch_size, in_channels, img_size, img_size]
        # orientation [batch_size, 1]
        enc_obs = self.observation_encoder(obs)  # enc_obs [batch_size, obs_encode_out]
        result = self.first_layer(enc_obs)  # [batch_size, dim_first_layer]
        result = result.repeat(num_par, 1)  # [batch_size * num_par, dim_first_layer]
        orientation = orientation.repeat(num_par, 1)  # [batch_size * num_par, 1]
        z = torch.randn_like(result)  # [batch_size * num_par, dim_first_layer]
        x = torch.cat([result, z, orientation], -1)  # [batch_size * num_par, dim_first_layer * 2 + 1]
        proposal = self.mlp(x)  # [batch_size * num_par, dim_state]

        return proposal

