import torch
import torch.nn as nn

from configs.environments.stanford import *
from configs.solver.dualsmc_lightdark import *
from src.methods.vts_lightdark.observation_encoder_lightdark import *
from src.methods.vts_lightdark.observation_generator_conv_lightdark import * 
from utils.utils import *

sep = Stanford_Environment_Params()
vlp = VTS_LightDark_Params()


class MeasureNetwork(nn.Module):
    def __init__(self, observation_encoder):
        super(MeasureNetwork, self).__init__()
        self.dim_m = vlp.dim_m #64 #16
        self.obs_encode_out = vlp.obs_encode_out
        self.dim_first_layer = vlp.dim_first_layer
        self.dim_lstm_hidden = vlp.dim_lstm_hidden
        self.dim_state = sep.dim_state

        self.observation_encoder = observation_encoder

        self.first_layer = nn.Sequential(
            nn.Linear(self.obs_encode_out, self.dim_first_layer),
            nn.ReLU()
        )

        self.lstm_replace = nn.Sequential(
            nn.Linear(self.dim_first_layer, self.dim_lstm_hidden),
            nn.ReLU()
        )
        self.lstm_out = nn.Sequential(
            nn.Linear(self.dim_lstm_hidden, self.dim_m),
            nn.ReLU()
        )

        mlp_hunits = vlp.mlp_hunits_zp
        self.mlp = nn.Sequential(
                nn.Linear(self.dim_m + self.dim_state + 1, mlp_hunits),
                #nn.LeakyReLU(),
                nn.ReLU(),
                nn.Linear(mlp_hunits, mlp_hunits),
                #nn.LeakyReLU(),
                nn.ReLU(),
                nn.Linear(mlp_hunits, mlp_hunits),
                #nn.LeakyReLU(),
                nn.ReLU(),
                nn.Linear(mlp_hunits, 1),
                nn.Sigmoid()
            )

    def m_model(self, state, orientation, obs, num_par=vlp.num_par_pf, obs_is_encoded=False, indep_enc=False):
        # state [batch_size * num_par, dim_state]
        # obs [batch_size, in_channels, img_size, img_size]
        # orientation [batch_size * num_par, 1]
        if not obs_is_encoded:
            if indep_enc:
                with torch.no_grad():
                    enc_obs = self.observation_encoder.encode(obs)  # [batch_size, obs_enc_out]
                    # Normalizing the output of the observation encoder
                    enc_obs = (enc_obs - torch.mean(enc_obs, -1, True))/torch.std(enc_obs, -1, keepdim=True)
            else:
                enc_obs = self.observation_encoder.encode(obs)  # [batch_size, obs_enc_out]
                # Normalizing the output of the observation encoder
                enc_obs = (enc_obs - torch.mean(enc_obs, -1, True))/torch.std(enc_obs, -1, keepdim=True)
        
        else:
            enc_obs = obs
        result = self.first_layer(enc_obs) # [batch_size, dim_first_layer]
        x = self.lstm_replace(result)  # [batch_size, dim_lstm_hidden]
        x = self.lstm_out(x)  # [batch_size, dim_m]
        x = x.repeat(num_par, 1)  # [batch_size * num_par, dim_m]        
        x = torch.cat((x, state, orientation), -1)  # [batch_size * num_par, dim_m + dim_state + 1]
        lik = self.mlp(x).view(-1, num_par)  # [batch_size, num_par]
        return lik



class ProposerNetwork(nn.Module):
    def __init__(self, observation_encoder):
        super(ProposerNetwork, self).__init__()
        #self.dim = 64

        self.obs_encode_out = vlp.obs_encode_out
        self.dim_first_layer = vlp.dim_first_layer
        self.dim_state = sep.dim_state

        self.observation_encoder = observation_encoder

        self.first_layer = nn.Sequential(
            nn.Linear(self.obs_encode_out, self.dim_first_layer),
            nn.ReLU()
        )

        mlp_hunits = vlp.mlp_hunits_zp
        self.mlp = nn.Sequential(
                nn.Linear(self.dim_first_layer * 2 + 1, mlp_hunits),
                #nn.LeakyReLU(),
                nn.ReLU(),
                nn.Linear(mlp_hunits, mlp_hunits),
                #nn.LeakyReLU(),
                nn.ReLU(),
                nn.Linear(mlp_hunits, self.dim_state),
                nn.ReLU()
                #nn.Sigmoid()
            )


    def forward(self, obs, orientation, num_par=vlp.num_par_pf, obs_is_encoded=False, indep_enc=False):
        # obs [batch_size, in_channels, img_size, img_size]
        # orientation [batch_size, 1]

        if not obs_is_encoded:
            if indep_enc:
                with torch.no_grad():
                    enc_obs = self.observation_encoder.encode(obs)  # [batch_size, obs_enc_out]
                    # Normalizing the output of the observation encoder
                    enc_obs = (enc_obs - torch.mean(enc_obs, -1, True))/torch.std(enc_obs, -1, keepdim=True)
            else:
                enc_obs = self.observation_encoder.encode(obs)  # [batch_size, obs_enc_out]
                # Normalizing the output of the observation encoder
                enc_obs = (enc_obs - torch.mean(enc_obs, -1, True))/torch.std(enc_obs, -1, keepdim=True)

        else:
            enc_obs = obs
        
        
        result = self.first_layer(enc_obs)  # [batch_size, dim_first_layer]
        result = result.repeat(num_par, 1)  # [batch_size * num_par, dim_first_layer]
        orientation = orientation.repeat(num_par, 1)  # [batch_size * num_par, 1]  
        z = torch.randn_like(result)  # [batch_size * num_par, dim_first_layer]
        x = torch.cat([result, z, orientation], -1)  # [batch_size * num_par, dim_first_layer * 2 + 1]
        proposal = self.mlp(x)  # [batch_size * num_par, dim_state]

        return proposal

