# author: @sdeglurkar, @jatucker4, @michaelhlim

import torch
import torch.nn as nn
import torch.nn.functional as F


# Observation Predictor
class ObsPredictorNetwork(nn.Module):
    def __init__(self, args):
        super(ObsPredictorNetwork, self).__init__()
        
        self.in_channels = args['in_channels']
        self.out_dim = args['state_enc_out_dim']
        self.dim_state = args['dim_state']
        self.device = args['device']

        hunits = 1024 

        self.state_encode = nn.Sequential(nn.Linear(self.dim_state, hunits),
                                          nn.ReLU(),
                                          nn.Linear(hunits, hunits),
                                          nn.ReLU(),
                                          nn.Linear(hunits, hunits),
                                          nn.ReLU(),
                                          nn.Linear(hunits, self.out_dim),
                                          nn.ReLU()).to(self.device)

        self.hdims = [32, 64, 128, 256, 512]
        self.op_input = nn.Linear(2 * self.out_dim, self.hdims[-1] * 4).to(self.device)
        
        self.op_net = nn.Sequential(nn.ConvTranspose2d(self.hdims[-1],
                                       self.hdims[-2],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                                    #nn.BatchNorm2d(self.hdims[-2]),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(self.hdims[-2],
                                       self.hdims[-3],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                                    #nn.BatchNorm2d(self.hdims[-3]),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(self.hdims[-3],
                                       self.hdims[-4],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                                    #nn.BatchNorm2d(self.hdims[-4]),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(self.hdims[-4],
                                       self.hdims[-5],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                                    #nn.BatchNorm2d(self.hdims[-5]),   
                                    nn.ReLU(),
                                    nn.Conv2d(self.hdims[-5], out_channels=self.in_channels,
                                      kernel_size= 3, padding= 1),
                                    #nn.Dropout(0.2),
                                    nn.Tanh()).to(self.device)


    def forward(self, state):
        # state: (batch_size, dim_state)
        state_enc = self.state_encode(state)  # (batch_size, state_enc_out_dim)
        z = torch.randn_like(state_enc)  # (batch_size, state_enc_out_dim)
        x = torch.cat([state_enc, z], -1)  # (batch_size, state_enc_out_dim * 2)
        result = self.op_input(x)  # (batch_size, 2048)
        result = result.view(-1, self.hdims[-1], 2, 2)  # (batch_size, 512, 2, 2)
        obs_prediction = self.op_net(result)  # (batch_size, in_channels, 32, 32)
        return obs_prediction


# Observation Model
class MeasureNetwork(nn.Module):
    def __init__(self, args):
        super(MeasureNetwork, self).__init__()

        self.in_channels = args['in_channels']
        self.dim_state = args['dim_state']
        self.device = args['device']

        self.hdims = [16, 32, 64]
        self.obs_encode = nn.Sequential(nn.Conv2d(self.in_channels, out_channels=self.hdims[0],
                                                    kernel_size=3, stride=2, padding=1),
                                          nn.MaxPool2d(kernel_size=3, stride=2),
                                          nn.BatchNorm2d(self.hdims[0]),
                                          nn.Conv2d(in_channels=self.hdims[0], out_channels=self.hdims[1],
                                                    kernel_size=3, stride=2, padding=1),
                                          nn.MaxPool2d(kernel_size=3, stride=2),
                                          nn.BatchNorm2d(self.hdims[1]),
                                          nn.Conv2d(in_channels=self.hdims[1], out_channels=self.hdims[2],
                                                    kernel_size=3, stride=2, padding=1),
                                          nn.BatchNorm2d(self.hdims[2]),
                                          nn.Dropout(0.2)).to(self.device)
        hunits = 128

        concat_dim = self.hdims[-1] + self.dim_state
        self.m_net = nn.Sequential(nn.Linear(concat_dim, hunits),
                                   nn.ReLU(),
                                   nn.Linear(hunits, hunits),
                                   nn.ReLU(),
                                   nn.Linear(hunits, hunits),
                                   nn.ReLU(),
                                   nn.Linear(hunits, 1),
                                   nn.Sigmoid()).to(self.device)


    def forward(self, state, obs):
        # state: (batch_size, dim_state)
        # obs: (batch_size, in_channels, 32, 32)
        obs_enc = self.obs_encode(obs)  # (batch_size, 64, 1, 1)
        obs_enc = obs_enc.view(-1, self.hdims[-1])  # (batch_size, 64)
        x = torch.cat((obs_enc, state), -1)  # (batch_size, 64 + dim_state)
        lik = self.m_net(x)  # (batch_size, 1)
        return lik







