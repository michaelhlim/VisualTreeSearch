import torch
import torch.nn as nn

from configs.environments.stanford import *
from configs.solver.dualsmc_lightdark import *
from utils.utils import *

sep = Stanford_Environment_Params()
dlp = DualSMC_LightDark_Params()


class ObservationEncoder(nn.Module):
    def __init__(self):
        super(ObservationEncoder, self).__init__()

        self.device = dlp.device

        self.in_channels = dlp.in_channels

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=16,
                      kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(2),
            #nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32,
                      kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(2),
            #nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=3, stride=2, padding=1),
            nn.Dropout(0.2),
            #nn.ReLU()
        )
        

    def forward(self, input):
        result = self.encoder(input)  # input [batch_size, in_channels, 32, 32]  result [batch_size, 64, 1, 1]
        result = torch.flatten(result, start_dim=1)  # [batch_size, 64]

        return result

    
    
