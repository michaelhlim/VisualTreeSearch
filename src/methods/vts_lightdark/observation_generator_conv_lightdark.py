import torch
import torch.nn as nn

from configs.environments.stanford import *
from configs.solver.vts_lightdark import *
from utils.utils import *

sep = Stanford_Environment_Params()
vlp = VTS_LightDark_Params()


class ObservationGeneratorConv(nn.Module):
    def __init__(self, hidden_dims=None):
        super(ObservationGeneratorConv, self).__init__()

        self.in_channels = vlp.in_channels

        #self.embed_cond_var = nn.Linear(dim_conditional_var, img_size * img_size)
        self.embed_obs = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1)

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        
        self.hidden_dims = hidden_dims.copy()

        in_channels = self.in_channels

        # Build Encoder
        for h_dim in hidden_dims:
            if h_dim == hidden_dims[-1]:
                modules.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, out_channels=h_dim,
                                kernel_size=3, stride=1, padding=1),
                        nn.LeakyReLU(vlp.leak_rate))
                )
            else:    
                modules.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, out_channels=h_dim,
                                kernel_size=3, stride=2, padding=1),
                        nn.LeakyReLU(vlp.leak_rate))
                )
            in_channels = h_dim

        self.encoder_conv = nn.Sequential(*modules)

        hidden_dims.reverse()

        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    #nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder_conv = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            #nn.ConvTranspose2d(hidden_dims[-1],
                            #                   hidden_dims[-1],
                            #                   kernel_size=3,
                            #                   stride=2,
                            #                   padding=1,
                            #                   output_padding=1),
                            #nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels=self.in_channels,
                                      kernel_size=3, padding=1),
                            nn.Tanh())


    def encode(self, obs):
        embedded_input = self.embed_obs(obs)  # [batch_size, in_channels, 32, 32]
        intermediate = self.encoder_conv(embedded_input)  # [batch_size, 512, 2, 2]
        intermediate = torch.flatten(intermediate, start_dim=1)  # [batch_size, 512*4]

        return intermediate


    def decode(self, intermediate):
        intermediate = intermediate.view(-1, self.hidden_dims[-1], 2, 2)  # [batch_size, 512, 2, 2]
        result = self.decoder_conv(intermediate)  # [batch_size, 32, 32, 32]
        result = self.final_layer(result)  # [batch_size, 3, 32, 32]
        return result 


    


    
    
