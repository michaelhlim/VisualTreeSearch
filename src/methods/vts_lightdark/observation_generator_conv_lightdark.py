import torch
import torch.nn as nn

from configs.environments.stanford import *
from configs.solver.vts_lightdark import *
from utils.utils import *

sep = Stanford_Environment_Params()
vlp = VTS_LightDark_Params()


class ObservationGeneratorConv(nn.Module):
    def __init__(self):
        super(ObservationGeneratorConv, self).__init__()

        self.in_channels = vlp.in_channels
        self.leak_rate = vlp.leak_rate_enc

        #self.embed_cond_var = nn.Linear(dim_conditional_var, img_size * img_size)
        self.embed_obs = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1)

        ## Encoder Convolutional Layers

        modules = []
        hidden_dims = vlp.hidden_dims_generator_conv
        
        self.hidden_dims = hidden_dims.copy()

        in_channels = self.in_channels

        # Build Encoder
        # for h_dim in hidden_dims:
        #     if h_dim == hidden_dims[-1]:
        #         modules.append(
        #             nn.Sequential(
        #                 nn.Conv2d(in_channels, out_channels=h_dim,
        #                         kernel_size=3, stride=1, padding=1),
        #                 nn.LeakyReLU(self.leak_rate))
        #         )
        #     else:    
        #         modules.append(
        #             nn.Sequential(
        #                 nn.Conv2d(in_channels, out_channels=h_dim,
        #                         kernel_size=3, stride=2, padding=1),
        #                 nn.LeakyReLU(self.leak_rate))
        #         )
        #     in_channels = h_dim

        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                            kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU(self.leak_rate))
            )   
  
            in_channels = h_dim

        self.encoder_conv = nn.Sequential(*modules)


        ## Encoder MLP Layers

        mlp_modules = []
        mlp_modules.append(nn.Linear(vlp.obs_encode_out_conv, vlp.mlp_hunits_enc[0]))
        mlp_modules.append(nn.LeakyReLU(self.leak_rate))
        for i in range(len(vlp.mlp_hunits_enc) - 1):
            ind = i + 1
            mlp_modules.append(nn.Linear(vlp.mlp_hunits_enc[ind-1], vlp.mlp_hunits_enc[ind]))
            mlp_modules.append(nn.LeakyReLU(self.leak_rate))
        
        # mlp_modules.append(nn.Linear(vlp.obs_encode_out_conv, vlp.mlp_hunits_enc1))
        # mlp_modules.append(nn.LeakyReLU(self.leak_rate))
        # mlp_modules.append(nn.Linear(vlp.mlp_hunits_enc1, vlp.mlp_hunits_enc2))
        # mlp_modules.append(nn.LeakyReLU(self.leak_rate))
        # mlp_modules.append(nn.Linear(vlp.mlp_hunits_enc2, vlp.mlp_hunits_enc3)) # mlp_hunits_enc3 = obs_encode_out
        # mlp_modules.append(nn.LeakyReLU(self.leak_rate))

        self.encoder_mlp = nn.Sequential(*mlp_modules)


        ## Decoder MLP Layers

        mlp_modules = []
        for i in range(len(vlp.mlp_hunits_enc) - 1):
            ind = len(vlp.mlp_hunits_enc) - i - 1
            mlp_modules.append(nn.Linear(vlp.mlp_hunits_enc[ind], vlp.mlp_hunits_enc[ind-1]))
            mlp_modules.append(nn.LeakyReLU(self.leak_rate))
        ind -= 1
        mlp_modules.append(nn.Linear(vlp.mlp_hunits_enc[ind], vlp.obs_encode_out_conv))
        mlp_modules.append(nn.LeakyReLU(self.leak_rate))

        # mlp_modules.append(nn.Linear(vlp.mlp_hunits_enc3, vlp.mlp_hunits_enc2))
        # mlp_modules.append(nn.LeakyReLU(self.leak_rate))
        # mlp_modules.append(nn.Linear(vlp.mlp_hunits_enc2, vlp.mlp_hunits_enc1))
        # mlp_modules.append(nn.LeakyReLU(self.leak_rate))
        # mlp_modules.append(nn.Linear(vlp.mlp_hunits_enc1, vlp.obs_encode_out_conv))
        # mlp_modules.append(nn.LeakyReLU(self.leak_rate))

        self.decoder_mlp = nn.Sequential(*mlp_modules)


        ## Decoder Convolutional Layers
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
                    nn.LeakyReLU(self.leak_rate))
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
                            nn.LeakyReLU(self.leak_rate),
                            nn.Conv2d(hidden_dims[-1], out_channels=self.in_channels,
                                      kernel_size=3, padding=1))
                            #nn.Tanh())


    def encode(self, obs):
        embedded_input = self.embed_obs(obs)  # [batch_size, in_channels, 32, 32]
        intermediate = self.encoder_conv(embedded_input)  # [batch_size, 64, 4, 4]  # [batch_size, 512, 2, 2]
        intermediate = torch.flatten(intermediate, start_dim=1)  # [batch_size, 64*16]  # [batch_size, 512*4]

        intermediate = self.encoder_mlp(intermediate) # [batch_size, 256]

        return intermediate


    def decode(self, intermediate):
        intermediate = self.decoder_mlp(intermediate) # [batch_size, 512*4]

        intermediate = intermediate.view(-1, self.hidden_dims[-1], 2, 2)  # [batch_size, 512, 2, 2]
        result = self.decoder_conv(intermediate)  # [batch_size, 32, 32, 32]
        result = self.final_layer(result)  # [batch_size, 3, 32, 32]
        return result 


    


    
    
