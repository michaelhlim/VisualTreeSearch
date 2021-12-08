import torch
import torch.nn as nn

from configs.environments.stanford import *
from configs.solver.vts_lightdark import *
from utils.utils import *

sep = Stanford_Environment_Params()
vlp = VTS_LightDark_Params()


class ObservationEncoderIndependent(nn.Module):
    def __init__(self):
        super(ObservationEncoderIndependent, self).__init__()

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
        if vlp.final_img_size == 2: # Don't do the downsampling at the final layer
            for h_dim in hidden_dims:
                if h_dim == hidden_dims[-1]:
                    modules.append(
                        nn.Sequential(
                            nn.Conv2d(in_channels, out_channels=h_dim,
                                    kernel_size=3, stride=1, padding=1),
                            nn.LeakyReLU(self.leak_rate))
                    )
                else:    
                    modules.append(
                        nn.Sequential(
                            nn.Conv2d(in_channels, out_channels=h_dim,
                                    kernel_size=3, stride=2, padding=1),
                            nn.LeakyReLU(self.leak_rate))
                    )
                in_channels = h_dim
        else:
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
        mlp_modules.append(nn.Linear(sep.dim_state + 1 + vlp.obs_encode_out_conv, vlp.mlp_hunits_enc[0]))
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
            if i == 0:
                mlp_modules.append(nn.Linear(sep.dim_state + 1 + vlp.mlp_hunits_enc[ind], vlp.mlp_hunits_enc[ind-1]))
            else:
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

        # Assumes that there is at most one convolutional layer in the encoder with stride 1
        # and that layer is at the end
        # If there is one such layer, it must be because the "final image size" is 2 
        # (can't be downsampled any more)
        # If there are no such layers, this final_layer is needed to make the encoder 
        # and decoder perfectly symmetric
        if vlp.final_img_size == 2:
            self.final_layer = nn.Sequential(   
                            #nn.BatchNorm2d(hidden_dims[-1]),
                            #nn.LeakyReLU(self.leak_rate),
                            nn.Conv2d(hidden_dims[-1], out_channels=self.in_channels,
                                      kernel_size=3, padding=1))
                            #nn.Tanh())
        else:
            self.final_layer = nn.Sequential(  
                            nn.ConvTranspose2d(hidden_dims[-1],
                                              hidden_dims[-1],
                                              kernel_size=3,
                                              stride=2,
                                              padding=1,
                                              output_padding=1),
                            #nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(self.leak_rate),
                            nn.Conv2d(hidden_dims[-1], out_channels=self.in_channels,
                                      kernel_size=3, padding=1))
                            #nn.Tanh())


    def encode(self, state, orientation, obs):
        embedded_input = self.embed_obs(obs)  # [batch_size, in_channels, img_size, img_size]
        intermediate = self.encoder_conv(embedded_input)  # [batch_size, final_num_channels, final_img_size, final_img_size]  
        intermediate = torch.flatten(intermediate, start_dim=1)  # [batch_size, obs_encode_out_conv] 
        
        mlp_input = torch.cat([state, orientation, intermediate], -1)  # [batch_size, dim_state + 1 + obs_encode_out_conv]
        
        intermediate = self.encoder_mlp(mlp_input) # [batch_size, obs_encode_out]

        return intermediate


    def decode(self, state, orientation, intermediate):  
        mlp_input = torch.cat([state, orientation, intermediate], -1)  # [batch_size, dim_state + 1 + obs_encode_out]

        intermediate = self.decoder_mlp(mlp_input) # [batch_size, obs_encode_out_conv]

        intermediate = intermediate.view(-1, self.hidden_dims[-1], 4, 4)  # [batch_size, final_num_channels, final_img_size, final_img_size]  
        result = self.decoder_conv(intermediate)  # [batch_size, first num channels, img_size/2, img_size/2] OR [batch_size, first num channels, img_size, img_size]
        result = self.final_layer(result)  # [batch_size, in_channels, img_size, img_size]
        return result 


    


    
    
