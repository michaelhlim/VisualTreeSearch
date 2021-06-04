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

        self.latent_dim = dlp.latent_dim
        self.img_size = sep.img_size

        dim_state = sep.dim_state
        in_channels = dlp.in_channels

        #self.embed_cond_var = nn.Linear(dim_state, self.img_size * self.img_size).to(self.device)
        self.embed_data = nn.Conv2d(in_channels, in_channels, kernel_size=1).to(self.device)

        modules = []
        hidden_dims = [32, 64, 128, 256, 512]
        self.hidden_dims = hidden_dims.copy()
        #in_channels += 1 # To account for the extra channel for the conditional variable

        # Build Encoder
        for h_dim in hidden_dims:
            if h_dim == hidden_dims[-1]:
                modules.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, out_channels=h_dim,
                                kernel_size= 3, stride= 1, padding  = 1),
                        #nn.BatchNorm2d(h_dim),
                        nn.LeakyReLU())
                )
            else:    
                modules.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, out_channels=h_dim,
                                kernel_size= 3, stride= 2, padding  = 1),
                        #nn.BatchNorm2d(h_dim),
                        nn.LeakyReLU())
                )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules).to(self.device)

        self.out = nn.Linear(hidden_dims[-1]*4, self.latent_dim)
        #self.fc_mu = nn.Linear(hidden_dims[-1]*4, self.latent_dim)
        #self.fc_var = nn.Linear(hidden_dims[-1]*4, self.latent_dim)


        # # Build Decoder
        # self.decoder_input = nn.Linear(self.latent_dim + dim_state, hidden_dims[-1] * 4).to(self.device)
        # hidden_dims.reverse()

        # modules = []
        # for i in range(len(hidden_dims) - 1):
        #     modules.append(
        #         nn.Sequential(
        #             nn.ConvTranspose2d(hidden_dims[i],
        #                                hidden_dims[i + 1],
        #                                kernel_size=3,
        #                                stride = 2,
        #                                padding=1,
        #                                output_padding=1),
        #             #nn.BatchNorm2d(hidden_dims[i + 1]),
        #             nn.LeakyReLU())
        #     )

        # self.decoder = nn.Sequential(*modules).to(self.device)

        # self.final_layer = nn.Sequential(
        #                     #nn.ConvTranspose2d(hidden_dims[-1],
        #                     #                   hidden_dims[-1],
        #                     #                   kernel_size=3,
        #                     #                   stride=2,
        #                     #                   padding=1,
        #                     #                   output_padding=1),
        #                     #nn.BatchNorm2d(hidden_dims[-1]),
        #                     nn.LeakyReLU(),
        #                     nn.Conv2d(hidden_dims[-1], out_channels= 3,
        #                               kernel_size= 3, padding= 1),
        #                     nn.Tanh()).to(self.device)
        
        # self.log_scale = nn.Parameter(torch.Tensor([0.0])).to(self.device)
        # self.calibration = dualsmc_lightdark_params.calibration

    def forward(self, input):
        # Runs encoder only

        #embedded_cond_var = self.embed_cond_var(state)  # [batch_size, 1024]
        #embedded_cond_var = embedded_cond_var.view(-1, self.img_size, self.img_size).unsqueeze(1)  # [batch_size, 1, 32, 32]
        embedded_input = self.embed_data(input)  # [batch_size, in_channels, 32, 32]
        x = embedded_input

        #x = torch.cat([embedded_input, embedded_cond_var], dim = 1)  # [batch_size, in_channels + 1, 32, 32]

        result = self.encoder(x)  # input [batch_size, 4, 32, 32]  result [batch_size, 512, 2, 2]
        result = torch.flatten(result, start_dim=1)  # [batch_size, 512*4]

        z = self.out(result)  # [batch_size, latent_dim]

        # # Split the result into mu and var components
        # # of the latent Gaussian distribution
        # mu = self.fc_mu(result)  # [batch_size, latent_dim]
        # log_var = self.fc_var(result) # [batch_size, latent_dim]

        # z = self.reparameterize(mu, log_var)

        # return [mu, log_var, z]

        return z

    
    # def reparameterize(self, mu, logvar):
    #     """
    #     Reparameterization trick to sample from N(mu, var) from
    #     N(0,1).
    #     :param mu: (Tensor) Mean of the latent Gaussian [B x D]
    #     :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
    #     :return: (Tensor) [B x D]
    #     """
    #     std = torch.exp(0.5 * logvar)  # [batch_size, latent_dim]
    #     eps = torch.randn_like(std)  # [batch_size, latent_dim]
    #     return eps * std + mu 
    
    # def decode(self, z):
    #     """
    #     Maps the given latent codes
    #     onto the image space.
    #     :param z: (Tensor) [B x D]
    #     :return: (Tensor) [B x C x H x W]
    #     """
    #     result = self.decoder_input(z)  # z [batch_size, latent_dim + dim_conditional_var] result [batch_size, 2048]
    #     result = result.view(-1, self.hidden_dims[-1], 2, 2)  # [batch_size, 512, 2, 2]
    #     result = self.decoder(result)  # [batch_size, 32, 32, 32]
    #     result = self.final_layer(result)  # [batch_size, 3, 32, 32]
    #     return result 

