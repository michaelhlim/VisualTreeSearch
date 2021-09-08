# From https://github.com/AntixK/PyTorch-VAE/blob/master/models/cvae.py

import torch
from torch import nn
from torch.nn import functional as F

from src.methods.vts_lightdark.observation_encoder_lightdark import *
from src.methods.vts_lightdark.observation_generator_conv_lightdark import *

from configs.environments.stanford import *
from configs.solver.vts_lightdark import *

sep = Stanford_Environment_Params()
vlp = VTS_LightDark_Params()



class ObservationGenerator(nn.Module):

    def __init__(self, observation_encoder):

        super(ObservationGenerator, self).__init__()

        self.latent_dim = vlp.latent_dim
        self.mlp_hunits = vlp.mlp_hunits_g
        self.leak_rate = vlp.leak_rate
        self.calibration = vlp.calibration
        self.beta = vlp.beta 

        #self.observation_encoder = ObservationEncoder()
        #self.conv = ObservationGeneratorConv()
        self.conv = observation_encoder

        encoder_modules = []
        encoder_modules.append(nn.Linear(sep.dim_state + 1 + vlp.obs_encode_out, self.mlp_hunits))
        encoder_modules.append(nn.LeakyReLU(self.leak_rate))
        for i in range(vlp.num_layers - 2):
            encoder_modules.append(nn.Linear(self.mlp_hunits, self.mlp_hunits))
            encoder_modules.append(nn.LeakyReLU(self.leak_rate))
        encoder_modules.append(nn.Linear(self.mlp_hunits, self.latent_dim))
        encoder_modules.append(nn.LeakyReLU(self.leak_rate))

        self.encoder = nn.Sequential(*encoder_modules)

        # distribution parameters
        self.fc_mu = nn.Linear(self.latent_dim, self.latent_dim)
        self.fc_var = nn.Linear(self.latent_dim, self.latent_dim)

        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

        decoder_modules = []
        decoder_modules.append(nn.Linear(sep.dim_state + 1 + self.latent_dim, self.mlp_hunits))
        decoder_modules.append(nn.LeakyReLU(self.leak_rate))
        for i in range(vlp.num_layers - 2):
            decoder_modules.append(nn.Linear(self.mlp_hunits, self.mlp_hunits))
            decoder_modules.append(nn.LeakyReLU(self.leak_rate))
        decoder_modules.append(nn.Linear(self.mlp_hunits, vlp.obs_encode_out))
        decoder_modules.append(nn.LeakyReLU(self.leak_rate))

        self.decoder = nn.Sequential(*decoder_modules)


    def encode(self, conditional_input, enc_obs_batch):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """

        encoder_input = torch.cat([conditional_input, enc_obs_batch], -1)  # [batch_size, dim_state + 1 + obs_encode_out]
        obs_encoded = self.encoder(encoder_input)  # [batch_size, latent_dim]
        mu, log_var = self.fc_mu(obs_encoded), self.fc_var(obs_encoded)  # [batch_size, latent_dim]

        return [mu, log_var]


    def decode(self, conditional_input, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        decoder_input = torch.cat([conditional_input, z], -1)  # [batch_size, latent_dim + dim_state + 1]
        enc_obs_hat = self.decoder(decoder_input)  # [batch_size, obs_encode_out]
        return enc_obs_hat


    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)  # [batch_size, latent_dim]
        eps = torch.randn_like(std)  # [batch_size, latent_dim]
        return eps * std + mu 


    def forward(self, conditional_input, enc_obs_batch):
        #enc_obs_batch = self.observation_encoder(enc_obs_batch, normalize=True)
        #intermediate = self.conv.encode(enc_obs_batch)  # [batch_size, obs_encode_out]
        with torch.no_grad():
            intermediate = self.conv.encode(enc_obs_batch)  # [batch_size, obs_encode_out]
            # Normalizing the output of the observation encoder
            intermediate = (intermediate - torch.mean(intermediate, -1, True))/torch.std(intermediate, -1, keepdim=True)

        mu, log_var = self.encode(conditional_input, intermediate)  # [batch_size, latent_dim]
        #mu, log_var = self.encode(conditional_input, enc_obs_batch)  # [batch_size, latent_dim]
        z = self.reparameterize(mu, log_var)  # [batch_size, latent_dim]

        recons = self.decode(conditional_input, z)
        recons = self.conv.decode(recons)

        #return [self.decode(conditional_input, z), enc_obs_batch, mu, log_var]
        return [recons, enc_obs_batch, mu, log_var]


    def gaussian_likelihood(self, x_hat, logscale, x):
        scale = torch.exp(logscale)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)

        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)
        log_pxz = log_pxz.reshape(log_pxz.shape[0], -1)
        return log_pxz.mean(dim=-1)


    def loss_function(self, *args):
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        enc_obs = args[1]
        mu = args[2]
        log_var = args[3]

        recons_loss = self.gaussian_likelihood(recons, self.log_scale, enc_obs).mean() 
        #recons_loss = -F.mse_loss(recons, input)

        if self.calibration:
            log_sigma = ((enc_obs - recons) ** 2).mean().sqrt().log()

            def softclip(tensor, min):
                """ Clips the tensor values at the minimum value min in a softway. Taken from Handful of Trials """
                result_tensor = min + F.softplus(tensor - min)

                return result_tensor

            log_sigma = softclip(log_sigma, -6)
            rec = self.gaussian_likelihood(recons, log_sigma, enc_obs)  # [batch_size]
            recons_loss = rec.mean()


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = -recons_loss + self.beta * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':-recons_loss, 'KLD':kld_loss}


    def sample(self, num_samples, conditional_input):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim).to(vlp.device)

        samples = self.decode(conditional_input, z)
        return samples
    

    def sample_interpolate(self, num_interps, conditional_var):
        return
        z_start = torch.randn(1, self.latent_dim)
        z_direction = torch.randn(1, self.latent_dim)

        samples_arr = []

        z = z_start
        for i in range(num_interps):
            inp = torch.cat([z, conditional_var], dim=1)
            samples = self.decode(inp)
            samples_arr.append(samples)
            z += z_direction
        
        return samples_arr, z_start, z_direction



    # def generate(self, state, enc_obs):
    #     """
    #     Given an input image x, returns the reconstructed image
    #     :param x: (Tensor) [B x C x H x W]
    #     :return: (Tensor) [B x C x H x W]
    #     """

    #     return self.forward(state, enc_obs)[0]




