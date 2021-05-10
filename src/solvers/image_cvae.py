# From https://github.com/AntixK/PyTorch-VAE/blob/master/models/cvae.py

import torch
from torch import nn
from torch.nn import functional as F
#from .types_ import *



class ConditionalVAE(nn.Module):

    def __init__(self, args):

        super(ConditionalVAE, self).__init__()

        if 'hidden_dims' in args:
            hidden_dims = args['hidden_dims']
        else:
            hidden_dims = None

        in_channels = args['in_channels']
        dim_conditional_var = args['dim_conditional_var']
        latent_dim = args['latent_dim']
        img_size = args['img_size']
        calibration = args['calibration']
        device = args['device']

        self.device = device

        self.latent_dim = latent_dim
        self.img_size = img_size

        self.embed_cond_var = nn.Linear(dim_conditional_var, img_size * img_size).to(self.device)
        self.embed_data = nn.Conv2d(in_channels, in_channels, kernel_size=1).to(self.device)

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        
        self.hidden_dims = hidden_dims.copy()

        in_channels += 1 # To account for the extra channel for the conditional variable

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    #nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules).to(self.device)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim).to(self.device)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim).to(self.device)
        #self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        #self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim + dim_conditional_var, hidden_dims[-1] * 4).to(self.device)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    #nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules).to(self.device)

        self.final_layer = nn.Sequential(
                            #nn.ConvTranspose2d(hidden_dims[-1],
                            #                   hidden_dims[-1],
                            #                   kernel_size=3,
                            #                   stride=2,
                            #                   padding=1,
                            #                   output_padding=1),
                            #nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh()).to(self.device)
        
        self.log_scale = nn.Parameter(torch.Tensor([0.0])).to(self.device)
        self.calibration = calibration


    def configure_optimizers(self, lr):
        return torch.optim.Adam(self.parameters(), lr=lr)


    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)  # input [batch_size, 4, 32, 32]  result [batch_size, 512, 1, 1]
        result = torch.flatten(result, start_dim=1)  # [batch_size, 512]

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)  # [batch_size, latent_dim]
        log_var = self.fc_var(result) # [batch_size, latent_dim]

        return [mu, log_var]

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)  # z [batch_size, latent_dim + dim_conditional_var] result [batch_size, 2048]
        result = result.view(-1, self.hidden_dims[-1], 2, 2)  # [batch_size, 512, 2, 2]
        result = self.decoder(result)  # [batch_size, 32, 32, 32]
        result = self.final_layer(result)  # [batch_size, 3, 32, 32]
        return result 

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

    def forward(self, input, conditional_var):
        # input [batch_size, in_channels, 32, 32]
        # conditional_var [batch_size, 1, dim_conditional_var]
        embedded_cond_var = self.embed_cond_var(conditional_var)  # [batch_size, 1, 1024]
        embedded_cond_var = embedded_cond_var.view(-1, self.img_size, self.img_size).unsqueeze(1)  # [batch_size, 1, 32, 32]
        embedded_input = self.embed_data(input)  # [batch_size, in_channels, 32, 32]

        x = torch.cat([embedded_input, embedded_cond_var], dim = 1)  # [batch_size, in_channels + 1, 32, 32]
        mu, log_var = self.encode(x)  # [batch_size, latent_dim]

        z = self.reparameterize(mu, log_var)  # [batch_size, latent_dim]

        conditional_var = conditional_var.squeeze(1)  # [batch_size, dim_conditional_var]
        z = torch.cat([z, conditional_var], dim = 1)  # [batch_size, latent_dim + dim_conditional_var]
        return  [self.decode(z), input, mu, log_var]   # decode(z)  [batch_size, in_channels, 32, 32]
    

    def gaussian_likelihood(self, x_hat, logscale, x):
        scale = torch.exp(logscale)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)

        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)
        log_pxz = log_pxz.reshape(log_pxz.shape[0], -1)
        return log_pxz.mean(dim=-1)


    def loss_function(self, beta, *args):
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        recons_loss = self.gaussian_likelihood(recons, self.log_scale, input).mean() 
        #recons_loss = -F.mse_loss(recons, input)

        if self.calibration:
            log_sigma = ((input - recons) ** 2).mean().sqrt().log()

            def softclip(tensor, min):
                """ Clips the tensor values at the minimum value min in a softway. Taken from Handful of Trials """
                result_tensor = min + F.softplus(tensor - min)

                return result_tensor

            log_sigma = softclip(log_sigma, -6)
            rec = self.gaussian_likelihood(recons, log_sigma, input)  # [batch_size]
            recons_loss = rec.mean()


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = -recons_loss + beta * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':-recons_loss, 'KLD':kld_loss}

    def sample(self, num_samples, conditional_var):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(self.device)

        z = torch.cat([z, conditional_var], dim=1)
        samples = self.decode(z)
        return samples

    def generate(self, x, conditional_var):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x, conditional_var)[0]




