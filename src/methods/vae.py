import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from torch import nn
from torch.nn import functional as F

DIM_STATE = 2
DIM_OBS = 2
DIM_HIDDEN = 256

BATCH_SIZE = 64

def make_simple_batch(batch_size):
    states_batch = np.random.rand(batch_size, 2)
    obs_batch = states_batch + np.random.normal(0, 0.1, (batch_size, 2))
    states_batch = torch.from_numpy(states_batch).float()
    obs_batch = torch.from_numpy(obs_batch).float()

    return states_batch, obs_batch

class VAE(nn.Module):
    def __init__(self, enc_out_dim=64, latent_dim=64):
        super(VAE, self).__init__()

        # encoder, decoder
        self.encoder = nn.Sequential(
            nn.Linear(DIM_OBS, DIM_HIDDEN),
            #nn.BatchNorm1d(DIM_HIDDEN, 0.8),
            #nn.LeakyReLU(0.2),
            nn.ReLU(),
            nn.Linear(DIM_HIDDEN, DIM_HIDDEN),
            #nn.BatchNorm1d(DIM_HIDDEN, 0.8),
            #nn.LeakyReLU(0.2),
            nn.ReLU(),
            nn.Linear(DIM_HIDDEN, DIM_HIDDEN),
            #nn.BatchNorm1d(DIM_HIDDEN, 0.8),
            #nn.LeakyReLU(0.2),
            nn.ReLU(),
            nn.Linear(DIM_HIDDEN, DIM_HIDDEN),
            #nn.BatchNorm1d(DIM_HIDDEN, 0.8),
            #nn.LeakyReLU(0.2),
            nn.ReLU(),
            nn.Linear(DIM_HIDDEN, enc_out_dim),
            #nn.BatchNorm1d(enc_out_dim, 0.8),
            #nn.LeakyReLU(0.2))
            nn.ReLU())
        self.decoder = nn.Sequential(
            nn.Linear(DIM_STATE + latent_dim, DIM_HIDDEN),
            #nn.BatchNorm1d(DIM_HIDDEN, 0.8),
            #nn.LeakyReLU(0.2),
            nn.ReLU(),
            nn.Linear(DIM_HIDDEN, DIM_HIDDEN),
            #nn.BatchNorm1d(DIM_HIDDEN, 0.8),
            #nn.LeakyReLU(0.2),
            nn.ReLU(),
            nn.Linear(DIM_HIDDEN, DIM_HIDDEN),
            #nn.BatchNorm1d(DIM_HIDDEN, 0.8),
            #nn.LeakyReLU(0.2),
            nn.ReLU(),
            nn.Linear(DIM_HIDDEN, DIM_HIDDEN),
            #nn.BatchNorm1d(DIM_HIDDEN, 0.8),
            #nn.LeakyReLU(0.2),
            nn.ReLU(),
            nn.Linear(DIM_HIDDEN, DIM_STATE),
            #nn.BatchNorm1d(DIM_STATE, 0.8),
            #nn.LeakyReLU(0.2))
            nn.ReLU())


        # distribution parameters
        self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self.fc_var = nn.Linear(enc_out_dim, latent_dim)

        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def gaussian_likelihood(self, x_hat, logscale, x):
        scale = torch.exp(logscale)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)

        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)
        return log_pxz.sum(dim=-1)

    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl

    def training_step(self, state_batch, obs_batch):
        # encode x to get the mu and variance parameters
        obs_encoded = self.encoder(obs_batch)
        #obs_encoded = self.encoder(state_batch)
        mu, log_var = self.fc_mu(obs_encoded), self.fc_var(obs_encoded)

        # sample z from q
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        decoder_input = torch.cat([state_batch, z], -1)
        #decoder_input = torch.cat([obs_batch, z], -1)

        # decoded
        obs_hat = self.decoder(decoder_input)

        # reconstruction loss
        recon_loss = self.gaussian_likelihood(obs_hat, self.log_scale, obs_batch)

        # kl
        beta = 1
        kl = self.kl_divergence(z, mu, std)
        #kl = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        # elbo
        elbo = (beta*kl - recon_loss)
        elbo = elbo.mean()

        return elbo


vae = VAE()
num_training_steps = 1000

t0 = time.time()
optimizer = vae.configure_optimizers()
for step in range(num_training_steps):
    print(step)
    state_batch, obs_batch = make_simple_batch(BATCH_SIZE)
    optimizer.zero_grad()
    loss = vae.training_step(state_batch, obs_batch)
    loss.backward()
    optimizer.step()
t1 = time.time()
print("Finished training. That took", t1 - t0, "seconds.")

for j in range(2):
    state = torch.from_numpy(np.random.rand(2)).reshape((1, 2))
    print("STATE", state)
    states_batch = torch.cat(BATCH_SIZE * [state]).float()
    obs_batch = states_batch.numpy() + np.random.normal(0, 0.1, (BATCH_SIZE, 2))
    #obs_encoded = vae.encoder(states_batch)
    obs_encoded = vae.encoder(torch.from_numpy(obs_batch).float())
    mu, log_var = vae.fc_mu(obs_encoded), vae.fc_var(obs_encoded)
    std = torch.exp(log_var / 2)
    q = torch.distributions.Normal(mu, std)
    z = q.rsample()
    decoder_input = torch.cat([states_batch, z], -1)
    #decoder_input = torch.cat([torch.from_numpy(obs_batch).float(), z], -1)

    obs_hat = vae.decoder(decoder_input.detach())

    obs_predicted_mean = np.mean(obs_hat[:, :2].detach().numpy(), axis=0)
    obs_predicted_std = np.std(obs_hat[:, :2].detach().numpy(), axis=0)
    print("OBS_PREDICTED_MEAN\n", obs_predicted_mean)
    print("OBS_PREDICTED_STD\n", obs_predicted_std)

    plt.scatter([state[0][0]], [state[0][1]], color='k')
    plt.scatter([obs[0] for obs in obs_batch], [obs[1] for obs in obs_batch], color='g')
    plt.scatter([obs[0] for obs in obs_hat.detach().numpy()],
                [obs[1] for obs in obs_hat.detach().numpy()], color='r')
    plt.show()



