import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from env import Environment
import glob
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
import torch
from torch import nn
from torch.nn import functional as F

DIM_STATE = 2
DIM_OBS = 4
DIM_HIDDEN = 256
LATENT_DIM = 64

BATCH_SIZE = 64
LR = 1e-4
BETA = 0.5
LEAK = 0.9

NUM_TRAINING_STEPS = 5000

pretrain_4d = False
calibration = True
learning_rate_scheduling = False


def make_simple_batch(batch_size):
    states_batch = np.random.rand(batch_size, DIM_STATE)
    obs_batch = states_batch + np.random.normal(0, 0.1, (batch_size, DIM_OBS))
    states_batch = torch.from_numpy(states_batch).float()
    obs_batch = torch.from_numpy(obs_batch).float()

    return states_batch, obs_batch

def make_batch(batch_size):
    if pretrain_4d:
        states_batch = []
        obs_batch = []
        for _ in range(batch_size):
            state = np.random.rand(DIM_STATE)
            obs = np.hstack([state, [0., 0.]]) + np.random.normal(0, 0.1, DIM_OBS)
            #obs = np.hstack([state, np.random.rand(DIM_STATE)]) + np.random.normal(0, 0.1, DIM_OBS)
            states_batch.append(state)
            obs_batch.append(obs)
        states_batch = torch.from_numpy(np.array(states_batch)).float()
        obs_batch = torch.from_numpy(np.array(obs_batch)).float()

        return states_batch, obs_batch

    else:
        states_batch = []
        obs_batch = []
        for _ in range(batch_size):
            env = Environment()
            state = env.state
            obs = env.get_observation()
            states_batch.append(state)
            obs_batch.append(obs)
        states_batch = torch.from_numpy(np.array(states_batch)).float()
        obs_batch = torch.from_numpy(np.array(obs_batch)).float()

        return states_batch, obs_batch


class VAE(nn.Module):
    def __init__(self, enc_out_dim=64, latent_dim=LATENT_DIM):
        super(VAE, self).__init__()

        # encoder, decoder
        self.encoder = nn.Sequential(
            nn.Linear(DIM_STATE + DIM_OBS, DIM_HIDDEN),
            #nn.BatchNorm1d(DIM_HIDDEN, 0.8),
            nn.LeakyReLU(LEAK),
            #nn.ReLU(),
            nn.Linear(DIM_HIDDEN, DIM_HIDDEN),
            #nn.BatchNorm1d(DIM_HIDDEN, 0.8),
            nn.LeakyReLU(LEAK),
            #nn.ReLU(),
            nn.Linear(DIM_HIDDEN, DIM_HIDDEN),
            #nn.BatchNorm1d(DIM_HIDDEN, 0.8),
            nn.LeakyReLU(LEAK),
            #nn.ReLU(),
            nn.Linear(DIM_HIDDEN, DIM_HIDDEN),
            #nn.BatchNorm1d(DIM_HIDDEN, 0.8),
            nn.LeakyReLU(LEAK),
            #nn.ReLU(),
            nn.Linear(DIM_HIDDEN, enc_out_dim),
            #nn.BatchNorm1d(enc_out_dim, 0.8),
            nn.LeakyReLU(LEAK))
            #nn.ReLU())
        self.decoder = nn.Sequential(
            nn.Linear(DIM_STATE + latent_dim, DIM_HIDDEN),
            #nn.BatchNorm1d(DIM_HIDDEN, 0.8),
            nn.LeakyReLU(LEAK),
            #nn.ReLU(),
            nn.Linear(DIM_HIDDEN, DIM_HIDDEN),
            #nn.BatchNorm1d(DIM_HIDDEN, 0.8),
            nn.LeakyReLU(LEAK),
            #nn.ReLU(),
            nn.Linear(DIM_HIDDEN, DIM_HIDDEN),
            #nn.BatchNorm1d(DIM_HIDDEN, 0.8),
            nn.LeakyReLU(LEAK),
            #nn.ReLU(),
            nn.Linear(DIM_HIDDEN, DIM_HIDDEN),
            #nn.BatchNorm1d(DIM_HIDDEN, 0.8),
            nn.LeakyReLU(LEAK),
            #nn.ReLU(),
            nn.Linear(DIM_HIDDEN, DIM_OBS),
            #nn.BatchNorm1d(DIM_OBS, 0.8),
            nn.LeakyReLU(LEAK))
            #nn.ReLU())


        # distribution parameters
        self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self.fc_var = nn.Linear(enc_out_dim, latent_dim)

        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LR)

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
        #encoder_input = obs_batch
        encoder_input = torch.cat([state_batch, obs_batch], -1)
        obs_encoded = self.encoder(encoder_input)
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
        #recon_loss = self.gaussian_likelihood(obs_hat, self.log_scale, state_batch)

        if calibration:
            log_sigma = ((obs_batch - obs_hat) ** 2).mean([0, 1], keepdim=True).sqrt().log()
            #log_sigma = torch.nn.Parameter(torch.full((1,), 0)[0].float(), requires_grad=True)

            def softclip(tensor, min):
                """ Clips the tensor values at the minimum value min in a softway. Taken from Handful of Trials """
                result_tensor = min + F.softplus(tensor - min)

                return result_tensor

            def gaussian_nll(mu, log_sigma, x):
                return 0.5 * torch.pow((x - mu) / log_sigma.exp(), 2) + log_sigma + 0.5 * np.log(2 * np.pi)

            log_sigma = softclip(log_sigma, -6)
            rec = self.gaussian_likelihood(obs_hat, log_sigma, obs_batch)
            #rec = gaussian_nll(obs_hat, log_sigma, obs_batch).sum()
            recon_loss = rec


        # kl
        beta = BETA
        #kl = self.kl_divergence(z, mu, std)
        kl = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        # elbo
        elbo = (beta*kl - recon_loss)
        elbo = elbo.mean()

        return elbo, kl.mean(), recon_loss.mean()


# Train

vae = VAE()
num_training_steps = NUM_TRAINING_STEPS

load_model = False
save_model = False
chkpt_path = "../../vae_checkpoints/"
if load_model:
    eventfiles = glob.glob(chkpt_path + '*')
    eventfiles.sort(key=os.path.getmtime)
    path = eventfiles[-1]
    #vae.load_state_dict(torch.load(path))

    pretrained_dict = torch.load(path)
    model_dict = vae.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    to_delete = []
    for k, v in pretrained_dict.items():
        if v.shape[0] != model_dict[k].shape[0]:
            to_delete.append(k)
        elif (len(v.shape) > 1) and (v.shape[1] != model_dict[k].shape[1]):
            to_delete.append(k)
    for k in to_delete:
        del pretrained_dict[k]
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    vae.load_state_dict(model_dict)

t0 = time.time()
optimizer = vae.configure_optimizers()
losses = []
testing_errors = []
chkpt_freq = num_training_steps/5

# Linear learning rate scheduling
final_lr = 5e-7
num_steps_interpolate = num_training_steps/2
slope = (final_lr - LR)/num_steps_interpolate

for step in range(num_training_steps):
    if learning_rate_scheduling:
        if step > num_training_steps - num_steps_interpolate:
           learning_rate = slope * (step - (num_training_steps - num_steps_interpolate))
           optimizer.param_groups[0]["lr"] = LR + learning_rate

    if DIM_OBS == 4:
        state_batch, obs_batch = make_batch(BATCH_SIZE)
    else:
        state_batch, obs_batch = make_simple_batch(BATCH_SIZE)

    optimizer.zero_grad()
    loss, kl, recon = vae.training_step(state_batch, obs_batch)
    loss.backward()
    optimizer.step()
    if step % 50 == 0:
        print(step, loss.item(), kl.item(), recon.item())
        losses.append((step, loss.item()))

    if step % 100 == 0:
        # Testing
        if DIM_OBS == 4:
            if pretrain_4d:
                state = np.random.rand(DIM_STATE)
                state_torch = torch.from_numpy(state).reshape((1, DIM_STATE))
                states_batch = torch.cat(BATCH_SIZE * [state_torch]).float()
                obs_batch = torch.cat((states_batch, torch.zeros(BATCH_SIZE, DIM_OBS - DIM_STATE)), dim=-1)
                obs_batch = obs_batch.numpy() + np.random.normal(0, 0.1, (BATCH_SIZE, DIM_OBS))
            else:
                env = Environment()
                state = env.state
                state_torch = torch.from_numpy(state).reshape((1, DIM_STATE))
                states_batch = torch.cat(BATCH_SIZE * [state_torch]).float()
                obs_batch = np.array([env.get_observation() for _ in range(BATCH_SIZE)])
        else:
            state = np.random.rand(DIM_STATE)
            state_torch = torch.from_numpy(state).reshape((1, DIM_STATE))
            states_batch = torch.cat(BATCH_SIZE * [state_torch]).float()
            obs_batch = states_batch.numpy() + np.random.normal(0, 0.1, (BATCH_SIZE, DIM_STATE))

        pz = torch.distributions.Normal(torch.zeros(BATCH_SIZE, LATENT_DIM),
                                        torch.ones(BATCH_SIZE, LATENT_DIM))
        z = pz.rsample()

        decoder_input = torch.cat([states_batch, z], -1)
        #decoder_input = torch.cat([torch.from_numpy(obs_batch).float(), z], -1)
        obs_hat = vae.decoder(decoder_input.detach())

        obs_predicted_mean = np.mean(obs_hat[:, :2].detach().numpy(), axis=0)
        obs_predicted_std = np.std(obs_hat[:, :2].detach().numpy(), axis=0)

        obs_batch_mean = np.mean(obs_batch[:, :2], axis=0)
        obs_batch_std = np.std(obs_batch[:, :2], axis=0)
        testing_error = (step, np.linalg.norm(obs_predicted_mean - obs_batch_mean),
                         np.linalg.norm(obs_predicted_std - 0.1 * np.ones(2)))
        # testing_error = (step, np.linalg.norm(obs_predicted_mean - state),
        #                  np.linalg.norm(obs_predicted_std - obs_batch_std))
        testing_errors.append(testing_error)

    # if step % chkpt_freq == 0:
    #     torch.save(vae.state_dict(), "../../vae_checkpoints/" + str(step))

#pickle.dump(losses, open("vae_losses_obs4d_dummy.p", "wb"))
if save_model:
    torch.save(vae.state_dict(), chkpt_path + 'pretrain2dto4d_' + str(time.time()))

t1 = time.time()
print("Finished training. That took", t1 - t0, "seconds.")


# Plot losses

steps = [loss[0] for loss in losses]
vae_losses = [loss[1] for loss in losses]
plt.plot(steps, vae_losses, label='losses')
plt.xlabel("Num training steps")
plt.ylabel("Loss")
plt.legend()
plt.show()


# Plot testing errors

steps = [error[0] for error in testing_errors]
mean_errors = [error[1] for error in testing_errors]
std_errors = [error[2] for error in testing_errors]
plt.plot(steps, mean_errors, label='mean errors')
plt.plot(steps, std_errors, label='std errors')
plt.xlabel("Num training steps")
plt.ylabel("Testing error")
plt.legend()
plt.show()


# Testing

mean_diff = 0
std_diff = 0
mean_rest_diff = 0
std_rest_diff = 0
num_tests = 2
for j in range(num_tests):
    if DIM_OBS == 4:
        if pretrain_4d:
            state = torch.from_numpy(np.random.rand(DIM_STATE)).reshape((1, DIM_STATE))
            print("STATE", state)
            states_batch = torch.cat(BATCH_SIZE * [state]).float()
            obs_batch = torch.cat((states_batch, torch.zeros(BATCH_SIZE, DIM_OBS - DIM_STATE)), dim=-1)
            obs_batch = obs_batch.numpy() + np.random.normal(0, 0.1, (BATCH_SIZE, DIM_OBS))
        else:
            env = Environment()
            state = torch.from_numpy(env.state).reshape((1, DIM_STATE))
            print("STATE", state)
            states_batch = torch.cat(BATCH_SIZE * [state]).float()
            obs_batch = np.array([env.get_observation() for _ in range(BATCH_SIZE)])
    else:
        state = torch.from_numpy(np.random.rand(DIM_STATE)).reshape((1, DIM_STATE))
        print("STATE", state)
        states_batch = torch.cat(BATCH_SIZE * [state]).float()
        obs_batch = states_batch.numpy() + np.random.normal(0, 0.1, (BATCH_SIZE, DIM_STATE))

    pz = torch.distributions.Normal(torch.zeros(BATCH_SIZE, LATENT_DIM),
                                    torch.ones(BATCH_SIZE, LATENT_DIM))
    z = pz.rsample()
    decoder_input = torch.cat([states_batch, z], -1)
    #decoder_input = torch.cat([torch.from_numpy(obs_batch).float(), z], -1)

    obs_hat = vae.decoder(decoder_input.detach())

    obs_batch_mean = np.mean(obs_batch[:, :2], axis=0)
    obs_batch_std = np.std(obs_batch[:, :2], axis=0)
    print("OBS_BATCH_MEAN\n", obs_batch_mean)
    print("OBS_BATCH_STD\n", obs_batch_std)
    obs_predicted_mean = np.mean(obs_hat[:, :2].detach().numpy(), axis=0)
    obs_predicted_std = np.std(obs_hat[:, :2].detach().numpy(), axis=0)
    print("OBS_PREDICTED_MEAN\n", obs_predicted_mean)
    print("OBS_PREDICTED_STD\n", obs_predicted_std)

    mean_diff += np.linalg.norm(obs_predicted_mean - obs_batch_mean)/num_tests
    std_diff += np.linalg.norm(obs_predicted_std - obs_batch_std)/num_tests

    if DIM_OBS == 4:
        rest_batch_mean = np.mean(obs_batch[:, 2:4], axis=0)
        rest_batch_std = np.std(obs_batch[:, 2:4], axis=0)
        print("REST_BATCH_MEAN\n", rest_batch_mean)
        print("REST_BATCH_STD\n", rest_batch_std)
        rest_predicted_mean = np.mean(obs_hat[:, 2:4].detach().numpy(), axis=0)
        rest_predicted_std = np.std(obs_hat[:, 2:4].detach().numpy(), axis=0)
        print("REST_PREDICTED_MEAN\n", rest_predicted_mean)
        print("REST_PREDICTED_STD\n", rest_predicted_std)

        mean_rest_diff += np.linalg.norm(rest_predicted_mean - rest_batch_mean) / num_tests
        std_rest_diff += np.linalg.norm(rest_predicted_std - rest_batch_std) / num_tests

    plt.scatter([state[0][0]], [state[0][1]], color='k')
    plt.scatter([obs[0] for obs in obs_batch], [obs[1] for obs in obs_batch], color='g')
    plt.scatter([obs[0] for obs in obs_hat.detach().numpy()],
                [obs[1] for obs in obs_hat.detach().numpy()], color='r')
    if DIM_OBS == 4:
        plt.scatter([obs[2] for obs in obs_batch], [obs[3] for obs in obs_batch], color='b')
        plt.scatter([obs[2] for obs in obs_hat.detach().numpy()],
                    [obs[3] for obs in obs_hat.detach().numpy()], color='m')
    plt.show()

print("OBS_MEAN_DIFF\n", mean_diff)
print("OBS_STD_DIFF\n", std_diff)
if DIM_OBS == 4:
    print("REST_MEAN_DIFF\n", mean_rest_diff)
    print("REST_STD_DIFF\n", std_rest_diff)


