# author: @sdeglurkar, @jatucker4, @michaelhlim

import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch
from torch import nn
import torch.nn.functional as F

from configs.environments.floor import *
from configs.solver.observation_generation import *
from configs.solver.dualsmc import *

from src.environments.floor import *
from src.methods.dualsmc_nolstm.replay_memory import *

cvae_params = CVAE_Params()


class ObservationGenerator(nn.Module):
    def __init__(self, enc_out_dim=cvae_params.enc_out_dim, latent_dim=cvae_params.latent_dim):
        super(ObservationGenerator, self).__init__()
        
        dim_hidden = cvae_params.dim_hidden
        leak_rate = cvae_params.leak

        # encoder, decoder
        self.encoder = nn.Sequential(
            nn.Linear(DIM_STATE + DIM_OBS, dim_hidden),
            nn.LeakyReLU(leak_rate),
            nn.Linear(dim_hidden, dim_hidden),
            nn.LeakyReLU(leak_rate),
            nn.Linear(dim_hidden, dim_hidden),
            nn.LeakyReLU(leak_rate),
            nn.Linear(dim_hidden, dim_hidden),
            nn.LeakyReLU(leak_rate),
            nn.Linear(dim_hidden, enc_out_dim),
            nn.LeakyReLU(leak_rate)).to(device)

        self.decoder = nn.Sequential(
            nn.Linear(DIM_STATE + latent_dim, dim_hidden),
            nn.LeakyReLU(leak_rate),
            nn.Linear(dim_hidden, dim_hidden),
            nn.LeakyReLU(leak_rate),
            nn.Linear(dim_hidden, dim_hidden),
            nn.LeakyReLU(leak_rate),
            nn.Linear(dim_hidden, dim_hidden),
            nn.LeakyReLU(leak_rate),
            nn.Linear(dim_hidden, DIM_OBS),
            nn.LeakyReLU(leak_rate)).to(device)

        # distribution parameters
        self.fc_mu = nn.Linear(enc_out_dim, latent_dim).to(device)
        self.fc_var = nn.Linear(enc_out_dim, latent_dim).to(device)

        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0])).to(device)

        self.env = Environment()

        self.optimizer = self.configure_optimizers()


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=cvae_params.lr)


    def gaussian_likelihood(self, x_hat, logscale, x):
        scale = torch.exp(logscale)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)

        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)
        return log_pxz.sum(dim=-1)


    def training_step(self, state_batch, obs_batch):
        # encode x to get the mu and variance parameters
        encoder_input = torch.cat([state_batch, obs_batch], -1).to(device)  # [batch_size, dim_state + dim_obs]
        obs_encoded = self.encoder(encoder_input).to(device)  # [batch_size, enc_out_dim]
        mu, log_var = self.fc_mu(obs_encoded), self.fc_var(obs_encoded)  # [batch_size, latent_dim]

        # sample z from q
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()  # [batch_size, latent_dim]
        decoder_input = torch.cat([state_batch.to(device), z], -1).to(device)  # [batch_size, latent_dim + dim_state]

        # decoded
        obs_hat = self.decoder(decoder_input).to(device)  # [batch_size, dim_obs]

        # reconstruction loss
        recon_loss = self.gaussian_likelihood(obs_hat, self.log_scale, obs_batch.to(device)) # [batch_size]

        if cvae_params.calibration:
            log_sigma = ((obs_batch.to(device) - obs_hat) ** 2).mean([0, 1], keepdim=True).sqrt().log()

            def softclip(tensor, min):
                """ Clips the tensor values at the minimum value min in a softway. Taken from Handful of Trials """
                result_tensor = min + F.softplus(tensor - min)

                return result_tensor

            log_sigma = softclip(log_sigma, -6)
            rec = self.gaussian_likelihood(obs_hat, log_sigma, obs_batch.to(device))  # [batch_size]
            recon_loss = rec

        # kl
        beta = cvae_params.beta
        kl = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1)  # [batch_size]

        # elbo
        elbo = (beta*kl - recon_loss)
        elbo = elbo.mean()

        return elbo, kl.mean(), recon_loss.mean()


    def online_training(self, state_batch, obs_batch):
        self.optimizer.zero_grad()
        loss, kl, recon = self.training_step(state_batch, obs_batch)
        loss_copy = loss.clone().detach()
        loss.backward()
        self.optimizer.step()

        return loss_copy

    def pretrain(self, save_model, save_path):
        print("Pretraining observation generator")
        optimizer = self.configure_optimizers()
        self.training_losses = []
        self.testing_errors = []

        printing_losses = []
        printing_kl = []
        printing_recon = []

        print_freq = 100
        test_freq = 100

        wall_step = int(70e3) #int(2 * cvae_params.num_training_steps/3)
        walls_arr1 = [-1, -1] # wall 0 means any state, wall -1 means no wall
        walls_arr2 = [0.1, 0.4, 0.6, 0.9, -1, -1, -1, 0]

        t0 = time.time()

        for step in range(cvae_params.num_training_steps):
            batch_size = cvae_params.batch_size

            if step >= wall_step:
                index = np.random.randint(len(walls_arr2))
                state_batch, obs_batch = self.env.make_batch_wall(batch_size, walls_arr2[index])
            else:
                index = np.random.randint(len(walls_arr1))
                state_batch, obs_batch = self.env.make_batch_wall(batch_size, walls_arr1[index])
            
            state_batch = torch.from_numpy(state_batch).float()
            obs_batch = torch.from_numpy(obs_batch).float()

            optimizer.zero_grad()
            loss, kl, recon = self.training_step(state_batch, obs_batch)
            loss.backward()
            optimizer.step()

            printing_losses.append(loss.item())
            printing_kl.append(kl.item())
            printing_recon.append(recon.item())

            if step % print_freq == 0:
                print("Step: ", step, ", G loss: ", np.mean(
                    printing_losses), ", KL: ", np.mean(
                    printing_kl), ", recon: ", np.mean(
                    printing_recon),)
                self.training_losses.append((step, loss.item()))

                printing_losses = []
                printing_kl = []
                printing_recon = []

            if step % test_freq == 0:
                # Testing
                test_result = self.test()

                obs_batch_mean = test_result['obs_batch_mean']
                obs_batch_std = test_result['obs_batch_std']
                obs_predicted_mean = test_result['obs_predicted_mean']
                obs_predicted_std = test_result['obs_predicted_std']

                testing_error = (step, np.linalg.norm(obs_predicted_mean - obs_batch_mean),
                                 np.linalg.norm(obs_predicted_std - obs_batch_std))
                self.testing_errors.append(testing_error)

        t1 = time.time()

        print("Done pretraining")

        # Training time
        return t1 - t0


    def sample(self, batch_size, states_batch):
        pz = torch.distributions.Normal(torch.zeros(batch_size, cvae_params.latent_dim),
                                        torch.ones(batch_size, cvae_params.latent_dim))
        z = pz.rsample().to(device)  # [batch_size, latent_dim]
        decoder_input = torch.cat([states_batch.to(device), z], -1)  # [batch_size, latent_dim + dim_state]

        obs_hat = self.decoder(decoder_input.detach()) # [batch_size, dim_obs]

        return obs_hat


    def plot_training_losses(self):
        # Training losses produced during pretraining
        steps = [loss[0] for loss in self.training_losses]
        vae_losses = [loss[1] for loss in self.training_losses]
        plt.plot(steps, vae_losses, label='training losses')
        plt.xlabel("Num training steps")
        plt.ylabel("Training Loss")
        plt.legend()
        plt.show()


    def plot_testing_losses(self):
        # Testing losses produced during pretraining
        steps = [error[0] for error in self.testing_errors]
        mean_errors = [error[1] for error in self.testing_errors]
        std_errors = [error[2] for error in self.testing_errors]
        plt.plot(steps, mean_errors, label='mean errors')
        plt.plot(steps, std_errors, label='std errors')
        plt.xlabel("Num training steps")
        plt.ylabel("Testing error")
        plt.legend()
        plt.show()


    def load_model(self, checkpoint_path):
        # Load most recent model in path
        eventfiles = glob.glob(checkpoint_path + '*')
        eventfiles.sort(key=os.path.getmtime)
        path = eventfiles[-1]

        pretrained_dict = torch.load(path)
        model_dict = self.state_dict()
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
        self.load_state_dict(model_dict)


    def test(self):
        batch_size = cvae_params.batch_size

        state, obs_batch = self.env.make_batch_single_state(batch_size)
        state = torch.from_numpy(state).reshape((1, DIM_STATE))
        states_batch = torch.cat(batch_size * [state]).float()

        obs_hat = self.sample(batch_size, states_batch)

        obs_predicted_mean = np.mean(obs_hat[:, :2].detach().cpu().numpy(), axis=0)
        obs_predicted_std = np.std(obs_hat[:, :2].detach().cpu().numpy(), axis=0)
        if DIM_OBS == 4:
            rest_predicted_mean = np.mean(obs_hat[:, 2:4].detach().cpu().numpy(), axis=0)
            rest_predicted_std = np.std(obs_hat[:, 2:4].detach().cpu().numpy(), axis=0)
        else:
            rest_predicted_mean = None
            rest_predicted_std = None

        obs_batch_mean = np.mean(obs_batch[:, :2], axis=0)
        obs_batch_std = np.std(obs_batch[:, :2], axis=0)
        if DIM_OBS == 4:
            rest_batch_mean = np.mean(obs_batch[:, 2:4], axis=0)
            rest_batch_std = np.std(obs_batch[:, 2:4], axis=0)
        else:
            rest_batch_mean = None
            rest_batch_std = None

        test_result = {'state': state,
                       'obs_predicted_mean': obs_predicted_mean,
                       'obs_predicted_std': obs_predicted_std,
                       'rest_predicted_mean': rest_predicted_mean,
                       'rest_predicted_std': rest_predicted_std,
                       'obs_batch_mean': obs_batch_mean,
                       'obs_batch_std': obs_batch_std,
                       'rest_batch_mean': rest_batch_mean,
                       'rest_batch_std': rest_batch_std,
                       'obs_batch': obs_batch,
                       'obs_hat': obs_hat}

        return test_result

    def test_with_prints(self, load_model=False, checkpoint_path=None):
        # Testing after pretraining

        if load_model:
            self.load_model(checkpoint_path)

        mean_diff = 0
        std_diff = 0
        mean_rest_diff = 0
        std_rest_diff = 0

        num_tests = cvae_params.num_tests

        for j in range(num_tests):
            test_result = self.test()

            state = test_result['state']
            obs_batch = test_result['obs_batch']
            obs_hat = test_result['obs_hat']

            obs_batch_mean = test_result['obs_batch_mean']
            obs_batch_std = test_result['obs_batch_std']
            obs_predicted_mean = test_result['obs_predicted_mean']
            obs_predicted_std = test_result['obs_predicted_std']

            print("OBS_BATCH_MEAN\n", obs_batch_mean)
            print("OBS_BATCH_STD\n", obs_batch_std)
            print("OBS_PREDICTED_MEAN\n", obs_predicted_mean)
            print("OBS_PREDICTED_STD\n", obs_predicted_std)

            mean_diff += np.linalg.norm(obs_predicted_mean - obs_batch_mean) / num_tests
            std_diff += np.linalg.norm(obs_predicted_std - obs_batch_std) / num_tests

            if DIM_OBS == 4:
                rest_batch_mean = test_result['rest_batch_mean']
                rest_batch_std = test_result['rest_batch_std']
                rest_predicted_mean = test_result['rest_predicted_mean']
                rest_predicted_std = test_result['rest_predicted_std']
                print("REST_BATCH_MEAN\n", rest_batch_mean)
                print("REST_BATCH_STD\n", rest_batch_std)
                print("REST_PREDICTED_MEAN\n", rest_predicted_mean)
                print("REST_PREDICTED_STD\n", rest_predicted_std)

                mean_rest_diff += np.linalg.norm(rest_predicted_mean - rest_batch_mean) / num_tests
                std_rest_diff += np.linalg.norm(rest_predicted_std - rest_batch_std) / num_tests

            plt.figure()
            plt.scatter([state[0][0]], [state[0][1]], color='k')
            plt.scatter([obs[0] for obs in obs_batch], [obs[1] for obs in obs_batch], color='g')
            plt.scatter([obs[0] for obs in obs_hat.detach().cpu().numpy()],
                        [obs[1] for obs in obs_hat.detach().cpu().numpy()], color='r')
            if DIM_OBS == 4:
                plt.scatter([obs[2] for obs in obs_batch], [obs[3] for obs in obs_batch], color='b')
                plt.scatter([obs[2] for obs in obs_hat.detach().cpu().numpy()],
                            [obs[3] for obs in obs_hat.detach().cpu().numpy()], color='m')
            plt.savefig("cvae_test" + str(j))                

        print("OBS_MEAN_DIFF\n", mean_diff)
        print("OBS_STD_DIFF\n", std_diff)
        if DIM_OBS == 4:
            print("REST_MEAN_DIFF\n", mean_rest_diff)
            print("REST_STD_DIFF\n", std_rest_diff)


