# author: @wangyunbo, @liubo
import random
import torch
from torch.distributions import Normal
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from solvers.generative_observation_prediction import ObservationGenerator
from utils.utils import *

# Configs for floor and no LSTM dual smc
from configs.environments.stanford import *
from configs.solver.vts_lightdark import *

# Methods for no LSTM dual smc
from src.methods.vts_lightdark.replay_memory import *
from src.methods.vts_lightdark.observation_generator_lightdark import *
from src.methods.vts_lightdark.observation_network_lightdark import *


vlp = VTS_LightDark_Params()
sep = Stanford_Environment_Params()

#########################
# Training Process
class VTS:
    def __init__(self):
        self.replay_buffer = ReplayMemory(vlp.replay_buffer_size)
        self.MSE_criterion = nn.MSELoss()
        self.BCE_criterion = nn.BCELoss()
        # Filtering
        self.measure_net = MeasureNetwork().to(vlp.device)
        self.pp_net = ProposerNetwork().to(vlp.device)
        self.generator = ObservationGenerator().to(vlp.device)
        self.measure_optimizer = Adam(self.measure_net.parameters(), lr=vlp.fil_lr)
        self.pp_optimizer = Adam(self.pp_net.parameters(), lr=vlp.fil_lr)
        self.generator_optimizer = Adam(self.generator.parameters(), lr=vlp.lr)

    def save_model(self, path):
        stats = {}
        stats['m_net'] = self.measure_net.state_dict()
        stats['pp_net'] = self.pp_net.state_dict()
        stats['generator'] = self.generator.state_dict()
        torch.save(stats, path)

    def load_model(self, path):
        stats = torch.load(path)
        # Filtering
        self.measure_net.load_state_dict(stats['m_net'])
        self.pp_net.load_state_dict(stats['pp_net'])
        self.generator.load_state_dict(stats['generator'])

    def get_mean_state(self, state, weight):
        if len(state.shape) == 2:
            # states: [num_particles, dim_state]
            # weights: [num_particles]
            state = torch.FloatTensor(state).to(vlp.device)
            weight = weight.unsqueeze(1).to(vlp.device)
            mean_state = torch.sum(state * weight, 0)
        elif len(state.shape) == 3:
            # states: torch.Size([batch, num_particles, dim_state])
            # weights: torch.Size([batch, num_particles])
            # return: torch.Size([batch, dim_state])
            weight = weight.unsqueeze(2).to(vlp.device)
            mean_state = torch.sum(state * weight, 1).view(state.shape[0], state.shape[2])
        return mean_state

    def density_loss(self, p, w, s):
        # p: [B * K, dim_s]
        # w: [B, K]
        # s: [B, dim_s]
        s = s.unsqueeze(1).repeat(1, vlp.num_par_pf, 1)  # [B, K, dim_s]
        x = torch.exp(-(p - s).pow(2).sum(-1))  # [B, K]
        x = (w * x).sum(-1)  # [B]
        loss = -torch.log(vlp.const + x)
        return loss

    def par_weighted_var(self, par_states, par_weight, mean_state):
        # par_states: [B, K, dim_s]
        # par_weight: [B, K]
        # mean_state: [B, dim_s]
        num_par = par_states.shape[1]
        mean_state = mean_state.unsqueeze(1).repeat(1, num_par, 1)  # [B, K, dim_s]
        x = par_weight * (par_states - mean_state).abs().sum(-1)  # [B, K]
        return x.sum(-1)  # [B]

    def par_var(self, par_states):
        # par_states: [B, K, dim_s]
        mean_state = par_states.mean(1).unsqueeze(1).repeat(1, vlp.num_par_pf, 1)  # mean_state: [B, K, dim_s]
        x = (par_states - mean_state).pow(2).sum(-1)  # [B, K]
        return x.mean(-1)  # [B]

    def online_training(self):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch, \
        obs, curr_par, mean_state, pf_sample = self.replay_buffer.sample(vlp.batch_size)
        state_batch = torch.FloatTensor(state_batch).to(vlp.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(vlp.device)
        action_batch = torch.FloatTensor(action_batch).to(vlp.device)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1).to(vlp.device)  # (B, 1)
        mask_batch = torch.FloatTensor(np.float32(1 - done_batch)).unsqueeze(1).to(vlp.device)
        curr_obs = torch.FloatTensor(obs).to(vlp.device)
        curr_par = torch.FloatTensor(curr_par).to(vlp.device)  # (B, K, dim_s)
        mean_state = torch.FloatTensor(mean_state).to(vlp.device) # (B, dim_s)
        curr_par_sample = torch.FloatTensor(pf_sample).to(vlp.device) # (B, M, 2)
        hidden = curr_obs
        cell = curr_obs

        # Observation generative model
        # obs_gen_loss = observation_generator.online_training(state_batch, curr_obs)

        # ------------------------
        #  Train Particle Proposer
        # ------------------------
        if vlp.pp_exist:
            self.pp_optimizer.zero_grad()
            state_propose = self.pp_net(curr_obs, vlp.num_par_pf)
            PP_loss = 0
            P_loss = PP_loss
            if 'mse' in vlp.pp_loss_type:
                PP_loss += self.MSE_criterion(state_batch.repeat(vlp.num_par_pf, 1), state_propose)
                P_loss = PP_loss
            if 'adv' in vlp.pp_loss_type:
                fake_logit, _, _ = self.measure_net.m_model(state_propose, curr_obs, hidden, cell, vlp.num_par_pf)  # (B, K)
                real_target = torch.ones_like(fake_logit)
                PP_loss += self.BCE_criterion(fake_logit, real_target)
                P_loss = PP_loss
            if 'density' in vlp.pp_loss_type:
                std = 0.1
                DEN_COEF = 1
                std_scale = torch.FloatTensor(np.array([2, 1])).to(vlp.device)
                par_s = state_propose.view(vlp.batch_size, -1, sep.dim_state) # [B * K, 2] -> [B, K, 2]
                true_s = state_batch.unsqueeze(1).repeat(1, vlp.num_par_pf, 1) # [B, 2] -> [B, K, 2]
                square_distance = ((par_s - true_s) * std_scale).pow(2).sum(-1)  # [B, K] scale all dimension to -1, +1
                true_state_lik = 1. / (2 * np.pi * std ** 2) * (-square_distance / (2 * std ** 2)).exp()
                pp_nll = -(vlp.const + true_state_lik.mean(1)).log().mean()
                PP_loss += DEN_COEF * pp_nll
                P_loss = PP_loss
            PP_loss.backward()
            self.pp_optimizer.step()
        # ------------------------
        #  Train Observation Model
        # ------------------------
        self.measure_optimizer.zero_grad()
        temp = curr_par.view(-1, sep.dim_state)
        fake_logit, _, _ = self.measure_net.m_model(temp, curr_obs, hidden, cell, vlp.num_par_pf)  # (B, K)
        if vlp.pp_exist:
            fake_logit_pp, _, _ = self.measure_net.m_model(state_propose.detach(),
                                                           curr_obs, hidden, cell, vlp.num_par_pf)  # (B, K)
            fake_logit = torch.cat((fake_logit, fake_logit_pp), -1)  # (B, 2K)
        fake_target = torch.zeros_like(fake_logit)
        fake_loss = self.BCE_criterion(fake_logit, fake_target)
        real_logit, _, _ = self.measure_net.m_model(state_batch, curr_obs, hidden, cell, 1)  # (batch, num_pars)
        real_target = torch.ones_like(real_logit)
        real_loss = self.BCE_criterion(real_logit, real_target)
        OM_loss = real_loss + fake_loss
        Z_loss = OM_loss
        OM_loss.backward()
        self.measure_optimizer.step()

        # ------------------------
        #  Train Observation Generator
        # ------------------------
        self.generator_optimizer.zero_grad()
        enc_obs = self.measure_net.observation_encoder(curr_obs)
        [recons, input, mu, log_var] = self.generator.forward(enc_obs, state_batch)
        args = [recons, input, mu, log_var]
        loss_dict = self.generator.loss_function(self.beta, *args)
        OG_loss = loss_dict['loss']
        G_loss = OG_loss
        OG_loss.backward()
        self.generator_optimizer.step()

        return P_loss, Z_loss, G_loss


    def pretraining_zp(self, state_batch, obs, curr_par):
        state_batch = torch.FloatTensor(state_batch).to(vlp.device)
        curr_par = torch.FloatTensor(curr_par).to(vlp.device)
        curr_obs = torch.FloatTensor(obs).to(vlp.device)
        hidden = curr_obs
        cell = curr_obs
        # ------------------------
        #  Train Particle Proposer
        # ------------------------
        if vlp.pp_exist:
            self.pp_optimizer.zero_grad()
            state_propose = self.pp_net(curr_obs, vlp.num_par_pf)
            PP_loss = 0
            P_loss = PP_loss
            if 'mse' in vlp.pp_loss_type:
                PP_loss += self.MSE_criterion(state_batch.repeat(vlp.num_par_pf, 1), state_propose)
                P_loss = PP_loss
            if 'adv' in vlp.pp_loss_type:
                fake_logit, _, _ = self.measure_net.m_model(state_propose, curr_obs, hidden, cell, vlp.num_par_pf)  # (B, K)
                real_target = torch.ones_like(fake_logit)
                PP_loss += self.BCE_criterion(fake_logit, real_target)
                P_loss = PP_loss
            if 'density' in vlp.pp_loss_type:
                std = 0.1
                DEN_COEF = 1
                std_scale = torch.FloatTensor(np.array([2, 1])).to(vlp.device)
                par_s = state_propose.view(vlp.batch_size, -1, sep.dim_state) # [B * K, 2] -> [B, K, 2]
                true_s = state_batch.unsqueeze(1).repeat(1, vlp.num_par_pf, 1) # [B, 2] -> [B, K, 2]
                square_distance = ((par_s - true_s) * std_scale).pow(2).sum(-1)  # [B, K] scale all dimension to -1, +1
                true_state_lik = 1. / (2 * np.pi * std ** 2) * (-square_distance / (2 * std ** 2)).exp()
                pp_nll = -(vlp.const + true_state_lik.mean(1)).log().mean()
                PP_loss += DEN_COEF * pp_nll
                P_loss = PP_loss
            PP_loss.backward()
            self.pp_optimizer.step()

        # ------------------------
        #  Train Observation Model
        # ------------------------
        self.measure_optimizer.zero_grad()
        temp = curr_par.view(-1, sep.dim_state)
        fake_logit, _, _ = self.measure_net.m_model(temp, curr_obs, hidden, cell, vlp.num_par_pf)  # (B, K)
        if vlp.pp_exist:
            fake_logit_pp, _, _ = self.measure_net.m_model(state_propose.detach(),
                                                           curr_obs, hidden, cell, vlp.num_par_pf)  # (B, K)
            fake_logit = torch.cat((fake_logit, fake_logit_pp), -1)  # (B, 2K)
        fake_target = torch.zeros_like(fake_logit)
        fake_loss = self.BCE_criterion(fake_logit, fake_target)
        real_logit, _, _ = self.measure_net.m_model(state_batch, curr_obs, hidden, cell, 1)  # (batch, num_pars)
        real_target = torch.ones_like(real_logit)
        real_loss = self.BCE_criterion(real_logit, real_target)
        OM_loss = real_loss + fake_loss
        Z_loss = OM_loss
        OM_loss.backward()
        self.measure_optimizer.step()

        return Z_loss, P_loss
    

    def pretraining_g(self, state_batch, enc_obs, curr_par):
        state_batch = torch.FloatTensor(state_batch).to(vlp.device)
        curr_par = torch.FloatTensor(curr_par).to(vlp.device)
        curr_obs = torch.FloatTensor(enc_obs).to(vlp.device)
        hidden = curr_obs
        cell = curr_obs

        # ------------------------
        #  Train Observation Generator
        # ------------------------
        self.generator_optimizer.zero_grad()
        [recons, input, mu, log_var] = self.generator.forward(curr_obs, state_batch)
        args = [recons, input, mu, log_var]
        loss_dict = self.generator.loss_function(self.beta, *args)
        OG_loss = loss_dict['loss']
        G_loss = OG_loss
        OG_loss.backward()
        self.generator_optimizer.step()

        return G_loss



