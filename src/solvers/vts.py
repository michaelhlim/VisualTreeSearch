# author: @sdeglurkar, @jatucker4, @michaelhlim

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam
from utils.utils import *

# Configs for floor and no LSTM dual smc
from configs.environments.floor import *
from configs.solver.dualsmc import *

# Methods for no LSTM dual smc
from src.methods.dualsmc_nolstm.replay_memory import *
from src.methods.dualsmc.q_network import *
from src.methods.dualsmc_nolstm.observation_network_nolstm import *
from src.methods.dualsmc.gaussian_policy import *


#########################
# Training Process
class VTS:
    def __init__(self):
        self.replay_buffer = ReplayMemory(replay_buffer_size)
        self.MSE_criterion = nn.MSELoss()
        self.BCE_criterion = nn.BCELoss()
        # Filtering
        self.measure_net = MeasureNetwork().to(device)
        self.pp_net = ProposerNetwork().to(device)
        self.measure_optimizer = Adam(self.measure_net.parameters(), lr=FIL_LR)
        self.pp_optimizer = Adam(self.pp_net.parameters(), lr=FIL_LR)

    def save_model(self, path, g):
        stats = {}
        stats['m_net'] = self.measure_net.state_dict()
        stats['pp_net'] = self.pp_net.state_dict()
        stats['generator'] = g.state_dict()
        torch.save(stats, path)

    def load_model(self, path, g):
        stats = torch.load(path)
        # Filtering
        self.measure_net.load_state_dict(stats['m_net'])
        self.pp_net.load_state_dict(stats['pp_net'])
        g.load_state_dict(stats['generator'])

    def get_mean_state(self, state, weight):
        if len(state.shape) == 2:
            # states: [num_particles, dim_state]
            # weights: [num_particles]
            state = torch.FloatTensor(state).to(device)
            weight = weight.unsqueeze(1).to(device)
            mean_state = torch.sum(state * weight, 0)
        elif len(state.shape) == 3:
            # states: torch.Size([batch, num_particles, dim_state])
            # weights: torch.Size([batch, num_particles])
            # return: torch.Size([batch, dim_state])
            weight = weight.unsqueeze(2).to(device)
            mean_state = torch.sum(state * weight, 1).view(state.shape[0], state.shape[2])
        return mean_state

    def density_loss(self, p, w, s):
        # p: [B * K, dim_s]
        # w: [B, K]
        # s: [B, dim_s]
        s = s.unsqueeze(1).repeat(1, NUM_PAR_PF, 1)  # [B, K, dim_s]
        x = torch.exp(-(p - s).pow(2).sum(-1))  # [B, K]
        x = (w * x).sum(-1)  # [B]
        loss = -torch.log(const + x)
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
        mean_state = par_states.mean(1).unsqueeze(1).repeat(1, NUM_PAR_PF, 1)  # mean_state: [B, K, dim_s]
        x = (par_states - mean_state).pow(2).sum(-1)  # [B, K]
        return x.mean(-1)  # [B]

    def soft_q_update(self, observation_generator):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch, \
        obs, curr_par, mean_state, pf_sample = self.replay_buffer.sample(BATCH_SIZE)
        state_batch = torch.FloatTensor(state_batch).to(device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(device)
        action_batch = torch.FloatTensor(action_batch).to(device)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1).to(device)  # (B, 1)
        mask_batch = torch.FloatTensor(np.float32(1 - done_batch)).unsqueeze(1).to(device)
        curr_obs = torch.FloatTensor(obs).to(device)
        curr_par = torch.FloatTensor(curr_par).to(device)  # (B, K, dim_s)
        mean_state = torch.FloatTensor(mean_state).to(device) # (B, dim_s)
        curr_par_sample = torch.FloatTensor(pf_sample).to(device) # (B, M, 2)
        hidden = curr_obs
        cell = curr_obs

        # Observation generative model
        obs_gen_loss = observation_generator.online_training(state_batch, curr_obs)


        # ------------------------
        #  Train Particle Proposer
        # ------------------------
        if PP_EXIST:
            self.pp_optimizer.zero_grad()
            state_propose = self.pp_net(curr_obs, NUM_PAR_PF)
            PP_loss = 0
            P_loss = PP_loss.clone().detach()
            if 'mse' in PP_LOSS_TYPE:
                PP_loss += self.MSE_criterion(state_batch.repeat(NUM_PAR_PF, 1), state_propose)
                P_loss = PP_loss.clone().detach()
            if 'adv' in PP_LOSS_TYPE:
                fake_logit, _, _ = self.measure_net.m_model(state_propose, curr_obs, hidden, cell, NUM_PAR_PF)  # (B, K)
                real_target = torch.ones_like(fake_logit)
                PP_loss += self.BCE_criterion(fake_logit, real_target)
                P_loss = PP_loss.clone().detach()
            if 'density' in PP_LOSS_TYPE:
                std = 0.1
                DEN_COEF = 1
                std_scale = torch.FloatTensor(np.array([2, 1])).to(device)
                par_s = state_propose.view(BATCH_SIZE, -1, DIM_STATE) # [B * K, 2] -> [B, K, 2]
                true_s = state_batch.unsqueeze(1).repeat(1, NUM_PAR_PF, 1) # [B, 2] -> [B, K, 2]
                square_distance = ((par_s - true_s) * std_scale).pow(2).sum(-1)  # [B, K] scale all dimension to -1, +1
                true_state_lik = 1. / (2 * np.pi * std ** 2) * (-square_distance / (2 * std ** 2)).exp()
                pp_nll = -(const + true_state_lik.mean(1)).log().mean()
                PP_loss += DEN_COEF * pp_nll
                P_loss = PP_loss.clone().detach()
            PP_loss.backward()
            self.pp_optimizer.step()
        # ------------------------
        #  Train Observation Model
        # ------------------------
        self.measure_optimizer.zero_grad()
        temp = curr_par.view(-1, DIM_STATE)

        fake_logit, _, _ = self.measure_net.m_model(temp, curr_obs, hidden, cell, NUM_PAR_PF)  # (B, K)

        if PP_EXIST:
            fake_logit_pp, _, _ = self.measure_net.m_model(state_propose.detach(),
                                                           curr_obs, hidden, cell, NUM_PAR_PF)  # (B, K)
            fake_logit = torch.cat((fake_logit, fake_logit_pp), -1)  # (B, 2K)
        fake_target = torch.zeros_like(fake_logit)
        fake_loss = self.BCE_criterion(fake_logit, fake_target)
        real_logit, _, _ = self.measure_net.m_model(state_batch, curr_obs, hidden, cell, 1)  # (batch, num_pars)
        real_target = torch.ones_like(real_logit)
        real_loss = self.BCE_criterion(real_logit, real_target)
        OM_loss = real_loss + fake_loss
        Z_loss = OM_loss.clone().detach()
        OM_loss.backward()
        self.measure_optimizer.step()

        return P_loss, Z_loss, obs_gen_loss

    def soft_q_update_individual(self, state_batch, obs, curr_par):
        state_batch = torch.FloatTensor(state_batch).to(device)
        curr_par = torch.FloatTensor(curr_par).to(device)
        curr_obs = torch.FloatTensor(obs).to(device)
        hidden = curr_obs
        cell = curr_obs

        import time
        
        # ------------------------
        #  Train Particle Proposer
        # ------------------------
        if PP_EXIST:
            self.pp_optimizer.zero_grad()
            state_propose = self.pp_net(curr_obs, NUM_PAR_PF)
            PP_loss = 0
            fake_logit, _, _ = self.measure_net.m_model(state_propose, curr_obs, hidden, cell, NUM_PAR_PF)  # (B, K)
            real_target = torch.ones_like(fake_logit)
            PP_loss += self.BCE_criterion(fake_logit, real_target)
            P_loss = PP_loss.clone().detach()
            PP_loss.backward()
            self.pp_optimizer.step()

        # ------------------------
        #  Train Observation Model
        # ------------------------
        self.measure_optimizer.zero_grad()
        temp = curr_par.view(-1, DIM_STATE)
        fake_logit, _, _ = self.measure_net.m_model(temp,
                                                    curr_obs, hidden, cell, NUM_PAR_PF)  # (B, K)
        if PP_EXIST:
            fake_logit_pp, _, _ = self.measure_net.m_model(state_propose.detach(),
                                                           curr_obs, hidden, cell, NUM_PAR_PF)  # (B, K)
            fake_logit = torch.cat((fake_logit, fake_logit_pp), -1)  # (B, 2K)
        fake_target = torch.zeros_like(fake_logit)
        fake_loss = self.BCE_criterion(fake_logit, fake_target)
        real_logit, _, _ = self.measure_net.m_model(state_batch, curr_obs, hidden, cell, 1)  # (batch, num_pars)
        real_target = torch.ones_like(real_logit)
        real_loss = self.BCE_criterion(real_logit, real_target)
        OM_loss = real_loss + fake_loss
        Z_loss = OM_loss.clone().detach()
        OM_loss.backward()
        self.measure_optimizer.step()

        return Z_loss, P_loss



