# author: @wangyunbo, @liubo
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from utils.utils import *

# Configs for Stanford env and DualSMC Light-Dark 
from configs.environments.stanford import *
from configs.solver.dualsmc_lightdark import *

# Methods for DualSMC Light-Dark
from src.methods.dualsmc_lightdark.dynamic_network import *
from src.methods.dualsmc_lightdark.gaussian_policy import *
from src.methods.dualsmc_lightdark.observation_network_lightdark import *
from src.methods.dualsmc_lightdark.q_network import *
from src.methods.dualsmc_lightdark.replay_memory import *


dlp = DualSMC_LightDark_Params()
sep = Stanford_Environment_Params()

#########################
# Training Process
class DualSMC:
    def __init__(self):
        self.replay_buffer = ReplayMemory(dlp.replay_buffer_size)
        self.gamma = dlp.gamma
        self.tau = dlp.tau
        self.alpha = dlp.alpha
        self.MSE_criterion = nn.MSELoss()
        self.BCE_criterion = nn.BCELoss()
        # Filtering
        self.dynamic_net = DynamicNetwork().to(dlp.device)
        self.measure_net = MeasureNetwork().to(dlp.device)
        self.pp_net = ProposerNetwork().to(dlp.device)
        self.dynamic_optimizer = Adam(self.dynamic_net.parameters(), lr=dlp.fil_lr)
        self.measure_optimizer = Adam(self.measure_net.parameters(), lr=dlp.fil_lr)
        self.pp_optimizer = Adam(self.pp_net.parameters(), lr=dlp.fil_lr)
        # Planning
        self.critic = QNetwork(sep.dim_state, sep.dim_action, dlp.mlp_hunits).to(device=dlp.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=dlp.pla_lr)
        self.critic_target = QNetwork(sep.dim_state, sep.dim_action, dlp.mlp_hunits).to(dlp.device)
        hard_update(self.critic_target, self.critic)
        self.target_entropy = -torch.prod(torch.Tensor(sep.dim_action).to(dlp.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=dlp.device)
        self.alpha_optim = Adam([self.log_alpha], lr=dlp.pla_lr)
        self.policy = GaussianPolicy(sep.dim_state * (dlp.num_par_smc_init + 1), sep.dim_action, dlp.mlp_hunits).to(dlp.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=dlp.pla_lr)

    def save_model(self, path):
        stats = {}
        stats['p_net'] = self.policy.state_dict()
        stats['c_net'] = self.critic.state_dict()
        stats['d_net'] = self.dynamic_net.state_dict()
        stats['m_net'] = self.measure_net.state_dict()
        stats['pp_net'] = self.pp_net.state_dict()
        torch.save(stats, path)

    def load_model(self, path):
        stats = torch.load(path)
        # Filtering
        self.dynamic_net.load_state_dict(stats['d_net'])
        self.measure_net.load_state_dict(stats['m_net'])
        self.pp_net.load_state_dict(stats['pp_net'])
        # Planning
        self.policy.load_state_dict(stats['p_net'])
        self.critic.load_state_dict(stats['c_net'])
        self.dynamic_net.load_state_dict(stats['d_net'])

    def get_mean_state(self, state, weight):
        if len(state.shape) == 2:
            # states: [num_particles, dim_state]
            # weights: [num_particles]
            state = torch.FloatTensor(state).to(dlp.device)
            weight = weight.unsqueeze(1).to(dlp.device)
            mean_state = torch.sum(state * weight, 0)
        elif len(state.shape) == 3:
            # states: torch.Size([batch, num_particles, dim_state])
            # weights: torch.Size([batch, num_particles])
            # return: torch.Size([batch, dim_state])
            weight = weight.unsqueeze(2).to(dlp.device)
            mean_state = torch.sum(state * weight, 1).view(state.shape[0], state.shape[2])
        return mean_state

    def density_loss(self, p, w, s):
        # p: [B * K, dim_s]
        # w: [B, K]
        # s: [B, dim_s]
        s = s.unsqueeze(1).repeat(1, dlp.num_par_pf, 1)  # [B, K, dim_s]
        x = torch.exp(-(p - s).pow(2).sum(-1))  # [B, K]
        x = (w * x).sum(-1)  # [B]
        loss = -torch.log(dlp.const + x)
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
        mean_state = par_states.mean(1).unsqueeze(1).repeat(1, dlp.num_par_pf, 1)  # mean_state: [B, K, dim_s]
        x = (par_states - mean_state).pow(2).sum(-1)  # [B, K]
        return x.mean(-1)  # [B]

    def get_q(self, state, action):
        qf1, qf2 = self.critic(state, action)
        q = torch.min(qf1, qf2)
        return q

    def soft_q_update(self, critic_update):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch, \
            obs, curr_par, mean_state, hidden, cell, pf_sample, curr_orientation = self.replay_buffer.sample(dlp.batch_size)
        state_batch = torch.FloatTensor(state_batch).to(dlp.device)  # [batch_size, dim_state]
        next_state_batch = torch.FloatTensor(next_state_batch).to(dlp.device)  # [batch_size, dim_state] 
        action_batch = torch.FloatTensor(action_batch).to(dlp.device)  # [batch_size, dim_action]
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1).to(dlp.device)  # [batch_size, 1]
        mask_batch = torch.FloatTensor(np.float32(1 - done_batch)).unsqueeze(1).to(dlp.device)  # [batch_size, 1]
        curr_obs = torch.FloatTensor(obs).to(dlp.device)  # [batch_size, in_channels, img_size, img_size]
        curr_par = torch.FloatTensor(curr_par).to(dlp.device)  # [batch_size, num_par_pf, dim_state]
        mean_state = torch.FloatTensor(mean_state).to(dlp.device) # [batch_size, dim_state]
        curr_par_sample = torch.FloatTensor(pf_sample).to(dlp.device) # [batch_size, num_par_smc_init, dim_state] 
        hidden = torch.FloatTensor(hidden).to(dlp.device)  # [batch_size, num_lstm_layer, 1, dim_lstm_hidden]
        hidden = torch.transpose(torch.squeeze(hidden), 0, 1).contiguous()  # [num_lstm_layer, batch_size, dim_lstm_hidden]
        cell = torch.FloatTensor(cell).to(dlp.device)  # [batch_size, num_lstm_layer, 1, dim_lstm_hidden]
        cell = torch.transpose(torch.squeeze(cell), 0, 1).contiguous()  # [num_lstm_layer, batch_size, dim_lstm_hidden]
        curr_orientation = torch.FloatTensor(curr_orientation).unsqueeze(1).to(dlp.device)

        # ------------------------
        #  Train Particle Proposer
        # ------------------------
        if dlp.pp_exist:
            self.pp_optimizer.zero_grad()
            state_propose = self.pp_net(curr_obs, curr_orientation, dlp.num_par_pf)  # [batch_size * num_par_pf, dim_state]
            PP_loss = 0
            if 'mse' in dlp.pp_loss_type:
                PP_loss += self.MSE_criterion(state_batch.repeat(dlp.num_par_pf, 1), state_propose)
            if 'adv' in dlp.pp_loss_type:
                fake_logit, _, _ = self.measure_net.m_model(state_propose, curr_orientation.repeat(dlp.num_par_pf, 1), 
                                    curr_obs, hidden, cell, dlp.num_par_pf)  # [batch_size, num_par_pf]
                real_target = torch.ones_like(fake_logit)
                PP_loss += self.BCE_criterion(fake_logit, real_target)
            if 'density' in dlp.pp_loss_type:
                std = 0.1
                DEN_COEF = 1
                std_scale = torch.FloatTensor(np.array([2, 1])).to(dlp.device)
                par_s = state_propose.view(dlp.batch_size, -1, sep.dim_state) # [B * K, 2] -> [B, K, 2]
                true_s = state_batch.unsqueeze(1).repeat(1, dlp.num_par_pf, 1) # [B, 2] -> [B, K, 2]
                square_distance = ((par_s - true_s) * std_scale).pow(2).sum(-1)  # [B, K] scale all dimension to -1, +1
                true_state_lik = 1. / (2 * np.pi * std ** 2) * (-square_distance / (2 * std ** 2)).exp()
                pp_nll = -(dlp.const + true_state_lik.mean(1)).log().mean()
                PP_loss += DEN_COEF * pp_nll
            P_loss = PP_loss.clone().detach()
            PP_loss.backward()
            self.pp_optimizer.step()

        # ------------------------
        #  Train Observation Model
        # ------------------------
        self.measure_optimizer.zero_grad()
        fake_logit, next_hidden, next_cell = self.measure_net.m_model(curr_par.view(-1, sep.dim_state), curr_orientation.repeat(dlp.num_par_pf, 1),
                                                                      curr_obs, hidden, cell, dlp.num_par_pf)  # [batch_size, num_par_pf]
        if dlp.pp_exist:
            fake_logit_pp, _, _ = self.measure_net.m_model(state_propose.detach(), curr_orientation.repeat(dlp.num_par_pf, 1),
                                                           curr_obs, hidden, cell, dlp.num_par_pf)  # [batch_size, num_par_pf]
            fake_logit = torch.cat((fake_logit, fake_logit_pp), -1)  # [batch_size, 2 * num_par_pf]
        fake_target = torch.zeros_like(fake_logit)
        fake_loss = self.BCE_criterion(fake_logit, fake_target)
        real_logit, _, _ = self.measure_net.m_model(state_batch, curr_orientation, curr_obs, hidden, cell, 1)  # [batch, 1]
        real_target = torch.ones_like(real_logit)
        real_loss = self.BCE_criterion(real_logit, real_target)
        OM_loss = real_loss + fake_loss
        Z_loss = OM_loss.clone().detach()
        OM_loss.backward()
        self.measure_optimizer.step()

        # ------------------------
        #  Train Transition Model
        # ------------------------
        self.dynamic_optimizer.zero_grad()
        state_predict = self.dynamic_net.t_model(state_batch, action_batch * sep.step_range)  # [batch_size, dim_state]
        TM_loss = self.MSE_criterion(state_predict, next_state_batch)
        T_loss = TM_loss.clone().detach()
        TM_loss.backward()
        self.dynamic_optimizer.step()

        # ------------------------
        #  Train SAC
        # ------------------------
        next_mean_state = self.dynamic_net.t_model(mean_state, action_batch * sep.step_range)  # [batch_size, dim_state]
        next_par_sample = self.dynamic_net.t_model(
            curr_par_sample.view(-1, sep.dim_state),
            action_batch.repeat(dlp.num_par_smc_init, 1) * sep.step_range)  # [batch_size * num_par_smc_init, dim_state]
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(
                next_mean_state, next_par_sample.view(dlp.batch_size, -1))  # [batch_size, dim_action]  [batch_size, 1]
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)  # [batch_size, 1]
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi  # [batch_size, 1]
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)  # [batch_size, 1]

        qf1, qf2 = self.critic(state_batch, action_batch)  # [batch_size, 1]
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        q1_loss = qf1_loss.clone().detach()
        q2_loss = qf2_loss.clone().detach()

        self.critic_optim.zero_grad()
        qf1_loss.backward()
        self.critic_optim.step()

        self.critic_optim.zero_grad()
        qf2_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(mean_state, curr_par_sample.view(dlp.batch_size, -1))  # [batch_size, dim_action]  [batch_size, 1]
        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp()

        if critic_update:
            soft_update(self.critic_target, self.critic, self.tau)


        return P_loss, T_loss, Z_loss, q1_loss, q2_loss