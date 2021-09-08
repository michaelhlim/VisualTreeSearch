# author: @wangyunbo, @liubo
import random
import torch
from torch.distributions import Normal
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from utils.utils import *

# Configs for floor and no LSTM dual smc
from configs.environments.stanford import *
from configs.solver.vts_lightdark import *

# Methods for no LSTM dual smc
from src.methods.vts_lightdark.replay_memory import *
from src.methods.vts_lightdark.observation_generator_lightdark import *
from src.methods.vts_lightdark.observation_generator_conv_lightdark import * 
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
        self.observation_encoder = ObservationGeneratorConv().to(vlp.device)
        self.measure_net = MeasureNetwork(self.observation_encoder).to(vlp.device)
        self.pp_net = ProposerNetwork(self.observation_encoder).to(vlp.device)
        self.generator = ObservationGenerator(self.observation_encoder).to(vlp.device)
        self.measure_optimizer = Adam(self.measure_net.parameters(), lr=vlp.zp_lr)
        self.pp_optimizer = Adam(self.pp_net.parameters(), lr=vlp.zp_lr)
        self.generator_optimizer = Adam(self.generator.parameters(), lr=vlp.g_lr)

    def save_model(self, path):
        stats = {}
        stats['obs_encoder'] = self.observation_encoder.state_dict()
        stats['m_net'] = self.measure_net.state_dict()
        stats['pp_net'] = self.pp_net.state_dict()
        stats['generator'] = self.generator.state_dict()
        torch.save(stats, path)

    def load_model(self, path, load_zp=True, load_g=True):
        stats = torch.load(path)
        self.observation_encoder.load_state_dict(stats['obs_encoder'])
        if load_zp:
            self.measure_net.load_state_dict(stats['m_net'])
            self.pp_net.load_state_dict(stats['pp_net'])
        if load_g:
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
            obs, curr_par, mean_state, pf_sample, curr_orientation = self.replay_buffer.sample(vlp.batch_size)
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
        curr_orientation = torch.FloatTensor(curr_orientation).unsqueeze(1).to(dlp.device)

        # Observation generative model
        # obs_gen_loss = observation_generator.online_training(state_batch, curr_obs)

        # ------------------------
        #  Train Particle Proposer
        # ------------------------
        if vlp.pp_exist:
            self.pp_optimizer.zero_grad()
            state_propose = self.pp_net(curr_obs, curr_orientation, vlp.num_par_pf)
            PP_loss = 0
            P_loss = PP_loss
            if 'mse' in vlp.pp_loss_type:
                PP_loss += self.MSE_criterion(state_batch.repeat(vlp.num_par_pf, 1), state_propose)
                P_loss = PP_loss
            if 'adv' in vlp.pp_loss_type:
                fake_logit, _, _ = self.measure_net.m_model(state_propose, curr_orientation.repeat(vlp.num_par_pf, 1), 
                                                            curr_obs, hidden, cell, vlp.num_par_pf)  # (B, K)
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
        fake_logit, _, _ = self.measure_net.m_model(curr_par.view(-1, sep.dim_state), 
                                                    curr_orientation.repeat(vlp.num_par_pf, 1), 
                                                    curr_obs, hidden, cell, vlp.num_par_pf)  # (B, K)
        if vlp.pp_exist:
            fake_logit_pp, _, _ = self.measure_net.m_model(state_propose.detach(), curr_orientation.repeat(vlp.num_par_pf, 1),
                                                           curr_obs, hidden, cell, vlp.num_par_pf)  # (B, K)
            fake_logit = torch.cat((fake_logit, fake_logit_pp), -1)  # (B, 2K)
        fake_target = torch.zeros_like(fake_logit)
        fake_loss = self.BCE_criterion(fake_logit, fake_target)
        real_logit, _, _ = self.measure_net.m_model(state_batch, curr_orientation, curr_obs, hidden, cell, 1)  # (batch, num_pars)
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
        conditional_input = torch.cat((state_batch, curr_orientation), -1)
        [recons, input, mu, log_var] = self.generator.forward(conditional_input, enc_obs)
        args = [recons, input, mu, log_var]
        loss_dict = self.generator.loss_function(self.beta, *args)
        OG_loss = loss_dict['loss']
        G_loss = OG_loss
        OG_loss.backward()
        self.generator_optimizer.step()

        return P_loss, Z_loss, G_loss


    def pretraining_zp(self, state_batch, curr_orientation, obs, curr_par):
        state_batch = torch.FloatTensor(state_batch).to(vlp.device)  # [batch_size, dim_state]
        curr_orientation = torch.FloatTensor(curr_orientation).unsqueeze(1).to(vlp.device)  # [batch_size, 1]
        curr_par = torch.FloatTensor(curr_par).to(vlp.device)  # [batch_size * num_par, dim_state]
        curr_obs = torch.FloatTensor(obs).to(vlp.device)  # [batch_size, in_channels, img_size, img_size]
        hidden = curr_obs
        cell = curr_obs
        # ------------------------
        #  Train Particle Proposer
        # ------------------------
        if vlp.pp_exist:
            self.pp_optimizer.zero_grad()
            state_propose = self.pp_net(curr_obs, curr_orientation, vlp.num_par_pf)  # [batch_size * num_par, dim_state]
            PP_loss = 0
            P_loss = PP_loss
            if 'mse' in vlp.pp_loss_type:
                PP_loss += self.MSE_criterion(state_batch.repeat(vlp.num_par_pf, 1), state_propose)
                P_loss = PP_loss
            if 'adv' in vlp.pp_loss_type:
                fake_logit, _, _ = self.measure_net.m_model(state_propose, curr_orientation.repeat(vlp.num_par_pf, 1),
                                                             curr_obs, hidden, cell, vlp.num_par_pf)  # [batch_size, num_par]
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
        fake_logit, _, _ = self.measure_net.m_model(curr_par.view(-1, sep.dim_state), 
                                                    curr_orientation.repeat(vlp.num_par_pf, 1), 
                                                    curr_obs, hidden, cell, vlp.num_par_pf)  # [batch_size, num_par]
        if vlp.pp_exist:
            fake_logit_pp, _, _ = self.measure_net.m_model(state_propose.detach(), curr_orientation.repeat(vlp.num_par_pf, 1),
                                                           curr_obs, hidden, cell, vlp.num_par_pf)  # [batch_size, num_par]
            fake_logit = torch.cat((fake_logit, fake_logit_pp), -1)  # [batch_size, 2 * num_par]
        fake_target = torch.zeros_like(fake_logit)
        fake_loss = self.BCE_criterion(fake_logit, fake_target)
        real_logit, _, _ = self.measure_net.m_model(state_batch, curr_orientation, 
                                                    curr_obs, hidden, cell, 1)  # [batch_size, 1]
        real_target = torch.ones_like(real_logit)
        real_loss = self.BCE_criterion(real_logit, real_target)
        OM_loss = real_loss + fake_loss
        Z_loss = OM_loss
        OM_loss.backward()
        self.measure_optimizer.step()

        return Z_loss, P_loss
    

    def pretraining_g(self, state_batch, curr_orientation, obs):
        state_batch = torch.FloatTensor(state_batch).to(vlp.device)  # [batch_size, dim_state]
        curr_orientation = torch.FloatTensor(curr_orientation).unsqueeze(1).to(vlp.device)  # [batch_size, 1]
        obs = torch.FloatTensor(obs).to(vlp.device)  # [batch_size, in_channels, img_size, img_size]

        #enc_obs = self.measure_net.observation_encoder(obs.detach())   # [batch_size, obs_encode_out]
        enc_obs = obs

        # ------------------------
        #  Train Observation Generator
        # ------------------------
        conditional_input = torch.cat((state_batch, curr_orientation), -1)  # [batch_size, dim_state + 1]
        self.generator_optimizer.zero_grad()
        #[recons, input, mu, log_var] = self.generator.forward(conditional_input, enc_obs.detach())
        [recons, input, mu, log_var] = self.generator.forward(conditional_input, enc_obs)
        args = [recons, input, mu, log_var]
        loss_dict = self.generator.loss_function(*args)
        OG_loss = loss_dict['loss']
        G_loss = OG_loss
        OG_loss.backward()
        self.generator_optimizer.step()

        return G_loss
    

    def test_models(self, batch_size, state, orientation, obs):
        state = torch.FloatTensor(state).to(vlp.device).detach()  # [batch_size, dim_state]
        orientation = torch.FloatTensor(orientation).unsqueeze(1).to(vlp.device).detach()  # [batch_size, 1]
        obs = torch.FloatTensor(obs)  # [batch_size, img_size, img_size, in_channels]
        obs = obs.permute(0, 3, 1, 2).to(vlp.device).detach()  # [batch_size, in_channels, 32, 32]

        real_logit, _, _ = self.measure_net.m_model(state, orientation, obs, None, None, 1)  # [batch_size, 1]
        state_propose = self.pp_net(obs, orientation, vlp.num_par_pf)   # [batch_size * num_par, dim_state]
        fake_logit, _, _ = self.measure_net.m_model(state_propose.detach(), orientation.repeat(vlp.num_par_pf, 1),
                                                           obs, None, None, vlp.num_par_pf)  # [batch_size, num_par]
        
        conditional_input = torch.cat((state, orientation), -1)  # [batch_size, dim_state + 1]
        enc_obs_hat = self.generator.sample(batch_size, conditional_input)  # [batch_size, obs_encode_out]
        

        # Looking at generated images
        image_hat = self.generator.conv.decode(enc_obs_hat)
        output = image_hat.permute(0, 2, 3, 1).squeeze(0)  # [32, 32, in_channels]
        output = output.detach().cpu().numpy()  
        import pickle
        import cv2
        normalization_data = pickle.load(open("data_normalization.p", "rb"))
        rmean, gmean, bmean, rstd, gstd, bstd = normalization_data   
        output[:, :, 0] = (output[:, :, 0] * rstd + rmean)
        output[:, :, 1] = (output[:, :, 1] * gstd + gmean)
        output[:, :, 2] = (output[:, :, 2] * bstd + bmean)
        output = output[:,:,::-1]   ## CV2 works in BGR space instead of RGB!! So dumb! -- converts to BGR
        cv2.imwrite("output.png", output)
        original = obs[0]  
        original = original.permute(1, 2, 0)  # [32, 32, in_channels]
        original = original.detach().cpu().numpy()
        original[:, :, 0] = (original[:, :, 0] * rstd + rmean)
        original[:, :, 1] = (original[:, :, 1] * gstd + gmean)
        original[:, :, 2] = (original[:, :, 2] * bstd + bmean)
        original = original[:,:,::-1]   ## CV2 works in BGR space instead of RGB!! So dumb!
        cv2.imwrite("original.png", original)


        #enc_obs = self.measure_net.observation_encoder(obs.detach())
        #enc_obs = self.generator.observation_encoder(obs.detach())
        enc_obs = self.generator.conv.encode(obs.detach())


        print("State:", state, "\nOrientation:", orientation, "\nReal Logit:", real_logit,
                "\nProposed States:", state_propose, "\nFake Logit:", fake_logit, "\nEncoded Observation:", enc_obs[0, :64], 
                "\nGenerated Encoded Observation:", enc_obs_hat[0, :64])
        # print("State:", state, "\nOrientation:", orientation, "\nEncoded Observation:", enc_obs[0, :64], 
        #         "\nGenerated Encoded Observation:", enc_obs_hat[0, :64])


        real_logit_gen, _, _ = self.measure_net.m_model(state, orientation, enc_obs_hat, None, None, 1, obs_is_encoded=True)  # [batch_size, 1]
        state_propose_gen = self.pp_net(enc_obs_hat, orientation, vlp.num_par_pf, obs_is_encoded=True)   # [batch_size * num_par, dim_state]
        fake_logit_gen, _, _ = self.measure_net.m_model(state_propose.detach(), orientation.repeat(vlp.num_par_pf, 1),
                                                           enc_obs_hat, None, None, vlp.num_par_pf, obs_is_encoded=True)  # [batch_size, num_par]

        print("\nReal Logit:", real_logit_gen, "\nProposed States:", state_propose_gen, "\nFake Logit:", fake_logit_gen)



        real_logit_gen_img, _, _ = self.measure_net.m_model(state, orientation, image_hat, None, None, 1)  # [batch_size, 1]
        state_propose_gen_img = self.pp_net(image_hat, orientation, vlp.num_par_pf)   # [batch_size * num_par, dim_state]
        fake_logit_gen_img, _, _ = self.measure_net.m_model(state_propose.detach(), orientation.repeat(vlp.num_par_pf, 1),
                                                           image_hat, None, None, vlp.num_par_pf)  # [batch_size, num_par]

        print("\nReal Logit:", real_logit_gen_img, "\nProposed States:", state_propose_gen_img, "\nFake Logit:", fake_logit_gen_img)