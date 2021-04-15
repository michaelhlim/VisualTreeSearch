import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from env import Environment
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import random
from scipy.stats import norm
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam
from torch.optim import RMSprop
import time


DIM_STATE = 2
DIM_OBS = 4
DIM_HIDDEN = 256
NUM_PAR_PF = 1   # num particles
FIL_LR = 1e-3 # filtering

DIM_LSTM_HIDDEN = 128
NUM_LSTM_LAYER = 2

BATCH_SIZE = 64

# Particle Proposer
class ProposerNetwork(nn.Module):
    def __init__(self):
        super(ProposerNetwork, self).__init__()
        self.dim = 64
        self.obs_encode = nn.Sequential(
            nn.Linear(DIM_OBS, DIM_HIDDEN),
            nn.ReLU(),
            nn.Linear(DIM_HIDDEN, DIM_HIDDEN),
            nn.ReLU(),
            nn.Linear(DIM_HIDDEN, self.dim),
            nn.ReLU()
        )
        self.p_net = nn.Sequential(
            nn.Linear(self.dim * 2, DIM_HIDDEN),
            nn.ReLU(),
            nn.Linear(DIM_HIDDEN, DIM_HIDDEN),
            nn.ReLU(),
            nn.Linear(DIM_HIDDEN, DIM_HIDDEN),
            nn.ReLU(),
            nn.Linear(DIM_HIDDEN, DIM_STATE)
        )

    def forward(self, obs, num_par=NUM_PAR_PF):
        obs_enc = self.obs_encode(obs)  # (B, C)
        obs_enc = obs_enc.repeat(num_par, 1)  # (B * num_par, C)
        z = torch.randn_like(obs_enc)  # (B * num_par, C)
        x = torch.cat([obs_enc, z], -1)  # (B * num_par, 2C)
        proposal = self.p_net(x)  # [B * num_par, 2]
        return proposal

class MeasureNetwork(nn.Module):
    def __init__(self):
        super(MeasureNetwork, self).__init__()
        self.dim_m = 16
        self.obs_encode = nn.Sequential(
            nn.Linear(DIM_OBS, DIM_HIDDEN),
            nn.ReLU(),
            nn.Linear(DIM_HIDDEN, DIM_HIDDEN),
            nn.ReLU(),
            nn.Linear(DIM_HIDDEN, DIM_HIDDEN),
            nn.ReLU()
        )
        # self.lstm = nn.LSTM(DIM_HIDDEN, DIM_LSTM_HIDDEN, NUM_LSTM_LAYER)
        # self.lstm_out = nn.Sequential(
        #     nn.Linear(DIM_LSTM_HIDDEN, self.dim_m),
        #     nn.ReLU()
        # )
        self.m_net = nn.Sequential(
            #nn.Linear(self.dim_m + DIM_STATE, DIM_HIDDEN),
            nn.Linear(DIM_HIDDEN + DIM_STATE, DIM_HIDDEN),
            nn.ReLU(),
            nn.Linear(DIM_HIDDEN, DIM_HIDDEN),
            nn.ReLU(),
            nn.Linear(DIM_HIDDEN, 1),
            nn.Sigmoid()
        )

    def m_model(self, state, obs, num_par=NUM_PAR_PF):
        # state: (B * K, dim_s)
        # obs: (B, dim_s)
        obs_enc = self.obs_encode(obs)  # (batch, dim_m)
        #x = obs_enc.unsqueeze(0)  # -> [1, batch_size, dim_obs]
        #x, (h, c) = self.lstm(x, (hidden, cell))
        #x = self.lstm_out(x[0])  # (batch, dim_m)
        x = obs_enc
        x = x.repeat(num_par, 1)  # (batch * num_par, dim_m)
        x = torch.cat((x, state), -1)  # (batch * num_par, dim_m + 2)
        lik = self.m_net(x).view(-1, num_par)  # (batch, num_par)
        return lik#, h, c



MSE_criterion = nn.MSELoss()
BCE_criterion = nn.BCELoss()
# Filtering
measure_net = MeasureNetwork()
pp_net = ProposerNetwork()
measure_optimizer = Adam(measure_net.parameters(), lr=FIL_LR)
pp_optimizer = Adam(pp_net.parameters(), lr=FIL_LR)

#state_batch, action_batch, reward_batch, next_state_batch, done_batch, \
#    obs, curr_par, mean_state, hidden, cell, pf_sample = self.replay_buffer.sample(BATCH_SIZE)
#state_batch = torch.FloatTensor(state_batch)
#next_state_batch = torch.FloatTensor(next_state_batch)
#action_batch = torch.FloatTensor(action_batch)
#reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1)  # (B, 1)
#mask_batch = torch.FloatTensor(np.float32(1 - done_batch)).unsqueeze(1)
#curr_obs = torch.FloatTensor(obs)
#curr_par = torch.FloatTensor(curr_par)  # (B, K, dim_s)
#mean_state = torch.FloatTensor(mean_state) # (B, dim_s)
#curr_par_sample = torch.FloatTensor(pf_sample) # (B, M, 2)
#hidden = torch.FloatTensor(hidden) # [128, NUM_LSTM_LAYER, 1, DIM_LSTM_HIDDEN]
#hidden = torch.transpose(torch.squeeze(hidden), 0, 1).contiguous()
#cell = torch.FloatTensor(cell)
#cell = torch.transpose(torch.squeeze(cell), 0, 1).contiguous()


def make_simple_batch(batch_size):
    states_batch = np.random.rand(batch_size, 2)
    #states_batch = np.tile(np.random.rand(2), batch_size).reshape((batch_size, 2))
    obs_batch = states_batch + np.random.normal(0, 0.1, (batch_size, 2))
    states_batch = torch.from_numpy(states_batch).float()
    obs_batch = torch.from_numpy(obs_batch).float()

    return states_batch, obs_batch

def make_batch(batch_size):
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


num_training_steps = 40000
for step in range(num_training_steps):
    print(step)

    #state_batch, curr_obs = make_simple_batch(BATCH_SIZE)
    state_batch, curr_obs = make_batch(BATCH_SIZE)

    # ------------------------
    #  Train Particle Proposer
    # ------------------------
    pp_optimizer.zero_grad()
    state_propose = pp_net(curr_obs, NUM_PAR_PF)
    PP_loss = 0

    fake_logit = measure_net.m_model(state_propose, curr_obs, NUM_PAR_PF)  # (B, K)
    real_target = torch.ones_like(fake_logit)
    PP_loss += BCE_criterion(fake_logit, real_target)

    PP_loss.backward()
    pp_optimizer.step()

    # ------------------------
    #  Train Observation Model
    # ------------------------
    measure_optimizer.zero_grad()
    curr_par = torch.from_numpy(np.random.rand(NUM_PAR_PF * BATCH_SIZE, 2)).float()
    fake_logit = measure_net.m_model(curr_par.view(-1, DIM_STATE), curr_obs, NUM_PAR_PF)  # (B, K)

    fake_logit_pp = measure_net.m_model(state_propose.detach(), curr_obs, NUM_PAR_PF)  # (B, K)
    fake_logit = torch.cat((fake_logit, fake_logit_pp), -1)  # (B, 2K)
    #fake_logit = fake_logit_pp

    fake_target = torch.zeros_like(fake_logit)
    fake_loss = BCE_criterion(fake_logit, fake_target)
    real_logit = measure_net.m_model(state_batch, curr_obs, 1)  # (batch, num_pars)
    real_target = torch.ones_like(real_logit)
    real_loss = BCE_criterion(real_logit, real_target)
    OM_loss = real_loss + fake_loss
    OM_loss.backward()
    measure_optimizer.step()


for j in range(6):
    env = Environment()
    state = torch.from_numpy(env.state).reshape((1, DIM_STATE))
    print("STATE", state)
    states_batch = torch.cat(BATCH_SIZE * [state]).float()
    obs_batch = torch.from_numpy(np.array([env.get_observation() for _ in range(BATCH_SIZE)])).float()

    # state = torch.from_numpy(np.random.rand(2)).reshape((1, DIM_STATE))
    # states_batch = torch.cat(BATCH_SIZE * [state]).float()
    # obs_batch = states_batch.numpy() + np.random.normal(0, 0.1, (BATCH_SIZE, 2))
    # obs_batch = torch.from_numpy(obs_batch).float()

    state_predicted = pp_net(obs_batch, NUM_PAR_PF)

    plt.scatter([state[0][0]], [state[0][1]], color='k')
    plt.scatter([obs[0] for obs in obs_batch], [obs[1] for obs in obs_batch], color='g')
    plt.scatter([state[0] for state in state_predicted.detach().numpy()],
                [state[1] for state in state_predicted.detach().numpy()], color='r')

    if DIM_OBS == 4:
        plt.scatter([obs[2] for obs in obs_batch], [obs[3] for obs in obs_batch], color='b')

    plt.show()

