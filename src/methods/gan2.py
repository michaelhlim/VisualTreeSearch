import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

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
DIM_OBS = 2
DIM_HIDDEN = 256
NUM_PAR_PF = 1   # num particles
FIL_LR = 1e-3 # filtering

DIM_LSTM_HIDDEN = 128
NUM_LSTM_LAYER = 2

BATCH_SIZE = 64

# Observation Predictor
class ObsPredictorNetwork(nn.Module):
    def __init__(self):
        super(ObsPredictorNetwork, self).__init__()
        self.dim = 64
        self.state_encode = nn.Sequential(
            nn.Linear(DIM_STATE, DIM_HIDDEN),
            nn.ReLU(),
            nn.Linear(DIM_HIDDEN, DIM_HIDDEN),
            nn.ReLU(),
            nn.Linear(DIM_HIDDEN, self.dim),
            nn.ReLU()
        )
        self.op_net = nn.Sequential(
            nn.Linear(self.dim * 2, DIM_HIDDEN),
            nn.ReLU(),
            nn.Linear(DIM_HIDDEN, DIM_HIDDEN),
            nn.ReLU(),
            nn.Linear(DIM_HIDDEN, DIM_HIDDEN),
            nn.ReLU(),
            nn.Linear(DIM_HIDDEN, DIM_OBS)
        )

    def forward(self, state, num_par=NUM_PAR_PF):
        state_enc = self.state_encode(state)  # (B, C)
        state_enc = state_enc.repeat(num_par, 1)  # (B * num_par, C)
        z = torch.randn_like(state_enc)  # (B * num_par, C)
        x = torch.cat([state_enc, z], -1)  # (B * num_par, 2C)
        obs_prediction = self.op_net(x)  # [B * num_par, 2]
        return obs_prediction

# Observation Model
class MeasureNetwork(nn.Module):
    def __init__(self):
        super(MeasureNetwork, self).__init__()
        self.dim_m = 16
        self.state_encode = nn.Sequential(
            nn.Linear(DIM_STATE, DIM_HIDDEN),
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
            nn.Linear(DIM_HIDDEN + DIM_OBS, DIM_HIDDEN),
            nn.ReLU(),
            nn.Linear(DIM_HIDDEN, DIM_HIDDEN),
            nn.ReLU(),
            nn.Linear(DIM_HIDDEN, 1),
            nn.Sigmoid()
        )

    def m_model(self, state, obs, num_par=NUM_PAR_PF):
        # state: (B * K, dim_s)
        # obs: (B, dim_s)
        #obs_enc = self.obs_encode(obs)  # (batch, dim_m)
        state_enc = self.state_encode(state)  # (batch, dim_m)

        #x = obs_enc.unsqueeze(0)  # -> [1, batch_size, dim_obs]
        #x, (h, c) = self.lstm(x, (hidden, cell))
        #x = self.lstm_out(x[0])  # (batch, dim_m)

        #x = obs_enc
        x = state_enc
        x = x.repeat(num_par, 1)  # (batch * num_par, dim_m)
        #x = torch.cat((x, state), -1)  # (batch * num_par, dim_m + 2)
        x = torch.cat((x, obs), -1)  # (batch * num_par, dim_m + 2)
        lik = self.m_net(x).view(-1, num_par)  # (batch, num_par)
        return lik#, h, c



MSE_criterion = nn.MSELoss()
BCE_criterion = nn.BCELoss()
# Filtering
measure_net = MeasureNetwork()
op_net = ObsPredictorNetwork()
measure_optimizer = Adam(measure_net.parameters(), lr=FIL_LR)
op_optimizer = Adam(op_net.parameters(), lr=FIL_LR)


def make_simple_batch(batch_size):
    states_batch = np.random.rand(batch_size, 2)
    #states_batch = np.tile(np.random.rand(2), batch_size).reshape((batch_size, 2))
    obs_batch = states_batch + np.random.normal(0, 0.1, (batch_size, 2))
    states_batch = torch.from_numpy(states_batch).float()
    obs_batch = torch.from_numpy(obs_batch).float()

    return states_batch, obs_batch


def make_simple_batch_multiple_modes(batch_size):
    rand_mode0 = np.random.randint(batch_size/3)
    rand_mode1 = np.random.randint(2*batch_size/3 - rand_mode0)
    rand_mode2 = batch_size - (rand_mode0 + rand_mode1)

    mode0_batch = np.tile(np.array([0., 0.]), rand_mode0).reshape((rand_mode0, 2))
    mode1_batch = np.tile(np.array([1., 0.]), rand_mode1).reshape((rand_mode1, 2))
    mode2_batch = np.tile(np.array([0., 1.]), rand_mode2).reshape((rand_mode2, 2))
    states_batch = np.vstack([mode0_batch, mode1_batch, mode2_batch])
    np.random.shuffle(states_batch)

    obs_batch = states_batch + np.random.normal(0, 0.1, (batch_size, 2))
    states_batch = torch.from_numpy(states_batch).float()
    obs_batch = torch.from_numpy(obs_batch).float()

    return states_batch, obs_batch


num_training_steps = 5000
for step in range(num_training_steps):
    print(step)

    state_batch, curr_obs = make_simple_batch(BATCH_SIZE)
    #state_batch, curr_obs = make_simple_batch_multiple_modes(BATCH_SIZE)

    # ----------------------------
    #  Train Observation Predictor
    # ----------------------------
    op_optimizer.zero_grad()
    obs_predicted = op_net(state_batch, NUM_PAR_PF)
    OP_loss = 0

    fake_logit = measure_net.m_model(state_batch, obs_predicted, NUM_PAR_PF)  # (B, K)
    real_target = torch.ones_like(fake_logit)
    OP_loss += BCE_criterion(fake_logit, real_target)

    OP_loss.backward()
    op_optimizer.step()

    # ------------------------
    #  Train Observation Model
    # ------------------------
    measure_optimizer.zero_grad()
    curr_par = torch.from_numpy(np.random.rand(NUM_PAR_PF * BATCH_SIZE, 2)).float()
    fake_logit = measure_net.m_model(state_batch, curr_par.view(-1, DIM_OBS), NUM_PAR_PF)  # (B, K)

    fake_logit_op = measure_net.m_model(state_batch, obs_predicted.detach(), NUM_PAR_PF)  # (B, K)
    fake_logit = torch.cat((fake_logit, fake_logit_op), -1)  # (B, 2K)
    #fake_logit = fake_logit_pp

    fake_target = torch.zeros_like(fake_logit)
    fake_loss = BCE_criterion(fake_logit, fake_target)
    real_logit = measure_net.m_model(state_batch, curr_obs, 1)  # (batch, num_pars)
    real_target = torch.ones_like(real_logit)
    real_loss = BCE_criterion(real_logit, real_target)
    OM_loss = real_loss + fake_loss
    OM_loss.backward()
    measure_optimizer.step()


for j in range(2):
    state = torch.from_numpy(np.random.rand(2)).reshape((1, 2))
    #state = torch.from_numpy(np.array([0., 0.])).reshape((1, 2))
    states_batch = torch.cat(BATCH_SIZE * [state]).float()
    obs_batch = states_batch.numpy() + np.random.normal(0, 0.1, (BATCH_SIZE, 2))
    #obs_batch = torch.from_numpy(obs_batch).float()
    obs_predicted = op_net(states_batch, NUM_PAR_PF)

    plt.scatter([state[0][0]], [state[0][1]], color='k')
    plt.scatter([obs[0] for obs in obs_batch], [obs[1] for obs in obs_batch], color='g')
    plt.scatter([obs[0] for obs in obs_predicted.detach().numpy()],
                [obs[1] for obs in obs_predicted.detach().numpy()], color='r')
    plt.show()


    # state = torch.from_numpy(np.array([1., 0.])).reshape((1, 2))
    # states_batch = torch.cat(BATCH_SIZE * [state]).float()
    # obs_batch = states_batch.numpy() + np.random.normal(0, 0.1, (BATCH_SIZE, 2))
    # # obs_batch = torch.from_numpy(obs_batch).float()
    # obs_predicted = op_net(states_batch, NUM_PAR_PF)
    #
    # plt.scatter([state[0][0]], [state[0][1]], color='k')
    # plt.scatter([obs[0] for obs in obs_batch], [obs[1] for obs in obs_batch], color='g')
    # plt.scatter([obs[0] for obs in obs_predicted.detach().numpy()],
    #             [obs[1] for obs in obs_predicted.detach().numpy()], color='r')
    # plt.show()
    #
    #
    # state = torch.from_numpy(np.array([0., 1.])).reshape((1, 2))
    # states_batch = torch.cat(BATCH_SIZE * [state]).float()
    # obs_batch = states_batch.numpy() + np.random.normal(0, 0.1, (BATCH_SIZE, 2))
    # # obs_batch = torch.from_numpy(obs_batch).float()
    # obs_predicted = op_net(states_batch, NUM_PAR_PF)
    #
    # plt.scatter([state[0][0]], [state[0][1]], color='k')
    # plt.scatter([obs[0] for obs in obs_batch], [obs[1] for obs in obs_batch], color='g')
    # plt.scatter([obs[0] for obs in obs_predicted.detach().numpy()],
    #             [obs[1] for obs in obs_predicted.detach().numpy()], color='r')
    # plt.show()

