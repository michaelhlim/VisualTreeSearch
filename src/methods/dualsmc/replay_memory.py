from utils.utils import *
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam
from configs.solver.dualsmc import *


########################
# Training Buffer
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, s_t, a_t, r_t, s_tp1, done, obs, curr_ps, mean_state, hidden, cell, pf_sample):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (s_t, a_t, r_t, s_tp1, done, obs, curr_ps, mean_state, hidden, cell, pf_sample)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, obs, curr_ps, mean_state, hidden, cell, pf_sample = map(
            np.stack, zip(*batch))
        return state, action, reward, next_state, done, obs, curr_ps, mean_state, hidden, cell, pf_sample

    def __len__(self):
        return len(self.buffer)