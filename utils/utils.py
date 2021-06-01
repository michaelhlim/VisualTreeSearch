# author: @wangyunbo, @liubo
import numpy as np
import datetime
import torch
import torch.nn as nn
import os
#from configs.environments.floor import *
#from configs.solver.dualsmc import *

CONST = 1e-6

def check_path(path):
    if not os.path.exists(path):
        print("[INFO] making folder %s" % path)
        os.makedirs(path)


def get_datetime():
    currentDT = datetime.datetime.now()
    return currentDT.strftime("%m-%d-%H_%M_%S")


def detect_collision(curr_state, next_state):
    if len(curr_state.shape) == 1:
        cx = curr_state[0]
        cy = curr_state[1]
        nx = next_state[0]
        ny = next_state[1]
    elif len(curr_state.shape) == 2:
        cx = curr_state[:, 0]
        cy = curr_state[:, 1]
        nx = next_state[:, 0]
        ny = next_state[:, 1]
    cond_hit = (nx < 0) + (nx > 2) + (ny < 0) + (ny > 1)  # hit the surrounding walls
    cond_hit += (cy <= 0.5) * (ny > 0.5)  # cross the middle wall
    cond_hit += (cy >= 0.5) * (ny < 0.5)  # cross the middle wall
    cond_hit += (cx <= 1.2) * (nx > 1.2) * ((cy > 0.9) * (cy < 1) + (ny > 0.9) * (ny < 1))
    cond_hit += (cx >= 1.2) * (nx < 1.2) * ((cy > 0.9) * (cy < 1) + (ny > 0.9) * (ny < 1))
    cond_hit += (cx <= 0.4) * (nx > 0.4) * ((cy > 0.4) * (cy < 0.5) + (ny > 0.4) * (ny < 0.5))
    cond_hit += (cx >= 0.4) * (nx < 0.4) * ((cy > 0.4) * (cy < 0.5) + (ny > 0.4) * (ny < 0.5))
    cond_hit += (cx <= 1.2) * (nx > 1.2) * ((cy > 0.5) * (cy < 0.6) + (ny > 0.5) * (ny < 0.6))
    cond_hit += (cx >= 1.2) * (nx < 1.2) * ((cy > 0.5) * (cy < 0.6) + (ny > 0.5) * (ny < 0.6))
    cond_hit += (cx <= 0.4) * (nx > 0.4) * ((cy > 0.0) * (cy < 0.1) + (ny > 0.0) * (ny < 0.1))
    cond_hit += (cx >= 0.4) * (nx < 0.4) * ((cy > 0.0) * (cy < 0.1) + (ny > 0.0) * (ny < 0.1))
    return cond_hit


def l2_distance(state, goal):
    if len(state.shape) == 1:
        dist = np.power((state[0] - goal[0]), 2) + np.power((state[1] - goal[1]), 2) + CONST
    elif len(state.shape) == 2:
        dist = (state[:, 0] - goal[:, 0]).pow(2) + (state[:, 1] - goal[:, 1]).pow(2) + CONST
    elif len(state.shape) == 3:
        dist = (state[:, :, 0] - goal[:, :, 0]).pow(2) + (state[:, :, 1] - goal[:, :, 1]).pow(2) + CONST
    return dist

def l2_distance_np(states, goals):
    dist = np.sum(np.power((states - goals), 2), axis=1) + CONST
    return dist


#########################
# Planning Network
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)



