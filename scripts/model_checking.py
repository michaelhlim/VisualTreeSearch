import numpy as np
import os.path
import torch

from configs.environments.stanford import *
from configs.solver.dualsmc_lightdark import *
from src.solvers.dualsmc_lightdark import DualSMC

dlp = DualSMC_LightDark_Params()
sep = Stanford_Environment_Params()

model = DualSMC()
load_path = "dualsmc07-01-23_17_21/dpf_online"
cwd = os.getcwd()
model.load_model(cwd + "/nets/" + load_path)

#state_arr = [3, 0.3]
#state_arr = [8, 1.4]
state_arr = [4, 0.3]
state = torch.FloatTensor(state_arr).unsqueeze(0).to(dlp.device)
action_mean, action_logstd = model.policy(state.repeat(1, 4))
qf1, qf2 = model.critic(state, action_mean)

action_mean = [[(3*np.pi/2 - np.pi)/np.pi]]
action_mean = torch.FloatTensor(action_mean).to(dlp.device)
qf11, qf21 = model.critic(state, action_mean)

next_state = model.dynamic_net.t_model(state, action_mean) 

new_theta = action_mean.detach().cpu().numpy() * np.pi + np.pi
new_theta = new_theta[0][0]
vector = np.array([np.cos(new_theta), np.sin(new_theta)]) * sep.velocity  # Go in the direction the new theta is
true_next_state = state_arr + vector

print("Action mean:", action_mean, "Action std:", torch.exp(action_logstd))
print("Action theta:", new_theta)
print("Predicted next state:", next_state, "True next state:", true_next_state)
print("Q-Values:", qf1, qf2, qf11, qf21)

