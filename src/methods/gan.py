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
DIM_OBS = 2
DIM_HIDDEN = 256
NUM_PAR_PF = 100   # num particles

DIM_LSTM_HIDDEN = 128
NUM_LSTM_LAYER = 2


# Observation Predictor
class ObsPredictorNetwork(nn.Module):
    def __init__(self):
        super(ObsPredictorNetwork, self).__init__()
        self.dim = 64
        #self.emb = nn.Embedding(DIM_STATE, 1)
        self.state_encode = nn.Sequential(
            nn.Linear(DIM_STATE, DIM_HIDDEN),
            #nn.BatchNorm1d(DIM_HIDDEN, 0.8),
            #nn.LeakyReLU(0.2),
            nn.ReLU(),
            nn.Linear(DIM_HIDDEN, DIM_HIDDEN),
            #nn.BatchNorm1d(DIM_HIDDEN, 0.8),
            #nn.LeakyReLU(0.2),
            nn.ReLU(),
            nn.Linear(DIM_HIDDEN, self.dim),
            #nn.LeakyReLU(0.2)
            nn.ReLU()
        )
        self.op_net = nn.Sequential(
            nn.Linear(self.dim * 2, DIM_HIDDEN),
            #nn.BatchNorm1d(DIM_HIDDEN, 0.8),
            #nn.LeakyReLU(0.2),
            nn.ReLU(),
            nn.Linear(DIM_HIDDEN, DIM_HIDDEN),
            #nn.BatchNorm1d(DIM_HIDDEN, 0.8),
            #nn.LeakyReLU(0.2),
            nn.ReLU(),
            nn.Linear(DIM_HIDDEN, DIM_HIDDEN),
            #nn.BatchNorm1d(DIM_HIDDEN, 0.8),
            #nn.LeakyReLU(0.2),
            nn.ReLU(),
            nn.Linear(DIM_HIDDEN, DIM_OBS)
        )


    def forward(self, state, num_obs=1):
        state_enc = self.state_encode(state)  # (batch, self.dim)
        #state_enc = self.emb(state.long())
        #state_enc = state_enc.squeeze(2)
        state_enc = state_enc.repeat(num_obs, 1)  # (batch * num_obs, self.dim)
        z = torch.randn_like(state_enc)  # (batch * num_obs, self.dim)
        x = torch.cat([state_enc, z], -1)  # (batch * num_obs, 2 * self.dim)
        #x = torch.multiply(state_enc, z)
        #x = state_enc
        obs_prediction = self.op_net(x)  # [batch * num_obs, 2]
        return obs_prediction

# Observation Model
class MeasureNetwork(nn.Module):
    def __init__(self):
        super(MeasureNetwork, self).__init__()
        #self.dim_m = 16
        self.obs_encode = nn.Sequential(
            nn.Linear(DIM_OBS, DIM_HIDDEN),
            #nn.Dropout(0.4),
            #nn.LeakyReLU(0.2),
            nn.ReLU(),
            nn.Linear(DIM_HIDDEN, DIM_HIDDEN),
            #nn.Dropout(0.4),
            #nn.LeakyReLU(0.2),
            nn.ReLU(),
            nn.Linear(DIM_HIDDEN, DIM_HIDDEN),
            #nn.Dropout(0.4),
            #nn.LeakyReLU(0.2)
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
            #nn.Dropout(0.4),
            #nn.LeakyReLU(0.2),
            nn.ReLU(),
            nn.Linear(DIM_HIDDEN, DIM_HIDDEN),
            #nn.Dropout(0.4),
            #nn.LeakyReLU(0.2),
            nn.ReLU(),
            nn.Linear(DIM_HIDDEN, 1),
            nn.Sigmoid()   # Comment this line out for Wasserstein
        )

    def m_model(self, state, obs, num_obs=1):
        # state: (B * K, dim_s)
        # obs: (B, dim_s)
        obs_enc = self.obs_encode(obs)  # (batch, self.dim)
        obs_enc = obs_enc.repeat(num_obs, 1)  # (batch * num_obs, DIM_HIDDEN)
        x = torch.cat((obs_enc, state), -1)   # (batch * num_obs, DIM_HIDDEN + 2)
        likelihood = self.m_net(x).view(-1, num_obs)  # (batch, num_obs)
        return likelihood

        # obs_enc = self.obs_encode(obs)  # (batch, dim_m)
        # x = obs_enc.unsqueeze(0)  # -> [1, batch_size, dim_obs]
        # x, (h, c) = self.lstm(x, (hidden, cell))
        # x = self.lstm_out(x[0])  # (batch, dim_m)
        # x = x.repeat(num_par, 1)  # (batch * num_par, dim_m)
        # x = torch.cat((x, state), -1)  # (batch * num_par, dim_m + 2)
        # lik = self.m_net(x).view(-1, num_par)  # (batch, num_par)
        # return lik, h, c


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

def make_simple_batch_old(batch_size):
    #states_batch = np.zeros((batch_size, 2))
    states_batch = np.tile(np.random.rand(2), batch_size).reshape((batch_size, 2))
    obs_batch = states_batch + np.random.normal(0, 1, (batch_size, 2))
    states_batch = torch.from_numpy(states_batch).float()
    obs_batch = torch.from_numpy(obs_batch).float()

    return states_batch, obs_batch


def make_simple_batch(batch_size):
    states_batch = np.random.rand(batch_size, 2)
    #states_batch = np.tile(np.random.rand(2), batch_size).reshape((batch_size, 2))
    obs_batch = states_batch + np.random.normal(0, 0.1, (batch_size, 2))
    states_batch = torch.from_numpy(states_batch).float()
    obs_batch = torch.from_numpy(obs_batch).float()

    return states_batch, obs_batch

def make_simple_batch_mode(batch_size, mode):
    if mode == 0:
        state = np.array([0., 0.])
    if mode == 1:
        state = np.array([1., 0.])
    if mode == 2:
        state = np.array([0., 1.])

    states_batch = np.tile(state, batch_size).reshape((batch_size, 2))
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


class Trainer():
    def __init__(self, args):
        self.measure_net = MeasureNetwork().float()
        self.op_net = ObsPredictorNetwork().float()
        self.MSE_criterion = nn.MSELoss()
        self.BCE_criterion = nn.BCELoss()
        self.lr = args['lr']
        self.betas = args['betas']
        self.measure_optimizer = Adam(self.measure_net.parameters(), lr=self.lr, betas=self.betas)
        #self.measure_optimizer = RMSprop(self.measure_net.parameters(), lr=0.00005)
        #self.op_optimizer = Adam(self.op_net.parameters(), lr=self.lr, betas=self.betas)
        self.op_optimizer = RMSprop(self.op_net.parameters(), lr=0.00005)
        self.batch_size = args['batch_size']
        self.num_training_steps = args['num_training_steps']
        self.print_freq = args['print_freq']
        self.chkpt_freq = args['chkpt_freq']
        self.chkpt_model = args['chkpt_model']
        self.measure_model_path = args['measure_model_path']
        self.op_model_path = args['op_model_path']
        self.measure_pickle_path = args['measure_pickle_path']
        self.op_pickle_path = args['op_pickle_path']
        self.generator_freq = args['generator_freq']

        if self.chkpt_model:
            eventfiles = glob.glob(self.measure_model_path + '*')
            eventfiles.sort(key=os.path.getmtime)
            path = eventfiles[-1]
            self.measure_net.load_state_dict(torch.load(path))

            eventfiles = glob.glob(self.op_model_path + '*')
            eventfiles.sort(key=os.path.getmtime)
            path = eventfiles[-1]
            self.op_net.load_state_dict(torch.load(path))

            self.m_losses = pickle.load(open(self.measure_pickle_path, "rb"))
            self.op_losses = pickle.load(open(self.op_pickle_path, "rb"))
            self.start_step = self.m_losses[-1][0]
        else:
            self.m_losses = []
            self.op_losses = []
            self.start_step = -1

    def train(self):
        t1 = time.time()
        #init_momentum_op = self.op_optimizer.param_groups[0]["momentum"]
        #final_momentum_op = 0.99
        #num_steps_interpolate = 30000
        #slope = (final_momentum_op - init_momentum_op)/num_steps_interpolate
        for step in range(self.num_training_steps):
            # if step > self.num_training_steps - num_steps_interpolate:
            #    momentum = slope * (step - (self.num_training_steps - num_steps_interpolate))
            #    self.op_optimizer.param_groups[0]["momentum"] = init_momentum_op + momentum

            real_step = self.start_step + step + 1

            # Make batch of training data
            #states_batch, obs_batch = make_batch(self.batch_size)
            states_batch, obs_batch = make_simple_batch(self.batch_size)

            #mode = np.random.randint(3)
            #states_batch, obs_batch = make_simple_batch_mode(self.batch_size, mode)

            #states_batch, obs_batch = make_simple_batch_multiple_modes(self.batch_size)

            # ------------------------
            #  Train Observation Predictor
            # ------------------------
            self.op_optimizer.zero_grad()
            obs_predicted = self.op_net(states_batch)
            fake_logit = self.measure_net.m_model(states_batch, obs_predicted)  # (B, K)

            if self.generator_freq != 1:  # Wasserstein
                OP_loss = -torch.mean(fake_logit)
            else:
                OP_loss = 0
                real_target = torch.ones_like(fake_logit)
                OP_loss += self.BCE_criterion(fake_logit, real_target)

            OP_loss.backward()
            self.op_optimizer.step()

            # ------------------------
            #  Train Observation Model
            # ------------------------
            if step % 1 == 0:
            #if step % self.generator_freq == 0:
                self.measure_optimizer.zero_grad()
                fake_logit_op = self.measure_net.m_model(states_batch, obs_predicted.detach())  # (B, K)
                real_logit = self.measure_net.m_model(states_batch, obs_batch)  # (batch, num_obs)

                curr_obs = torch.from_numpy(np.random.rand(self.batch_size, 2)).float()
                fake_logit = self.measure_net.m_model(states_batch, curr_obs.view(-1, DIM_OBS))  # (B, K)
                fake_logit = torch.cat((fake_logit, fake_logit_op), -1)  # (B, 2K)

                if self.generator_freq != 1:  # Wasserstein
                    OM_loss = -torch.mean(real_logit) + torch.mean(fake_logit)
                else:
                    fake_target = torch.zeros_like(fake_logit)
                    fake_loss = self.BCE_criterion(fake_logit, fake_target)
                    real_target = torch.ones_like(real_logit)
                    real_loss = self.BCE_criterion(real_logit, real_target)
                    OM_loss = real_loss + 2.5 * fake_loss

                OM_loss.backward()
                self.measure_optimizer.step()

                if self.generator_freq != 1:
                    clip_value = 0.01
                    for p in self.measure_net.parameters():
                        p.data.clamp_(-clip_value, clip_value)


            if step % self.print_freq == 0:
                print("STEP", real_step, "OP_LOSS", OP_loss.item(), "M_LOSS", OM_loss.item())
                self.op_losses.append((real_step, OP_loss.item()))
                self.m_losses.append((real_step, OM_loss.item()))
                #print("OP MOMENTUM", self.op_optimizer.param_groups[0]["momentum"])

            if step % self.chkpt_freq == 0:
                torch.save(self.measure_net.state_dict(), self.measure_model_path + str(real_step))
                pickle.dump(self.m_losses, open(self.measure_pickle_path, "wb"))
                torch.save(self.op_net.state_dict(), self.op_model_path + str(real_step))
                pickle.dump(self.op_losses, open(self.op_pickle_path, "wb"))

        t2 = time.time()
        print("Finished training. That took", t2 - t1, "seconds.")


class Tester():
    def __init__(self, args):
        self.measure_net = MeasureNetwork().float()
        self.op_net = ObsPredictorNetwork().float()
        self.batch_size = args['batch_size']
        self.measure_model_path = args['measure_model_path']
        self.op_model_path = args['op_model_path']
        self.measure_pickle_path = args['measure_pickle_path']
        self.op_pickle_path = args['op_pickle_path']

        eventfiles = glob.glob(self.measure_model_path + '*')
        eventfiles.sort(key=os.path.getmtime)
        path = eventfiles[-1]
        self.measure_net.load_state_dict(torch.load(path))

        eventfiles = glob.glob(self.op_model_path + '*')
        eventfiles.sort(key=os.path.getmtime)
        path = eventfiles[-1]
        self.op_net.load_state_dict(torch.load(path))

        self.m_losses = pickle.load(open(self.measure_pickle_path, "rb"))
        self.op_losses = pickle.load(open(self.op_pickle_path, "rb"))

    def test(self):
        # obs_batch = []
        # env = Environment()
        # state = torch.from_numpy(env.state).reshape((1, 2))
        # states_batch = torch.cat(self.batch_size*[state]).float()
        # for _ in range(self.batch_size):
        #     obs = env.get_observation()
        #     obs_batch.append(obs)
        # obs_predicted = self.op_net(states_batch)


        # state = torch.from_numpy(np.random.rand(2) + 0 * np.ones(2)).reshape((1, 2))
        # states_batch = torch.cat(self.batch_size * [state]).float()
        # obs_batch = states_batch.numpy() + np.random.normal(0, 1, (self.batch_size, 2))
        # obs_predicted = self.op_net(states_batch)


        #state = np.array([np.zeros(2)])
        # states = [ np.array([np.array([0., 0.])]), np.array([np.array([1., 0.])]),
        #           np.array([np.array([0., 1.])]) ]
        states = [np.array([np.random.rand(2) + 0 * np.ones(2)]), np.array([np.random.rand(2) + 0 * np.ones(2)]),
                  np.array([np.random.rand(2) + 0 * np.ones(2)])]
        for state in states:
            print("STATE", state)
            #state = np.array([np.array([1., 0.])])
            states_batch = np.tile(state, self.batch_size).reshape((self.batch_size, 2))
            obs_batch = states_batch + np.random.normal(0, 0.1, (self.batch_size, 2))
            states_batch = torch.from_numpy(states_batch).float()
            obs_predicted = self.op_net(states_batch)


            plt.scatter([state[0][0]], [state[0][1]], color='k')
            plt.scatter([obs[0] for obs in obs_batch], [obs[1] for obs in obs_batch], color='g')
            #plt.scatter([obs[2] for obs in obs_batch], [obs[3] for obs in obs_batch], color='g')
            plt.scatter([obs[0] for obs in obs_predicted.detach().numpy()],
                        [obs[1] for obs in obs_predicted.detach().numpy()], color='r')
            #plt.scatter([obs[2] for obs in obs_predicted.detach().numpy()],
            #            [obs[3] for obs in obs_predicted.detach().numpy()], color='b')
            plt.show()

            obs_batch = np.array(obs_batch)
            obs_batch_mean = np.mean(obs_batch[:, :2], axis=0)
            obs_batch_std = np.std(obs_batch[:, :2], axis=0)
            print("OBS_BATCH_MEAN\n", obs_batch_mean)
            print("OBS_BATCH_STD\n", obs_batch_std)
            obs_predicted_mean = np.mean(obs_predicted[:, :2].detach().numpy(), axis=0)
            obs_predicted_std = np.std(obs_predicted[:, :2].detach().numpy(), axis=0)
            print("OBS_PREDICTED_MEAN\n", obs_predicted_mean)
            print("OBS_PREDICTED_STD\n", obs_predicted_std)

            # states_batch, obs_batch = make_batch(self.batch_size)
            #
            # obs_predicted = self.op_net(states_batch)
            # print("OBS_BATCH\n", obs_batch)
            # print("OBS_PREDICTED\n", obs_predicted)
            # print("DIFF OBS_BATCH AND OBS_PREDICTED\n", torch.norm(obs_predicted - obs_batch, dim=-1),
            #       torch.mean(torch.norm(obs_predicted - obs_batch, dim=-1)))
            # print("DIFF OBS_PREDICTED AND STATES\n", obs_predicted[:, :2] - states_batch)
            # print("DIFF OBS_BATCH AND STATES\n", obs_batch[:, :2] - states_batch)

            obs_batch = torch.from_numpy(obs_batch).float()
            probabilities_fake = self.measure_net.m_model(states_batch, obs_predicted.detach())
            print("PROBABILITIES FOR FAKE DATA\n", torch.mean(probabilities_fake))
            probabilities_real = self.measure_net.m_model(states_batch, obs_batch)
            print("PROBABILITIES FOR REAL DATA\n", torch.mean(probabilities_real))
            print("\n")

            # states_batch = states_batch.numpy()
            # diff_obs_pred_states = obs_predicted[:, :2].detach().numpy() - states_batch
            # probabilities_fake_gaussian = norm.pdf(diff_obs_pred_states, 0, 0.01)
            # probabilities_fake_gaussian = np.prod(probabilities_fake_gaussian, axis=1)
            # print("GAUSSIAN PROBABILITIES FOR FAKE DATA\n", np.mean(probabilities_fake_gaussian))
            # diff_obs_batch_states = obs_batch[:, :2] - states_batch
            # probabilities_real_gaussian = norm.pdf(diff_obs_batch_states, 0, 0.01)
            # probabilities_real_gaussian = np.prod(probabilities_real_gaussian, axis=1)
            # print("GAUSSIAN PROBABILITIES FOR REAL DATA\n", np.mean(probabilities_real_gaussian))



    def plot_training_losses(self):
        steps = [loss[0] for loss in self.op_losses]

        op_losses = [loss[1] for loss in self.op_losses]
        plt.plot(steps, op_losses, label='op_losses')
        plt.xlabel("Num training steps")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

        m_losses = [loss[1] for loss in self.m_losses]
        plt.plot(steps, m_losses, label='m_losses')
        plt.xlabel("Num training steps")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()


def argparser():
    arguments = sys.argv[1:]

    chkpt_model = False

    if "train" in arguments:
        TRAIN = True
        index = arguments.index("train")
        try:
            chkpt_model = (arguments[index + 1] == "chkpt_model")
        except:
            chkpt_model = False
    else:
        TRAIN = False

    if "test" in arguments:
        TEST = True
    else:
        TEST = False


    return TRAIN, TEST, chkpt_model


if __name__ == "__main__":
    TRAIN, TEST, chkpt_model = argparser()

    lr = 1e-3
    betas = (0.5, 0.9)   # for the Adam optimizer
    batch_size = 64
    num_training_steps = 5000
    print_freq = num_training_steps/100
    chkpt_freq = num_training_steps/5
    measure_model_path = "../../measure_checkpoints_tuning9/"
    op_model_path = "../../op_checkpoints_tuning9/"
    measure_pickle_path = "m_losses_tuning9.p"
    op_pickle_path = "op_losses_tuning9.p"
    generator_freq = 1 # Set to 1 if not Wasserstein

    args = {"lr": lr, "betas": betas, "batch_size": batch_size, "num_training_steps": num_training_steps,
            "print_freq": print_freq, "chkpt_freq": chkpt_freq, "chkpt_model": chkpt_model,
            "measure_model_path": measure_model_path, "op_model_path": op_model_path,
            "measure_pickle_path": measure_pickle_path, "op_pickle_path": op_pickle_path,
            "generator_freq": generator_freq}

    if TRAIN:
        trainer = Trainer(args)
        trainer.train()

    if TEST:
        tester = Tester(args)
        tester.test()
        tester.plot_training_losses()


