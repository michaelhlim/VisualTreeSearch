# author: @sdeglurkar, @jatucker4, @michaelhlim

import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from torch import nn
from torch.nn import functional as F

from configs.solver.image_cgan import *
from src.solvers.image_cgan import *

image_cgan_params = Image_CGAN_Params()


class GANTrain():

    def __init__(self):
        self.device = torch.device(image_cgan_params.device)

        self.state_enc_out_dim = image_cgan_params.state_enc_out_dim
        self.dim_state = image_cgan_params.dim_state
        self.in_channels = image_cgan_params.in_channels

        args = {'state_enc_out_dim': self.state_enc_out_dim, 'dim_state': self.dim_state, 
                'in_channels': self.in_channels, "device": self.device}

        self.op_model = ObsPredictorNetwork(args)
        self.m_model = MeasureNetwork(args)

        self.num_training_steps = image_cgan_params.num_training_steps
        self.data_path = image_cgan_params.data_path
        self.test_data_path = image_cgan_params.test_data_path
        self.batch_size = image_cgan_params.batch_size
        self.lr = image_cgan_params.lr
        self.normalization = image_cgan_params.normalization

        # self.m_optimizer = torch.optim.Adam(self.m_model.parameters(), lr=self.lr)
        # self.op_optimizer = torch.optim.Adam(self.op_model.parameters(), lr=self.lr)
        self.m_optimizer = torch.optim.SGD(self.m_model.parameters(), lr=self.lr)
        self.op_optimizer = torch.optim.SGD(self.op_model.parameters(), lr=self.lr)

        self.img_size = image_cgan_params.img_size
        self.data_files = glob.glob(self.data_path)
        self.testing_data_files = glob.glob(self.test_data_path)
    
        self.op_save_path = image_cgan_params.op_save_path
        self.m_save_path = image_cgan_params.m_save_path
        self.op_training_loss_path = image_cgan_params.op_training_loss_path
        self.m_training_loss_path = image_cgan_params.m_training_loss_path

        self.op_test_model_path = image_cgan_params.op_test_model_path
        self.m_test_model_path = image_cgan_params.m_test_model_path

        self.test_output_path = image_cgan_params.test_output_path
        self.test_true_path = image_cgan_params.test_true_path

        self.rmean, self.gmean, self.bmean, self.rstd, self.gstd, self.bstd = self.preprocess_data()
    

    def preprocess_data(self):
        # For normalizing the images - per channel mean and std

        rmean = 0
        gmean = 0
        bmean = 0
        rstd = 0
        gstd = 0
        bstd = 0
        for i in range(len(self.data_files)):
            img_path = self.data_files[i]
            src = cv2.imread(img_path, cv2.IMREAD_COLOR)
            src = src[:,:,::-1]   ## CV2 works in BGR space instead of RGB -- now src is in RGB
            
            rmean += src[:, :, 0].mean()/len(self.data_files)
            gmean += src[:, :, 1].mean()/len(self.data_files)
            bmean += src[:, :, 2].mean()/len(self.data_files)
            
            rstd += src[:, :, 0].std()/len(self.data_files)
            gstd += src[:, :, 1].std()/len(self.data_files)
            bstd += src[:, :, 2].std()/len(self.data_files)
        
        return rmean, gmean, bmean, rstd, gstd, bstd

    
    def get_training_batch(self, batch_size):
        states = []
        images = []
        precision = 4
        indices = np.random.randint(0, len(self.data_files), batch_size)
        for index in indices:
            img_path = self.data_files[index]
            src = cv2.imread(img_path, cv2.IMREAD_COLOR)
            
            if self.normalization:
                img_rslice = (src[:, :, 0] - self.rmean)/self.rstd
                img_gslice = (src[:, :, 1] - self.gmean)/self.gstd
                img_bslice = (src[:, :, 2] - self.bmean)/self.bstd

                img = np.stack([img_rslice, img_gslice, img_bslice], axis=-1)

                images.append(img)
            else:
                src = src[:,:,::-1]   ## CV2 works in BGR space instead of RGB -- converts to RGB
                src = (src - src.mean())/src.std()
                images.append(src)

            splits = img_path[:-precision].split('_')
            state = np.array([[np.round(float(elem), 3) for elem in splits[-3:]]])
            states.append(state)
        
        return np.array(states), np.array(images)


    def get_testing_batch(self, batch_size):
        states = []
        images = []
        precision = 4
        indices = np.random.randint(0, len(self.testing_data_files), batch_size)
        for index in indices:
            img_path = self.testing_data_files[index]
            print(img_path)
            src = cv2.imread(img_path, cv2.IMREAD_COLOR)
            
            if self.normalization:
                src = src[:,:,::-1]   ## CV2 works in BGR space instead of RGB -- converts to RGB
                img_rslice = (src[:, :, 0] - self.rmean)/self.rstd
                img_gslice = (src[:, :, 1] - self.gmean)/self.gstd
                img_bslice = (src[:, :, 2] - self.bmean)/self.bstd

                img = np.stack([img_rslice, img_gslice, img_bslice], axis=-1)

                images.append(img)
            else:
                src = src[:,:,::-1]   ## CV2 works in BGR space instead of RGB
                src = (src - src.mean())/src.std()
                images.append(src)
            
            splits = img_path[:-precision].split('_')
            state = np.array([[np.round(float(elem), 3) for elem in splits[-3:]]])
            states.append(state)
        
        return np.array(states), np.array(images) 


    def train(self):
        MSE_criterion = nn.MSELoss()
        BCE_criterion = nn.BCELoss()

        print_freq = 50
        update_freq = 10
        self.op_training_losses = []
        self.m_training_losses = []

        t0 = time.time()
        for step in range(self.num_training_steps):
            states, images = self.get_training_batch(self.batch_size)
            states = torch.from_numpy(states).float().to(self.device)
            states = states.squeeze(1)
            images = torch.from_numpy(images).float().to(self.device)
            images = images.permute(0, 3, 1, 2)  # [batch_size, in_channels, 32, 32]

            # ----------------------------
            #  Train Observation Predictor
            # ----------------------------
            noise = torch.randn_like(states)
            noisy_states = states #+ noise
            self.op_optimizer.zero_grad()
            obs_predicted = self.op_model(noisy_states)  # (batch_size, in_channels, 32,)
            OP_loss = 0

            fake_logit = self.m_model(noisy_states, obs_predicted)  # (batch_size, 1)
            real_target = torch.ones_like(fake_logit) * 0.9
            OP_loss += BCE_criterion(fake_logit, real_target)

            OP_loss.backward()
            self.op_optimizer.step()

            # ------------------------
            #  Train Discriminator
            # ------------------------
            if step % update_freq == 0:
                self.m_optimizer.zero_grad()

                noise = torch.randn_like(states)
                noisy_states = states + noise
                fake_logit = self.m_model(noisy_states, obs_predicted.detach())  # (batch_size, 1)
                fake_target = torch.zeros_like(fake_logit)  # (batch_size, 1)
                fake_loss = BCE_criterion(fake_logit, fake_target)

                real_logit = self.m_model(noisy_states, images)  # (batch_size, 1)
                real_target = torch.ones_like(real_logit)
                real_loss = BCE_criterion(real_logit, real_target)
                OM_loss = real_loss + fake_loss
                OM_loss.backward()
                self.m_optimizer.step()


            if step % print_freq == 0:
                print(step, OP_loss.item(), OM_loss.item())
                self.op_training_losses.append((step, OP_loss.item()))
                self.m_training_losses.append((step, OM_loss.item()))

        
        t1 = time.time()
        torch.save(self.op_model.state_dict(), self.op_save_path)
        torch.save(self.m_model.state_dict(), self.m_save_path)
        
        steps = [loss[0] for loss in self.op_training_losses]
        op_losses = [loss[1] for loss in self.op_training_losses]
        m_losses = [loss[1] for loss in self.m_training_losses]
        plt.figure()
        plt.plot(steps, op_losses)
        plt.savefig(self.op_training_loss_path) 
        plt.figure()
        plt.plot(steps, m_losses)
        plt.savefig(self.m_training_loss_path) 

        print("TRAINING TIME", t1 - t0)


    def test(self):
        self.op_model.load_state_dict(torch.load(self.op_test_model_path))
        num_tests = 1
        
        states, images = self.get_testing_batch(num_tests)
        states = torch.from_numpy(states).float().to(self.device)
        images = torch.from_numpy(images).float().to(self.device)

        states = states.detach()
        images = images.detach()

        output = self.op_model(states[0])  # [1, in_channels, 32, 32]
        output = output.permute(0, 2, 3, 1).squeeze(0)  # [32, 32, in_channels]
        output = output.detach().cpu().numpy()      
        if self.normalization:
            output[:, :, 0] = (output[:, :, 0] * self.rstd + self.rmean)
            output[:, :, 1] = (output[:, :, 1] * self.gstd + self.gmean)
            output[:, :, 2] = (output[:, :, 2] * self.bstd + self.bmean)
        else:
            output = (output + (255./2)/100) * 100.  # "Denormalization"

        original = images[0]  # [32, 32, in_channels]
        original = original.detach().cpu().numpy()
        if self.normalization:
            original[:, :, 0] = (original[:, :, 0] * self.rstd + self.rmean)
            original[:, :, 1] = (original[:, :, 1] * self.gstd + self.gmean)
            original[:, :, 2] = (original[:, :, 2] * self.bstd + self.bmean)
        else:
            original = (original + (255./2)/100) * 100.  # "Denormalization"

        print("STATE", states[0])
        state = states[0]

        output = output[:,:,::-1]   ## CV2 works in BGR space instead of RGB -- converts to BGR
        original = original[:,:,::-1]   ## CV2 works in BGR space instead of RGB
        state_str = ".png"
        cv2.imwrite(self.test_output_path + state_str, output)
        cv2.imwrite(self.test_true_path + state_str, original)


gan_train = GANTrain()
if image_cgan_params.train:
    gan_train.train()
else:
    gan_train.test()

