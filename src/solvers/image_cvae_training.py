import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from torch import nn
from torch.nn import functional as F

from configs.solver.image_cvae import *
from src.solvers.image_cvae import *

image_cvae_params = Image_CVAE_Params()

torch.backends.cudnn.enabled = False


class VAETrain:

    def __init__(self):
        self.device = torch.device(image_cvae_params.device)

        self.in_channels = image_cvae_params.in_channels
        self.dim_conditional_var = image_cvae_params.dim_conditional_var
        self.latent_dim = image_cvae_params.latent_dim
        self.img_size = image_cvae_params.img_size
        self.calibration = image_cvae_params.calibration
        self.normalization = image_cvae_params.normalization

        args = {'in_channels': self.in_channels, 'dim_conditional_var': self.dim_conditional_var,
                'latent_dim': self.latent_dim, 'img_size': self.img_size, 'calibration': self.calibration, 
                'device': self.device}
        self.model = ConditionalVAE(args)

        self.num_training_steps = image_cvae_params.num_training_steps
        self.data_path = image_cvae_params.data_path
        self.test_data_path = image_cvae_params.test_data_path
        self.batch_size = image_cvae_params.batch_size
        self.beta = image_cvae_params.beta
        self.lr = image_cvae_params.lr

        self.data_files = glob.glob(self.data_path)
        self.testing_data_files = glob.glob(self.test_data_path)
    
        self.save_path = image_cvae_params.save_path
        self.training_loss_path = image_cvae_params.training_loss_path
        self.test_model_path = image_cvae_params.test_model_path
        self.test_output_path = image_cvae_params.test_output_path
        self.test_true_path = image_cvae_params.test_true_path

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
            src = src[:,:,::-1]   ## CV2 works in BGR space instead of RGB!! So dumb! --- now src is in RGB
            
            rmean += src[:, :, 0].mean()/len(self.data_files)
            gmean += src[:, :, 1].mean()/len(self.data_files)
            bmean += src[:, :, 2].mean()/len(self.data_files)
            
            rstd += src[:, :, 0].std()/len(self.data_files)
            gstd += src[:, :, 1].std()/len(self.data_files)
            bstd += src[:, :, 2].std()/len(self.data_files)
        
        return rmean, gmean, bmean, rstd, gstd, bstd

        #     src = src/len(self.data_files)
        #     if i == 0:
        #         mean_image = src
        #     else:
        #         mean_image = src + mean_image
        
        # return mean_image


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
                src = src[:,:,::-1]   ## CV2 works in BGR space instead of RGB!! So dumb! -- converts to RGB
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
                src = src[:,:,::-1]   ## CV2 works in BGR space instead of RGB!! So dumb! -- converts to RGB
                img_rslice = (src[:, :, 0] - self.rmean)/self.rstd
                img_gslice = (src[:, :, 1] - self.gmean)/self.gstd
                img_bslice = (src[:, :, 2] - self.bmean)/self.bstd

                img = np.stack([img_rslice, img_gslice, img_bslice], axis=-1)

                images.append(img)
            else:
                src = src[:,:,::-1]   ## CV2 works in BGR space instead of RGB!! So dumb!
                src = (src - src.mean())/src.std()
                images.append(src)
            
            splits = img_path[:-precision].split('_')
            state = np.array([[np.round(float(elem), 3) for elem in splits[-3:]]])
            states.append(state)
        
        return np.array(states), np.array(images)    

    
    def train(self):
        optimizer = self.model.configure_optimizers(self.lr)
        print_freq = 50
        self.training_losses = []

        t0 = time.time()
        for step in range(self.num_training_steps):
            states, images = self.get_training_batch(self.batch_size)
            states = torch.from_numpy(states).float().to(self.device)
            images = torch.from_numpy(images).float().to(self.device)
            images = images.permute(0, 3, 1, 2)  # [batch_size, in_channels, 32, 32]

            optimizer.zero_grad()
            [recons, input, mu, log_var] = self.model.forward(images, states)
            args = [recons, input, mu, log_var]
            loss_dict = self.model.loss_function(self.beta, *args)
            loss = loss_dict['loss']
            recon_loss = loss_dict['Reconstruction_Loss']
            kl_loss = loss_dict['KLD']
            loss.backward()
            optimizer.step()

            if step % print_freq == 0:
                print(step, loss.item(), kl_loss.item(), recon_loss.item())
                self.training_losses.append((step, loss.item()))
        
        t1 = time.time()
        torch.save(self.model.state_dict(), self.save_path)

        plt.figure()
        steps = [loss[0] for loss in self.training_losses]
        losses = [loss[1] for loss in self.training_losses]
        plt.plot(steps, losses)
        plt.savefig(self.training_loss_path) 

        print("TRAINING TIME", t1 - t0)
    

    def test(self):
        self.model.load_state_dict(torch.load(self.test_model_path))
        num_tests = 1
        
        states, images = self.get_testing_batch(num_tests)
        states = torch.from_numpy(states).float().to(self.device)
        images = torch.from_numpy(images).float().to(self.device)

        states = states.detach()
        images = images.detach()

        output = self.model.sample(1, states[0])  # [1, in_channels, 32, 32]
        output = output.permute(0, 2, 3, 1).squeeze(0)  # [32, 32, in_channels]
        output = output.detach().cpu().numpy()        
        #output = (output + 68./100) * 100.
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

        output = output[:,:,::-1]   ## CV2 works in BGR space instead of RGB!! So dumb! -- converts to BGR
        original = original[:,:,::-1]   ## CV2 works in BGR space instead of RGB!! So dumb!
        #state_str = '_' + str(state[0][0]) + '_' + str(state[0][1]) + '_' + str(state[0][2]) + ".png"
        state_str = ".png"
        cv2.imwrite(self.test_output_path + state_str, output)
        cv2.imwrite(self.test_true_path + state_str, original)


vae_train = VAETrain()
if image_cvae_params.train:
    vae_train.train()
else:
    vae_train.test()

        
