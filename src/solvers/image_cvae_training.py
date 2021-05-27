import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import time
import torch
from torch import nn, reshape
from torch.nn import functional as F

from configs.solver.image_cvae import *
from src.solvers.image_cvae import *

image_cvae_params = Image_CVAE_Params()

torch.backends.cudnn.enabled = False


class SaveFeatures():
    features=None
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = ((output.cpu()).data).numpy()
    def remove(self): self.hook.remove()


class VAETrain:

    def __init__(self):
        self.device = torch.device(image_cvae_params.device)

        self.in_channels = image_cvae_params.in_channels
        self.dim_conditional_var = image_cvae_params.dim_conditional_var
        self.latent_dim = image_cvae_params.latent_dim
        self.mlp_hunits = image_cvae_params.mlp_hunits
        self.img_size = image_cvae_params.img_size
        self.calibration = image_cvae_params.calibration
        self.normalization = image_cvae_params.normalization

        args = {'in_channels': self.in_channels, 'dim_conditional_var': self.dim_conditional_var,
                'latent_dim': self.latent_dim, 'mlp_hunits': self.mlp_hunits, 'img_size': self.img_size,
                'calibration': self.calibration, 'device': self.device}
        self.model = ConditionalVAE(args)

        #self.num_training_steps = image_cvae_params.num_training_steps
        self.num_epochs = image_cvae_params.num_epochs
        self.num_training_data = image_cvae_params.num_training_data
        self.batch_size = image_cvae_params.batch_size
        self.steps_per_epoch = int(np.ceil(self.num_training_data/self.batch_size))
        self.data_path = image_cvae_params.data_path
        self.test_data_path = image_cvae_params.test_data_path
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
        remove = 4
        indices = np.random.choice(range(len(self.data_files)), batch_size, replace=False)
        #indices = np.random.randint(0, len(self.data_files), batch_size)
        for index in indices:
            img_path = self.data_files[index]
            src = cv2.imread(img_path, cv2.IMREAD_COLOR)
            
            if self.normalization:
                ####################### I THINK THERE IS A BUG HERE -- SHOULD BE CHANGING TO RGB ##################
                img_rslice = (src[:, :, 0] - self.rmean)/self.rstd
                img_gslice = (src[:, :, 1] - self.gmean)/self.gstd
                img_bslice = (src[:, :, 2] - self.bmean)/self.bstd

                img = np.stack([img_rslice, img_gslice, img_bslice], axis=-1)

                images.append(img)
            else:
                src = src[:,:,::-1]   ## CV2 works in BGR space instead of RGB!! So dumb! -- converts to RGB
                src = (src - src.mean())/src.std()
                images.append(src)

            splits = img_path[:-remove].split('_')
            state = np.array([[np.round(float(elem), 3) for elem in splits[-3:]]])
            states.append(state)
        
        return np.array(states), np.array(images)
    

    def get_training_batch(self, batch_size, data_files_indices, epoch_step):
        states = []
        images = []
        remove = 4

        if (epoch_step + 1)*batch_size > len(data_files_indices):
            indices = data_files_indices[epoch_step*batch_size:]
        else:
            indices = data_files_indices[epoch_step*batch_size:(epoch_step + 1)*batch_size]

        for index in indices:
            img_path = self.data_files[index]
            src = cv2.imread(img_path, cv2.IMREAD_COLOR)
            
            if self.normalization:
                ####################### I THINK THERE IS A BUG HERE -- SHOULD BE CHANGING TO RGB ##################
                img_rslice = (src[:, :, 0] - self.rmean)/self.rstd
                img_gslice = (src[:, :, 1] - self.gmean)/self.gstd
                img_bslice = (src[:, :, 2] - self.bmean)/self.bstd

                img = np.stack([img_rslice, img_gslice, img_bslice], axis=-1)

                images.append(img)
            else:
                src = src[:,:,::-1]   ## CV2 works in BGR space instead of RGB!! So dumb! -- converts to RGB
                src = (src - src.mean())/src.std()
                images.append(src)

            splits = img_path[:-remove].split('_')
            state = np.array([[np.round(float(elem), 3) for elem in splits[-3:]]])
            states.append(state)
        
        return np.array(states), np.array(images)


    def get_testing_batch(self, batch_size):
        states = []
        images = []
        remove = 4
        #indices = list(range(len(self.testing_data_files)))
        indices = np.random.choice(range(len(self.testing_data_files)), batch_size, replace=False)
        #indices = np.random.randint(0, len(self.testing_data_files), batch_size)
        for index in indices:
            img_path = self.testing_data_files[index]
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
            
            splits = img_path[:-remove].split('_')
            state = np.array([[np.round(float(elem), 3) for elem in splits[-3:]]])
            states.append(state)
        
        return np.array(states), np.array(images)    

    
    def train(self):
        optimizer = self.model.configure_optimizers(self.lr)
        print_freq = 50
        self.training_losses = []

        t0 = time.time()
        #for step in range(self.num_training_steps):
        for epoch in range(self.num_epochs):
            print("Epoch:", epoch)
            data_files_indices = list(range(len(self.data_files)))
            np.random.shuffle(data_files_indices)

            for step in range(self.steps_per_epoch):
                states, images = self.get_training_batch(self.batch_size, data_files_indices, step)
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
                    true_step = epoch * self.steps_per_epoch + step
                    print("Step:", step, true_step, "Loss:", loss.item(), "KL Loss:", kl_loss.item(), 
                            "Recons Loss:", recon_loss.item())
                    self.training_losses.append((true_step, loss.item()))
        
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

        output = self.model.sample(num_tests, states[0])  # [1, in_channels, 32, 32]
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
    

    def test_interpolate(self):
        self.model.load_state_dict(torch.load(self.test_model_path))
        num_tests = 1
        
        states, images = self.get_testing_batch(num_tests)
        states = torch.from_numpy(states).float().to(self.device)
        images = torch.from_numpy(images).float().to(self.device)

        states = states.detach()
        images = images.detach()

        output_arr, z_start, z_direction = self.model.sample_interpolate(10, states[0])  # [1, in_channels, 32, 32]
        for i in range(len(output_arr)):
            output = output_arr[i]
        
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
            state_str = str(i) + ".png"
            cv2.imwrite(self.test_output_path + state_str, output)
            cv2.imwrite(self.test_true_path + state_str, original)
    

    def test_tsne(self, analysis_dataset, hallway_dataset, module, reshape):
        self.model.load_state_dict(torch.load(self.test_model_path))

        layer = self.model._modules.get(module)  
        activated_features = SaveFeatures(layer)

        if analysis_dataset:
            num_tests = 500
        elif hallway_dataset:
            num_tests = 200
        else:  # This is the hard dataset
            num_tests = 150

        states, images = self.get_testing_batch(num_tests)
        states = torch.from_numpy(states).float().to(self.device)
        images = torch.from_numpy(images).float().to(self.device)

        states = states.detach()
        images = images.detach()

        out = self.model(images.permute(0, 3, 1, 2), states)
        output = self.model.sample(num_tests, states.squeeze(1))  # [num_tests, in_channels, 32, 32]  

        states = states.squeeze(1).cpu().numpy()
        # (unique, counts) = np.unique(states, axis=0, return_counts=True)
        # frequencies = np.asarray((unique, counts)).T
        # print(frequencies)

        e = 0.0
        ne = np.pi/4
        n = np.pi/2
        nw = 3*np.pi/4 
        w = np.pi
        sw = 5*np.pi/4 
        s = 3*np.pi/2 
        se = 7*np.pi/4
        eps = 1e-3

        category1_theta = np.argwhere((np.abs(states[:, 2] - e) <= eps) | 
                                        (np.abs(states[:, 2] - ne) <= eps) | 
                                        (np.abs(states[:, 2] - n) <= eps)).flatten()
        category2_theta = np.argwhere((np.abs(states[:, 2] - nw) <= eps) | 
                                        (np.abs(states[:, 2] - w) <= eps) | 
                                        (np.abs(states[:, 2] - sw) <= eps)).flatten()
        category3_theta = np.argwhere((np.abs(states[:, 2] - s) <= eps) | 
                                        (np.abs(states[:, 2] - se) <= eps)).flatten()

        category1_y = np.argwhere(states[:, 1] <= 21).flatten()
        category2_y = np.argwhere((states[:, 1] > 21) & (states[:, 1] <= 23)).flatten()
        category3_y = np.argwhere((states[:, 1] > 23) & (states[:, 1] <= 25)).flatten()

        category1_x = np.argwhere(states[:, 0] <= 26).flatten()
        category2_x = np.argwhere((states[:, 0] > 26) & (states[:, 1] <= 29)).flatten()
        category3_x = np.argwhere((states[:, 0] > 29) & (states[:, 1] <= 32.5)).flatten()

        plotting_dictionary = {}

        if analysis_dataset:
            plotting_dictionary["20 < y < 21"] = category1_y
            plotting_dictionary["21 < y < 23"] = category2_y
            plotting_dictionary["23 < y < 25"] = category3_y
        elif hallway_dataset:
            plotting_dictionary["north/east, 24 < x < 26"] = np.intersect1d(category1_theta, category1_x)
            plotting_dictionary["north/east, 26 < x < 29"] = np.intersect1d(category1_theta, category2_x)
            plotting_dictionary["north/east, 29 < x < 32.5"] = np.intersect1d(category1_theta, category3_x)
            plotting_dictionary["west, 24 < x < 26"] = np.intersect1d(category2_theta, category1_x)
            plotting_dictionary["west, 26 < x < 29"] = np.intersect1d(category2_theta, category2_x)
            plotting_dictionary["west, 29 < x < 32.5"] = np.intersect1d(category2_theta, category3_x)
            plotting_dictionary["south/east, 24 < x < 26"] = np.intersect1d(category3_theta, category1_x)
            plotting_dictionary["south/east, 26 < x < 29"] = np.intersect1d(category3_theta, category2_x)
            plotting_dictionary["south/east, 29 < x < 32.5"] = np.intersect1d(category3_theta, category3_x)
        else:
            plotting_dictionary["north/east, 20 < y < 21"] = np.intersect1d(category1_theta, category1_x)
            plotting_dictionary["north/east, 21 < y < 23"] = np.intersect1d(category1_theta, category2_x)
            plotting_dictionary["north/east, 23 < y < 25"] = np.intersect1d(category1_theta, category3_x)
            plotting_dictionary["west, 20 < y < 21"] = np.intersect1d(category2_theta, category1_x)
            plotting_dictionary["west, 21 < y < 23"] = np.intersect1d(category2_theta, category2_x)
            plotting_dictionary["west, 23 < y < 25"] = np.intersect1d(category2_theta, category3_x)
            plotting_dictionary["south/east, 20 < y < 21"] = np.intersect1d(category3_theta, category1_x)
            plotting_dictionary["south/east, 21 < y < 23"] = np.intersect1d(category3_theta, category2_x)
            plotting_dictionary["south/east, 23 < y < 25"] = np.intersect1d(category3_theta, category3_x)


        tsne_features = activated_features.features.reshape(-1, reshape)
        feature_embeddings = TSNE(n_components=2).fit_transform(tsne_features)
        activated_features.remove()

        for (key, value) in plotting_dictionary.items():
            if len(value) > 0:
                plt.scatter(feature_embeddings[value, 0], feature_embeddings[value, 1], label=key)

        plt.legend(bbox_to_anchor=(0.85, 1.05), loc='upper left', fontsize='xx-small')
        plt.savefig("TSNE")



vae_train = VAETrain()
if image_cvae_params.train:
    vae_train.train()
else:
    #vae_train.test()
    analysis_dataset = False
    hallway_dataset = True
    # vae_train.test_tsne(analysis_dataset=analysis_dataset, hallway_dataset=hallway_dataset, 
    #                     module="final_layer", reshape=vae_train.img_size**2 * vae_train.in_channels)
    # vae_train.test_tsne(analysis_dataset=analysis_dataset, hallway_dataset=hallway_dataset, 
    #                     module="encoder", reshape=512*4)
    # vae_train.test_tsne(analysis_dataset=analysis_dataset, hallway_dataset=hallway_dataset,
    #                     module="decoder_mlp", reshape=512*4)
    # vae_train.test_tsne(analysis_dataset=analysis_dataset, hallway_dataset=hallway_dataset,
    #                     module="encoder_mlp", reshape=128)
    vae_train.test_tsne(analysis_dataset=analysis_dataset, hallway_dataset=hallway_dataset,
                        module="fc_mu", reshape=64)
    # vae_train.test_tsne(analysis_dataset=analysis_dataset, hallway_dataset=hallway_dataset,
    #                     module="decoder", reshape=32*vae_train.img_size**2)

        
