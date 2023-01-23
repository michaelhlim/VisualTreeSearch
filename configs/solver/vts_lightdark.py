from configs.environments.stanford import *
sep = Stanford_Environment_Params()


class VTS_LightDark_Params():
    def __init__(self):

        self.device = 'cuda:1'
        self.torch_seed = 1
        self.random_seed = 1
        self.np_random_seed = 1

        ######################
        self.model_name = 'vts_lightdark'
        
        ## Encoder --- this is represented by the file observation_generator_conv_lightdark
        self.in_channels = 3
        self.leak_rate_enc = 0
        # Channel dimensions for generator_conv_lightdark
        self.hidden_dims_generator_conv = [32, 64, 128, 256, 512]  
        # Each convolution in generator_conv_lightdark downsamples by a factor of 2
        # So the final "image size" after the convolutions is img_size * (1/2)^(# layers)
        # And the output shape is [batch_size, final # of channels, final image size, final image size]
        # obs_encode_out_conv is the size that comes from doing a torch.flatten on the above shape
        self.final_num_channels = self.hidden_dims_generator_conv[-1]
        self.final_img_size = int((0.5)**(len(self.hidden_dims_generator_conv)) * sep.img_size)
        if self.final_img_size < 2:
            self.final_img_size = 2 # The last convolution will not downsample in this case
        self.obs_encode_out_conv = self.final_num_channels * self.final_img_size**2 
        self.mlp_hunits_enc = [1024, 512, 256] 
        # This should be the same as self.mlp_hunits_enc[-1]
        self.obs_encode_out = self.mlp_hunits_enc[-1] 

        # If the encoder is trained independently, this is nonzero
        self.num_epochs_e = 0 

        ## Z and P
        self.dim_m = 256 
        self.dim_first_layer = 256  
        self.dim_lstm_hidden = 256 
        self.num_lstm_layer = 2      
        self.mlp_hunits_zp = 128 
        self.zp_lr = 3e-4
        self.num_epochs_zp = 400 

        ## G
        self.diff_pattern = True  # Training: Pre-generate the corrupted indices per noisy image
        self.num_layers = 5
        self.latent_dim = 128 
        self.mlp_hunits_g = 256 
        self.dim_conditional_var = 2
        self.leak_rate_g = 0
        self.calibration = True
        self.g_lr = 3e-4 
        self.beta = 1
        self.num_epochs_g = 400          

        ######################
        # Training
        self.num_training_data = 16000 
        self.max_episodes_train = 2000 
        self.max_episodes_test = 500 
        self.batch_size = 64 
        self.summary_iter = 100
        self.save_iter = 1 
        self.display_iter = 1 
        self.show_traj = True
        self.show_distr = True

        ######################
        # Filtering
        self.pf_resample_step = 3 
        self.num_par_pf = 100
        self.pp_exist = True
        self.pp_ratio = 0.3
        self.pp_loss_type = 'adv'  # 'mse', 'adv', 'density'
        self.pp_decay = True
        self.decay_rate = 0.9
        self.pp_std = False 
        self.std_thres = 0.07
        self.std_alpha = 100
        self.pp_effective = False
        self.effective_thres = int(2/3 * self.num_par_pf)
        self.effective_alpha = 1000

        # ######################
        # PFTDPW    
        self.num_query = 100  
        self.ucb_exploration = 10.0  
        self.k_observation = 4.0  
        self.alpha_observation = 0.25   
        self.k_action = 3.0   
        self.alpha_action = 0.25     
        self.num_par_pftdpw = 100
        self.horizon = 10
        self.discount = 0.99    

        # ######################
        # SAC
        self.replay_buffer_size = 100000  # Not really used because we don't do online training

