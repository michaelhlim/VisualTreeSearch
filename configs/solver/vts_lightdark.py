from configs.environments.stanford import *
sep = Stanford_Environment_Params()


class VTS_LightDark_Params():
    def __init__(self):

        self.device = 'cuda:1'
        self.torch_seed = 4
        self.random_seed = 4
        self.np_random_seed = 4

        ######################
        self.model_name = 'vts_lightdark'
        
        ## Encoder --- this is represented by the file observation_generator_conv_lightdark
        self.in_channels = 3
        self.leak_rate_enc = 0
        # Channel dimensions for generator_conv_lightdark
        self.hidden_dims_generator_conv = [32, 64, 128, 256, 512]  #[16, 32, 64]  # [32, 64, 128, 256, 512]
        # Each convolution in generator_conv_lightdark downsamples by a factor of 2
        # So the final "image size" after the convolutions is img_size * (1/2)^(# layers)
        # And the output shape is [batch_size, final # of channels, final image size, final image size]
        # obs_encode_out_conv is the size that comes from doing a torch.flatten on the above shape
        self.final_num_channels = self.hidden_dims_generator_conv[-1]
        self.final_img_size = int((0.5)**(len(self.hidden_dims_generator_conv)) * sep.img_size)
        if self.final_img_size < 2:
            self.final_img_size = 2 # The last convolution will not downsample in this case
        self.obs_encode_out_conv = self.final_num_channels * self.final_img_size**2 
        self.mlp_hunits_enc = [1024, 512, 256] #[512, 256, 128, 64, 32, 16] #[1024, 512, 256]
        # This should be the same as self.mlp_hunits_enc[-1]
        self.obs_encode_out = self.mlp_hunits_enc[-1] #16 #256 

        # If the encoder is trained independently, this is nonzero
        self.num_epochs_e = 0 

        ## Z and P
        self.dim_m = 256 #self.obs_encode_out #256
        self.dim_first_layer = 256 #self.obs_encode_out #256 #64 
        self.dim_lstm_hidden = 256 #self.obs_encode_out #256 #64 
        self.num_lstm_layer = 2      
        self.mlp_hunits_zp = 128 #self.obs_encode_out #128
        self.zp_lr = 3e-4
        self.num_epochs_zp = 400 #0 #400 

        ## G
        self.diff_pattern = True  # Training: Pre-generate the corrupted indices per noisy image
        self.num_layers = 5
        self.latent_dim = 128 #32
        self.mlp_hunits_g = 256 #128
        self.dim_conditional_var = 2
        self.leak_rate_g = 0
        self.calibration = True
        self.g_lr = 3e-4 #1e-3 
        self.beta = 1
        self.num_epochs_g = 400 #0 #400         

        ######################
        # Training
        self.num_training_data = 16000 #8000
        self.max_episodes_train = 2000 #5000
        self.max_episodes_test = 500 #20 #10 #1 #500 #1000
        self.batch_size = 64 #128 #64
        # self.fil_lr = 3e-4 #1e-3 # filtering
        # self.pla_lr = 3e-4 #1e-3 # planning
        self.summary_iter = 100
        self.save_iter = 1 #40 #100
        self.display_iter = 1 #4 #10
        self.show_traj = True
        self.show_distr = True

        ######################
        # Filtering
        self.pf_resample_step = 3 
        self.num_par_pf = 100
        self.pp_exist = True
        self.pp_ratio = 0.3
        self.pp_loss_type = 'adv'  # 'mse', 'adv', 'density'

        # ######################
        # PFTDPW    
        self.num_query = 100  # Change 1000
        self.ucb_exploration = 10.0  # Needs to be changed by order of 10's    2*max reward
        self.k_observation = 4.0  # Change 3
        self.alpha_observation = 0.25   # Check POMCPOW paper
        self.k_action = 3.0   # Change 5    # Don't do any widening? 
        self.alpha_action = 0.25     # Check POMCPOW paper
        self.num_par_pftdpw = 100
        self.horizon = 10
        self.discount = 0.99    # Change ---   0.9 or 0.95  

        # ######################
        # SAC
        self.replay_buffer_size = 100000  # Not really used because we don't do online training

