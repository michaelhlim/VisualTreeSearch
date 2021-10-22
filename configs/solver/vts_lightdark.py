class VTS_LightDark_Params():
    def __init__(self):

        self.device = 'cuda:1'

        ######################
        self.model_name = 'vts_lightdark'

        self.in_channels = 3

        ## Z and P
        self.dim_m = 256
        self.dim_first_layer = 256 #64 
        self.dim_lstm_hidden = 256 #64 
        self.num_lstm_layer = 2
        self.obs_encode_out = 256 #2048 #64
        self.mlp_hunits_zp = 128
        self.zp_lr = 3e-4
        self.num_epochs_zp = 0 #400 

        ## G
        self.num_layers = 5
        self.latent_dim = 128 #32
        self.mlp_hunits_g = 256 #128
        self.dim_conditional_var = 2
        self.leak_rate_g = 0
        self.calibration = True
        self.g_lr = 3e-4 #1e-3 
        self.beta = 1
        self.num_epochs_g = 400 

        ## Encoder
        self.leak_rate_enc = 0
        self.mlp_hunits_enc1 = 1024
        self.mlp_hunits_enc2 = 512
        self.mlp_hunits_enc3 = 256
        self.obs_encode_out_conv = 2048

        ######################
        # Training
        self.num_training_data = 16000 #8000
        #self.train = True
        #self.num_pretraining_steps = 50
        self.max_episodes_train = 2000 #5000
        self.max_episodes_test = 20 #10 #1 #500 #1000
        self.batch_size = 64 #128 #64
        self.fil_lr = 3e-4 #1e-3 # filtering
        self.pla_lr = 3e-4 #1e-3 # planning
        self.summary_iter = 100
        self.save_iter = 1 #40 #100
        self.display_iter = 1 #4 #10
        #self.pretrain = 500e3
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
        # Planning
        # self.num_par_smc_init = 3
        # self.num_par_smc = 10 #30
        # self.horizon = 10
        # self.smcp_mode = 'topk' # 'samp', 'topk'
        # self.smcp_resample = True
        # self.smcp_resample_step = 1 #3

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
        # self.log_sig_max = 2
        # self.log_sig_min = -20
        self.replay_buffer_size = 100000
        # self.alpha = 1.0
        # self.gamma = 0.95
        # self.tau = 0.005
        # self.const = 1e-6
        # self.critic_update = 1 #50

