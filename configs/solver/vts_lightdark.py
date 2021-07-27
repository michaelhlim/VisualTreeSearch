class VTS_LightDark_Params():
    def __init__(self):

        self.device = 'cuda:2'

        ######################
        self.model_name = 'vts_lightdark'
        ## Z and P
        self.dim_first_layer = 64 
        #self.dim_encode = 64
        self.in_channels = 3
        self.obs_encode_out = 64
        self.mlp_hunits_zp = 128
        ## G
        self.num_layers = 3
        self.latent_dim = 32
        self.mlp_hunits_g = 128
        self.dim_conditional_var = 2
        self.leak_rate = 0
        #self.img_size = 32 
        #self.normalization = False
        self.calibration = False
        #self.lr = 1e-3 
        #self.beta = 1
        #self.batch_size = 64 #32 
        #self.num_training_data = 8000 #7000 #500 
        #self.num_epochs = 2 #200 #100 #200 

        ######################
        # Training
        self.train = True
        self.num_pretraining_steps = 50
        self.max_episodes_train = 2000 #5000
        self.max_episodes_test = 500 #1000
        self.batch_size = 128 #64
        self.fil_lr = 3e-4 #1e-3 # filtering
        self.pla_lr = 3e-4 #1e-3 # planning
        self.summary_iter = 100
        self.save_iter = 40 #100
        self.display_iter = 4 #10
        self.pretrain = 500e3
        self.show_traj = True
        self.show_distr = False

        ######################
        # Filtering
        self.pf_resample_step = 3 
        self.num_par_pf = 100
        self.pp_exist = True
        self.pp_ratio = 0.3
        self.pp_loss_type = 'adv'  # 'mse', 'adv', 'density'

        # ######################
        # Planning
        self.num_par_smc_init = 3
        self.num_par_smc = 10 #30
        self.horizon = 10
        self.smcp_mode = 'topk' # 'samp', 'topk'
        self.smcp_resample = True
        self.smcp_resample_step = 1 #3

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
        self.log_sig_max = 2
        self.log_sig_min = -20
        self.replay_buffer_size = 100000
        self.alpha = 1.0
        self.gamma = 0.95
        self.tau = 0.005
        self.const = 1e-6
        self.critic_update = 1 #50

