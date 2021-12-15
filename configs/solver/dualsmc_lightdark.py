class DualSMC_LightDark_Params():
    def __init__(self):

        self.device = 'cuda:1'
        self.torch_seed = 1
        self.random_seed = 1
        self.np_random_seed = 1

        ######################
        # Network
        self.model_name = 'dualsmc_lightdark'
        self.dim_m = 64
        self.dim_first_layer = 64 #128 #64 
        self.dim_lstm_hidden = 64 
        self.num_lstm_layer = 2
        self.dim_encode = 64

        self.in_channels = 3
        self.obs_encode_out = 64 #256 #64
        self.mlp_hunits = 128
        #self.calibration = False
        self.normalization = False

        ## (Deep) Encoder, if used
        self.mlp_hunits_enc1 = 1024
        self.mlp_hunits_enc2 = 512
        self.mlp_hunits_enc3 = 256
        self.obs_encode_out_conv = 2048

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

