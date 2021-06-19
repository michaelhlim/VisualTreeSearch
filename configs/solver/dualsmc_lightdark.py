class DualSMC_LightDark_Params():
    def __init__(self):

        self.device = 'cuda:1'

        ######################
        # Network
        self.model_name = 'dualsmc'
        self.dim_first_layer = 64 
        self.dim_lstm_hidden = 64 
        self.num_lstm_layer = 2
        self.dim_encode = 64

        self.in_channels = 3
        self.obs_encode_out = 64
        self.mlp_hunits = 128
        self.calibration = False
        self.normalization = False

        ######################
        # Training
        self.train = True
        self.num_pretraining_steps = 50
        self.max_episodes_train = 5000 #10 #10000
        self.max_episodes_test = 500 #5 #1000
        self.batch_size = 64
        self.fil_lr = 1e-3 # filtering
        self.pla_lr = 1e-3 # planning
        self.summary_iter = 100
        self.save_iter = 100
        self.display_iter = 4 #1 #10
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
        self.num_par_smc = 30
        self.horizon = 10
        self.smcp_mode = 'topk' # 'samp', 'topk'
        self.smcp_resample = True
        self.smcp_resample_step = 3

        # ######################
        # SAC
        self.log_sig_max = 2
        self.log_sig_min = -20
        self.replay_buffer_size = 100000
        self.alpha = 1.0
        self.gamma = 0.95
        self.tau = 0.005
        self.const = 1e-6

