class CVAE_Params():
    def __init__(self):
        self.dim_hidden = 256
        self.enc_out_dim = 64
        self.latent_dim = 64
        self.batch_size = 64
        self.lr = 5e-5
        self.beta = 1
        self.leak = 0.2
        self.num_training_steps = 100 # 70000
        self.calibration = True
        self.save_model = False
        self.save_path = "data/pretrain_cvae"
        self.num_tests = 20

