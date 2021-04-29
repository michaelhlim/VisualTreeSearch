class CVAE_Params():
    def __init__(self):
        self.dim_hidden = 256
        self.enc_out_dim = 64
        self.latent_dim = 64
        self.batch_size = 64
        self.lr = 5e-5
        self.beta = 1
        self.leak = 0.2
        self.num_training_steps = 500000  # 70000
        self.calibration = True
        self.num_tests = 20

