class Image_CVAE_Params():
    def __init__(self):
        self.in_channels = 3
        self.latent_dim = 64
        self.dim_conditional_var = 3
        self.img_size = 32 
        self.num_training_steps = int(2e3) #int(50e3)
        self.data_path = "/home/ext_drive/sampada_deglurkar/vae_stanford/training/rgbs/*"
        self.save_path = "image_cvae_calib_normlr1e-3.pth"
        self.training_loss_path = "image_cvae_training_losses_calib_normlr1e-3"
        self.test_data_path = "/home/ext_drive/sampada_deglurkar/vae_stanford/testing/rgbs/*"
        self.test_model_path = "/home/sampada_deglurkar/VisualTreeSearch/src/solvers/image_cvae_calib_normlr1e-3.pth"
        self.test_output_path = "output_test_calib_normlr1e-3.png"
        self.batch_size = 64
        self.lr = 1e-3 #5e-5
        self.beta = 1
        self.device = 'cuda:1'
        self.calibration = True

        # Epoch size: 2000/batch_size = 31.25 steps
        # 1500 steps is about 50 epochs

