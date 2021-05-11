class Image_CVAE_Params():
    def __init__(self):
        self.in_channels = 3
        self.latent_dim = 64
        self.dim_conditional_var = 3
        self.img_size = 32 
        self.batch_size = 64
        self.num_training_data = 7000
        self.num_epochs = 100
        self.num_training_steps = int((self.num_training_data/self.batch_size) * self.num_epochs)  
        self.data_path = "/home/ext_drive/sampada_deglurkar/vae_stanford/training_hard/rgbs/*"
        self.save_path = "image_cvae_hard.pth"
        self.training_loss_path = "image_cvae_hard"
        self.test_data_path = "/home/ext_drive/sampada_deglurkar/vae_stanford/testing_hard/rgbs/*"
        self.test_model_path = "/home/sampada_deglurkar/VisualTreeSearch/src/solvers/image_cvae_hard.pth"
        self.test_output_path = "test_output_hard"
        self.test_true_path = "test_true_hard"
        self.lr = 1e-3 
        self.beta = 1
        self.device = 'cuda:1'
        self.calibration = True
        self.normalization = False

        self.train = False


