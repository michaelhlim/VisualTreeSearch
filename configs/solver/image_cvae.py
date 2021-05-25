class Image_CVAE_Params():
    def __init__(self):
        self.in_channels = 3
        self.latent_dim = 64
        self.mlp_hunits = 128
        self.dim_conditional_var = 3
        self.img_size = 32 
        self.normalization = False
        self.device = 'cuda:1'
        
        # Training
        self.calibration = False
        self.lr = 1e-3 
        self.beta = 1
        self.batch_size = 64 #32 
        self.num_training_data = 8000 #7000 #500 
        self.num_epochs = 200 #100 #200 
        #self.num_training_steps = int((self.num_training_data/self.batch_size) * self.num_epochs)  
        self.data_path = "/home/ext_drive/sampada_deglurkar/vae_stanford/training_hallway/rgbs/*"
        self.save_path = "image_cvae_hallway.pth"
        self.training_loss_path = "image_cvae_hallway"
        
        # Testing
        self.test_data_path = "/home/ext_drive/sampada_deglurkar/vae_stanford/testing_hallway/rgbs/*"
        self.test_model_path = "/home/sampada_deglurkar/VisualTreeSearch/src/solvers/image_cvae_hallway.pth"
        self.test_output_path = "test_output_hallway"
        self.test_true_path = "test_true_hallway"
        

        self.train = False


