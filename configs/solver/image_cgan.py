# author: @sdeglurkar, @jatucker4, @michaelhlim

class Image_CGAN_Params():
    def __init__(self):
        self.state_enc_out_dim = 512 #64
        self.dim_state = 3
        self.in_channels = 3
        self.img_size = 32 
        self.batch_size = 64
        self.num_training_data = 2000  #easy: 2000, medium: 6000, hard: 7000
        self.num_epochs = 150 #100
        self.num_training_steps = int((self.num_training_data/self.batch_size) * self.num_epochs)  
        self.data_path = "/home/ext_drive/sampada_deglurkar/vae_stanford/training_easy/rgbs/*"
        self.op_save_path = "image_op_cgan.pth"
        self.m_save_path = "image_m_cgan.pth"
        self.op_training_loss_path = "image_op_cgan"
        self.m_training_loss_path = "image_m_cgan"
        self.test_data_path = "/home/ext_drive/sampada_deglurkar/vae_stanford/testing_easy/rgbs/*"
        self.op_test_model_path = "/home/sampada_deglurkar/VisualTreeSearch/src/solvers/image_op_cgan.pth"
        self.m_test_model_path = "/home/sampada_deglurkar/VisualTreeSearch/src/solvers/image_m_cgan.pth"
        self.test_output_path = "test_output_cgan"
        self.test_true_path = "test_true_cgan"
        self.lr = 1e-3 
        self.normalization = True
        self.device = 'cuda:1'

        self.train = False


