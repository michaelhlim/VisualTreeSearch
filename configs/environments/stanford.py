class Stanford_Environment_Params():
    def __init__(self):
        self.epi_reward = 100
        self.step_reward = -1
        self.dim_action = 1
        self.velocity = 0.2 #0.1 #0.5 
        self.dim_state = 2
        self.img_size = 32 
        self.obs_std = 0.01
        self.step_range = 1 #0.05
        self.max_steps = 200
        self.discount = 0.99
        self.noise_amount = 0.4 #0.8 #0.4 #1.0 #0.4 #0.15
        self.occlusion_amount = 20
        self.fig_format = '.png'
        self.img = 'data/img/'
        self.ckpt = 'data/ckpt/'

        self.training_data_path = "/home/ext_drive/sampada_deglurkar/vae_stanford/training_hallway/rgbs/*"
        self.testing_data_path = "/home/ext_drive/sampada_deglurkar/vae_stanford/testing_hallway/rgbs/*"
        # self.training_data_path = "/home/sampada/training_hallway/rgbs/*"
        # self.testing_data_path = "/home/sampada/testing_hallway/rgbs/*"
        


