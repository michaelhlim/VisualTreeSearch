class Stanford_Environment_Params():
    def __init__(self):
        self.epi_reward = 100 
        self.step_reward = -1 
        self.dim_action = 1
        self.velocity = 0.2 
        self.dim_state = 2
        self.img_size = 32 
        self.dim_obs = 4
        self.obs_std_light = 0.01
        self.obs_std_dark = 0.1
        self.step_range = 1 
        self.max_steps = 200  
        self.noise_amount = 0.4 
        self.occlusion_amount = 15
        self.salt_vs_pepper = 0.5
        self.fig_format = '.png'
        self.img = 'data/img/'
        self.ckpt = 'data/ckpt/'
        self.training_data_path = "/home/sampada/vae_stanford/training_hallway/rgbs/*"
        self.testing_data_path = "/home/sampada/vae_stanford/testing_hallway/rgbs/*"
        self.normalization = True


