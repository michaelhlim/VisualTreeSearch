class Stanford_Environment_Params():
    def __init__(self):
        self.epi_reward = 100
        self.step_reward = -1
        self.dim_action = 1
        self.velocity = 0.1 #0.5 #0.1
        self.dim_state = 3
        self.img_size = 32 
        self.dim_obs = 4
        self.obs_std = 0.01
        self.end_range = 0.01
        self.step_range = 1 #0.05
        self.max_steps = 200
        self.discount = 0.99
        self.fig_format = '.png'
        self.img = 'data/img/'
        self.ckpt = 'data/ckpt/'
        


