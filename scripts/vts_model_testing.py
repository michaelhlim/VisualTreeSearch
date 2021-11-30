import cv2
import numpy
import torch 

from src.environments.stanford import *
from src.solvers.vts_lightdark import VTS

from configs.environments.stanford import *
from configs.solver.vts_lightdark import *

vlp = VTS_LightDark_Params()
sep = Stanford_Environment_Params()


model = VTS(shared_enc=True)
env = StanfordEnvironment(disc_thetas=True) 
load_paths = ["vts_lightdark11-30-12_51_28"]

cwd = os.getcwd()
if len(load_paths) > 1:
    model.load_model(cwd + "/nets/" + load_paths[0] + "/vts_pre_trained", load_g=False) # Load Z/P
    model.load_model(cwd + "/nets/" + load_paths[1] + "/vts_pre_trained", load_zp=False) # Load G
else:
    model.load_model(cwd + "/nets/" + load_paths[0] + "/vts_pre_trained") # Load all models

normalization_data = env.preprocess_data()

num_tests = 1
states, orientations, images, blurred_images = env.get_testing_batch(num_tests, normalization_data)
model.test_models_encobs(num_tests, states, orientations, images, blurred_images, env) 


# num_tests = 1
# states, orientations, images, blurred_images = env.get_testing_batch(num_tests, normalization_data)
# model.test_models(num_tests, states, orientations, images, blurred_images, env) 

# num_tests = 1
# states, orientations, images, blurred_images = env.get_testing_batch(num_tests, normalization_data)
# model.test_models_old(num_tests, states, orientations, images, blurred_images, env) 

# num_tests = 200
# states, orientations, images, blurred_images = env.get_testing_batch(num_tests, normalization_data)
# model.test_tsne("enc", "encoder_conv", vlp.obs_encode_out, num_tests, states, orientations, images, blurred_images)
#model.test_tsne("g", "decoder", vlp.obs_encode_out, num_tests, states, orientations, images, blurred_images)
#model.test_tsne("zp", "mlp", 1, num_tests, states, orientations, images, blurred_images)

