import cv2
import numpy
import torch 

from src.environments.stanford import *
from src.solvers.vts_lightdark import VTS

from configs.environments.stanford import *
from configs.solver.vts_lightdark import *

vlp = VTS_LightDark_Params()
sep = Stanford_Environment_Params()


model = VTS()
env = StanfordEnvironment() 
load_path = "vts_lightdark09-08-15_29_14"
cwd = os.getcwd()
model.load_model(cwd + "/nets/" + load_path + "/vts_pre_trained")

normalization_data = env.preprocess_data()

num_tests = 1
states, orientations, images, blurred_images = env.get_testing_batch(num_tests, normalization_data)
#model.test_models(num_tests, states, orientations, images, blurred_images) 

num_tests = 200
states, orientations, images, blurred_images = env.get_testing_batch(num_tests, normalization_data)
model.test_tsne("enc", "encoder", vlp.obs_encode_out, num_tests, states, orientations, images, blurred_images)

