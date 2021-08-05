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
load_path = "vts_lightdark08-05-13_08_55"
cwd = os.getcwd()
model.load_model(cwd + "/nets/" + load_path + "/vts_pre_trained")

num_tests = 1
normalization_data = env.preprocess_data()
states, orientations, images = env.get_testing_batch(num_tests, normalization_data)

model.test_models(num_tests, states, orientations, images) 
