# author: @wangyunbo

device = 'cuda:0'
torch_seed = 10
random_seed = 10
np_random_seed = 10

######################
# Network
MODEL_NAME = 'dualsmc'
DIM_HIDDEN = 256
DIM_LSTM_HIDDEN = 128
NUM_LSTM_LAYER = 2
DIM_ENCODE = 64

######################
# Training
TRAIN = True
MAX_EPISODES_TRAIN = 10000
MAX_EPISODES_TEST = 1000
BATCH_SIZE = 64
FIL_LR = 1e-3 # filtering
PLA_LR = 1e-3 # planning
SUMMARY_ITER = 100
SAVE_ITER = 100
DISPLAY_ITER = 10
PRETRAIN = 500000
SHOW_TRAJ = True
SHOW_DISTR = False

######################
# Filtering
PF_RESAMPLE_STEP = 3
NUM_PAR_PF = 100
PP_EXIST = True
PP_RATIO = 0.3
PP_LOSS_TYPE = 'adv'  # 'mse', 'adv', 'density'
PP_DECAY = True
DECAY_RATE = 0.9
PP_STD = False
STD_THRES = 0.07
STD_ALPHA = 150
PP_EFFECTIVE = False
EFFECTIVE_THRES = int(2/3 * NUM_PAR_PF)
EFFECTIVE_ALPHA = 1000

# ######################
# Planning
NUM_PAR_SMC_INIT = 3
NUM_PAR_SMC = 30
HORIZON = 10
SMCP_MODE = 'topk' # 'samp', 'topk'
SMCP_RESAMPLE = True
SMCP_RESAMPLE_STEP = 3

# ######################
# SAC
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
replay_buffer_size = 100000
alpha = 1.0
gamma = 0.95
tau = 0.005
const = 1e-6

