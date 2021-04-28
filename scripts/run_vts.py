# author: @wangyunbo, @liubo
import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import shutil
import math
import time
from utils.utils import *
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

# VTS w/ No LSTM
from src.solvers.vts import VTS
from src.solvers.generative_observation_prediction import *
from src.methods.pftdpw.pftdpw import *
from src.environments.floor import *
from plotting.floor import *
from configs.environments.floor import *
from configs.solver.dualsmc import *
from configs.solver.pftdpw import *
from statistics import mean, stdev

def vts(model, observation_generator, experiment_id, foldername, train):
    ################################
    # Create variables necessary for tracking diagnostics
    ################################
    step_list = []
    dist_list = []
    time_list_episode = []
    reward_list_episode = []
    episode_Z_loss = []
    episode_P_loss = []
    episode_G_loss = []
    rmse_per_step = np.zeros((MAX_STEPS))
    tot_time = 0
    ################################
    # Create logs for diagnostics
    ################################
    save_path = CKPT + experiment_id
    img_path = IMG + experiment_id
    check_path(save_path)
    check_path(img_path)
    str123 = experiment_id + ".txt"
    str1234 = experiment_id + "every_10_eps" + ".txt"
    file1 = open(foldername + "/" + str123, 'w+')
    file2 = open(foldername + "/" + str1234, 'w+')

    # Begin main dualSMC loop
    for episode in range(MAX_EPISODES):
        episode += 1
        env = Environment()
        filter_dist = 0
        trajectory = []
        time_list_step = []
        reward_list_step = []

        hidden = np.zeros((NUM_LSTM_LAYER, 1, DIM_LSTM_HIDDEN))
        cell = np.zeros((NUM_LSTM_LAYER, 1, DIM_LSTM_HIDDEN))

        curr_state = env.state
        curr_obs = env.get_observation()
        trajectory.append(curr_state)

        par_states = np.random.rand(NUM_PAR_PF, 2)
        par_states[:, 0] = par_states[:, 0] * 0.4 + 0.8
        par_states[:, 1] = par_states[:, 1] * 0.3 + 0.1 + np.random.randint(2, size=NUM_PAR_PF) * 0.5
        par_weight = torch.log(torch.ones((NUM_PAR_PF)).to(device) * (1.0 / float(NUM_PAR_PF)))
        normalized_weights = torch.softmax(par_weight, -1)
        mean_state = model.get_mean_state(par_states, normalized_weights).detach().cpu().numpy()

        pft_planner = PFTDPW(env, model.measure_net, observation_generator)

        if SHOW_TRAJ and episode % DISPLAY_ITER == 0:
            traj_dir = img_path + "/iter-" + str(episode)
            if os.path.exists(traj_dir):
                shutil.rmtree(traj_dir)
            os.mkdir(traj_dir)

        num_par_propose = int(NUM_PAR_PF * PP_RATIO)
        for step in range(MAX_STEPS):
            # 1. observation model
            # 2. planning
            # 3. re-sample
            # 4. transition model
            step_Z_loss = []
            step_P_loss = []
            step_G_loss = []
            #######################################
            # Observation model
            lik, _, _ = model.measure_net.m_model(
                torch.FloatTensor(par_states).to(device),
                torch.FloatTensor(curr_obs).to(device),
                torch.FloatTensor(hidden).to(device),
                torch.FloatTensor(cell).to(device))
            par_weight += lik.squeeze()  # (NUM_PAR_PF)
            normalized_weights = torch.softmax(par_weight, -1)

            if SHOW_DISTR and episode % DISPLAY_ITER == 0:
                if step < 10:
                    file_name = 'im00' + str(step)
                elif step < 100:
                    file_name = 'im0' + str(step)
                else:
                    file_name = 'im' + str(step)
                frm_name = traj_dir + '/' + file_name + '_distr' + FIG_FORMAT
                weights = normalized_weights.detach().cpu().numpy()
                fig1, ax1 = plt.subplots()
                plt.hist(weights, bins=np.logspace(-5, 0, 50))
                ax1.set_xscale("log")
                ax1.set_xlim(1e-5, 1e0)
                plt.savefig(frm_name)
                plt.close()

            curr_s = par_states.copy()
            tic = time.perf_counter()
            #######################################
            # Planning
            states_init = par_states
            action = pft_planner.solve(par_states, normalized_weights.detach().cpu().numpy())

            mean_state = model.get_mean_state(par_states, normalized_weights).detach().cpu().numpy()
            filter_rmse = math.sqrt(pow(mean_state[0] - curr_state[0], 2) + pow(mean_state[1] - curr_state[1], 2))
            rmse_per_step[step] += filter_rmse
            filter_dist += filter_rmse

            toc = time.perf_counter()
            #######################################
            if SHOW_TRAJ and episode % DISPLAY_ITER == 0:
                if step < 10:
                    file_name = 'im00' + str(step)
                elif step < 100:
                    file_name = 'im0' + str(step)
                else:
                    file_name = 'im' + str(step)
                frm_name = traj_dir + '/' + file_name + '_par' + FIG_FORMAT

                if PP_EXIST and step % PF_RESAMPLE_STEP == 0:
                    plot_par(frm_name, curr_state, mean_state, resample_state, proposal_state, smc_xy)

            #######################################
            # Update the environment
            reward = env.step(action)
            next_state = env.state
            next_obs = env.get_observation()
            #######################################
            if train:
                model.replay_buffer.push(curr_state, action, reward, next_state, env.done, curr_obs,
                                         curr_s, mean_state, states_init)
                if len(model.replay_buffer) > BATCH_SIZE:
                    p_loss, z_loss, obs_gen_loss = \
                        model.soft_q_update(observation_generator)

                    step_P_loss.append(p_loss.item())
                    step_Z_loss.append(z_loss.item())
                    step_G_loss.append(obs_gen_loss.item())
            #######################################
            # Transition Model
            par_states, _, _ = env.transition(par_states, normalized_weights.detach().cpu().numpy(), action)

            #######################################            
            curr_state = next_state
            curr_obs = next_obs
            hidden = 0
            cell = 0
            trajectory.append(next_state)
            # Recording data
            time_this_step = toc - tic
            time_list_step.append(time_this_step)
            reward_list_step.append(reward)

            if step % 5 == 0:
                print(step, curr_state, action)
            if env.done:
                break

        # TODO TRY BOTH MEAN AND NOT FOR THE LOSS PLOTS
        # Get the average loss of each model for this episode if we are training
        if train:
            episode_P_loss.append(mean(step_P_loss))
            episode_Z_loss.append(mean(step_Z_loss))
            episode_G_loss.append(mean(step_G_loss))

        # Get the sum of the episode time
        tot_time = sum(time_list_step)
        # Get the running average of the time
        avg_time_this_episode = tot_time / len(time_list_step)
        time_list_episode.append(avg_time_this_episode)

        # Get the total reward this episode
        tot_reward = sum(reward_list_step)
        reward_list_episode.append(tot_reward)

        filter_dist = filter_dist / (step + 1)
        dist_list.append(filter_dist)
        step_list.append(step)

        if episode >= SUMMARY_ITER:
            step_list.pop(0)
            dist_list.pop(0)

        if episode % SAVE_ITER == 0:
            model_path = save_path + '_' + str(episode)
            model.save_model(model_path)
            print("save model to %s" % model_path)

        if episode % DISPLAY_ITER == 0:
            episode_list = [episode_P_loss, episode_Z_loss, episode_G_loss]
            st2 = img_path + "/" + str(episode)
            name_list = ['particle_loss', 'observation_loss', 'generative_loss']
            if train:
                visualize_learning(st2, episode_list, time_list_episode, step_list, reward_list_episode, episode, name_list)
            else:
                visualize_learning(st2, None, time_list_episode, step_list, reward_list_episode, episode, name_list)
            st = img_path + "/" + str(episode) + "-trj" + FIG_FORMAT
            print("plotting ... save to %s" % st)
            plot_maze(figure_name=st, states=np.array(trajectory))

            if episode >= SUMMARY_ITER:
                total_iter = SUMMARY_ITER
            else:
                total_iter = episode

            interaction = 'Episode %s: mean/stdev steps taken = %s / %s, reward = %s / %s, avg_plan_time = %s / %s, avg_dist = %s / %s' % (
                episode, mean(step_list), stdev(step_list), mean(reward_list_episode), stdev(reward_list_episode),
                mean(time_list_episode), stdev(time_list_episode), mean(dist_list), stdev(dist_list))
            print('\r{}'.format(interaction))
            file2.write('\n{}'.format(interaction))
            file2.flush()

        # Repeat the above code block for writing to the text file every episode instead of every 10
        if episode >= SUMMARY_ITER:
            total_iter = SUMMARY_ITER
        else:
            total_iter = episode

        interaction = 'Episode %s: steps = %s, reward = %s, avg_plan_time = %s, avg_dist = %s' % (
            episode, step, tot_reward, avg_time_this_episode, sum(dist_list) / total_iter)
        print('\r{}'.format(interaction))
        file1.write('\n{}'.format(interaction))
        file1.flush()

    rmse_per_step = rmse_per_step / MAX_EPISODES
    print(rmse_per_step)
    file1.close()
    file2.close()


def vts_driver(load_path=None, pre_training=True, save_pretrained_model=True,
                   end_to_end=True, save_model=True, test=True):
    # This block of code creates the folders for plots
    settings = "data/vts_indv"
    foldername = settings + get_datetime()
    os.mkdir(foldername)
    experiment_id = "vts" + get_datetime()
    save_path = CKPT + experiment_id
    check_path(save_path)

    # Create a model and environment object
    model = VTS()
    env = Environment()

    observation_generator = ObservationGenerator()

    # Let the user load in a previous model
    if load_path is not None:
        model.load_model(load_path)

    # This is where we need to perform individual training (if the user wants).
    # The process for this is to (1) create a observation and state batch.
    # Then (2) only train Z and P in an adversarial manner using the custom
    # soft_q_update function

    if pre_training:
        print("Beginning pre-training")
        print_freq = 50
        measure_loss = []
        proposer_loss = []
        # First we'll do train individually for 64 batches
        for batch in range(50):
            state_batch, obs_batch, par_batch = env.make_batch(64)
            # Pull a random state and observation from the batch
            # curr_state = random.choice(state_batch)
            # curr_obs = random.choice(obs_batch)
            curr_state = state_batch
            curr_obs = obs_batch

            # Create the current particle variable for soft q update
            curr_s = par_batch

            # Train Z and P using the soft q update function
            Z_loss, P_loss = model.soft_q_update_individual(curr_state, curr_obs, curr_s)
            measure_loss.append(Z_loss.item())
            proposer_loss.append(P_loss.item())

            # Print loss and stuff
            if batch % print_freq == 0:
                print(batch, np.mean(measure_loss), np.mean(proposer_loss))

        # Observation generative model
        training_time = observation_generator.pretrain()

        if save_pretrained_model:
            model_path = save_path + "pre_trained_" + str(pre_training)
            model.save_model(model_path)
            print("saving pre-trained model to %s" % model_path)

    if end_to_end:
        train = True
        # After pretraining move into the end to end training
        vts(model, observation_generator, experiment_id, foldername, train)

    if save_model:
        # Save the model
        model.save_model(save_path + "after_training")
        print("saving model to %s" % save_path)

    if test:
        train = False
        vts(model, observation_generator, experiment_id, foldername, train)


if __name__ == "__main__":
    if MODEL_NAME == 'dualsmc':
        vts_driver()

