# author: @sdeglurkar, @jatucker4, @michaelhlim

import cv2
import math
import numpy as np
import os.path
import random
import shutil
from statistics import mean, stdev
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import time
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from utils.utils import *

# VTS w/ No LSTM
from configs.environments.stanford import *
from configs.solver.vts_lightdark import *

from plotting.stanford import *

from src.environments.stanford import *
from src.methods.vts_lightdark.pftdpw_lightdark import *
from src.solvers.vts_lightdark import VTS


vlp = VTS_LightDark_Params()
sep = Stanford_Environment_Params()


def vts_lightdark(model, experiment_id, train, model_path, 
                shared_enc=False, independent_enc=False,
                test_env_is_diff=False, test_img_is_diff=False):
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
    rmse_per_step = np.zeros((sep.max_steps))
    tot_time = 0
    ################################
    # Create logs for diagnostics
    ################################
    if train:
        print("=========================\nTraining for iterations:", vlp.max_episodes_train)
        experiment_str = experiment_id + "/train"
        num_loops = vlp.max_episodes_train
    else:
        print("=========================\nTesting for iterations:", vlp.max_episodes_test)
        experiment_str = experiment_id + "/test"
        num_loops = vlp.max_episodes_test

    save_path = sep.ckpt + experiment_str
    img_path = sep.img + experiment_str
    check_path(save_path)
    check_path(img_path)
    txt_path = experiment_id + ".txt"
    txt_path_10 = experiment_id + "every_10_eps" + ".txt"
    file1 = open(save_path + "/" + txt_path, 'w+')
    file2 = open(save_path + "/" + txt_path_10, 'w+')


    env = StanfordEnvironment(disc_thetas=True)

    # If the test environment is different - ie there's a new trap
    if not train and test_env_is_diff:
        env.set_test_trap(test_trap_is_random=True)

    normalization_data = env.preprocess_data()

    # Begin main dualSMC loop
    for episode in range(num_loops):
        episode += 1

        if episode != 1:
            env.reset_environment()

        filter_dist = 0
        trajectory = []
        time_list_step = []
        reward_list_step = []

        curr_state = env.state
        curr_orientation = env.orientation
        if not train and test_img_is_diff:
            curr_obs, curr_pre_norm, _, _, _ = env.get_observation(normalization_data=normalization_data, occlusion=True)
        else:     
            curr_obs, curr_pre_norm, _, _, _ = env.get_observation(normalization_data=normalization_data) 

        trajectory.append(curr_state)

        par_states, par_orientations = env.make_pars(vlp.num_par_pf)   
        par_weight = torch.log(torch.ones((vlp.num_par_pf)).to(vlp.device) * (1.0 / float(vlp.num_par_pf)))
        normalized_weights = torch.softmax(par_weight, -1)
        mean_state = model.get_mean_state(par_states, normalized_weights).detach().cpu().numpy()  # Goes into replay buffer

        pft_planner = PFTDPW(env, model.measure_net, model.generator, shared_enc)  # Why is this created in each episode

        if vlp.show_traj and episode % vlp.display_iter == 0:
            check_path(img_path + "/iters/")
            traj_dir = img_path + "/iters/" + "/iter-" + str(episode)
            if os.path.exists(traj_dir):
                shutil.rmtree(traj_dir)
            os.mkdir(traj_dir)

            check_path(img_path + "/obs/")
            img_dir = img_path + "/obs/" + "/iter-" + str(episode)
            if os.path.exists(img_dir):
                shutil.rmtree(img_dir)
            os.mkdir(img_dir)
        
        if vlp.show_distr and episode % vlp.display_iter == 0:
            check_path(img_path + "/distrs/")
            distr_dir = img_path + "/distrs/" + "/distr-" + str(episode)
            if os.path.exists(distr_dir):
                shutil.rmtree(distr_dir)
            os.mkdir(distr_dir)
                

        num_par_propose = int(vlp.num_par_pf * vlp.pp_ratio)

        for step in range(sep.max_steps):

            if vlp.show_traj and episode % vlp.display_iter == 0:
                if step < 10:
                    file_name = 'im00' + str(step)
                elif step < 100:
                    file_name = 'im0' + str(step)
                else:
                    file_name = 'im' + str(step)
                img_name = img_dir + '/' + file_name + sep.fig_format

                cv2.imwrite(img_name, curr_pre_norm)

            # 1. observation model
            # 2. planning
            # 3. re-sample
            # 4. transition model
            step_Z_loss = []
            step_P_loss = []
            step_G_loss = []
            #######################################
            # Observation model
            curr_obs_tensor = torch.FloatTensor(curr_obs).permute(2, 0, 1)  # [in_channels, img_size, img_size]
            if step == 0:
                lik = model.measure_net.m_model(     # [1, num_par]
                    torch.FloatTensor(par_states).to(vlp.device),
                    torch.FloatTensor(par_orientations).to(vlp.device),
                    curr_obs_tensor.unsqueeze(0).to(vlp.device))
            else:
                lik = model.measure_net.m_model(     # [1, num_par]
                    torch.FloatTensor(par_states).to(vlp.device),
                    torch.FloatTensor(np.tile([curr_orientation], (vlp.num_par_pf, 1))).to(vlp.device),
                    curr_obs_tensor.unsqueeze(0).to(vlp.device))

            par_weight += lik.squeeze()  # [num_par_pf]
            normalized_weights = torch.softmax(par_weight, -1)

            if vlp.show_distr and episode % vlp.display_iter == 0:
                if step < 10:
                    file_name = 'im00' + str(step)
                elif step < 100:
                    file_name = 'im0' + str(step)
                else:
                    file_name = 'im' + str(step)
                frm_name = distr_dir + '/' + file_name + '_par' + sep.fig_format
                weights = normalized_weights.detach().cpu().numpy()
                fig1, ax1 = plt.subplots()
                plt.hist(weights, bins=np.logspace(-5, 0, 50))
                ax1.set_xscale("log")
                ax1.set_xlim(1e-5, 1e0)
                plt.savefig(frm_name)
                plt.close()

            curr_s = par_states.copy()  # Goes into replay buffer
            tic = time.perf_counter()
            #######################################
            # Planning
            states_init = par_states   # Goes into replay buffer
            # action = pft_planner.solve(par_states, normalized_weights.detach().cpu().numpy()) # Action already includes velocity
            action, future_actions = pft_planner.solve_viz(par_states, normalized_weights.detach().cpu().numpy())
            # For visualizing the planned trajectory
            mean_s = model.get_mean_state(par_states, normalized_weights).detach().cpu().numpy()

            #######################################
            # Resampling
            if step % vlp.pf_resample_step == 0:
                if vlp.pp_exist:
                    # For de-localization problem: don't propose so many particles
                    if vlp.pp_decay:
                        # Exponential decay in proposed particles according to decay rate
                        decayed_num_propose = int(num_par_propose * vlp.decay_rate**step)
                        if decayed_num_propose <= 0:
                            proposal_state = None
                        else:
                            proposal_state = model.pp_net(curr_obs_tensor.unsqueeze(0).to(vlp.device), 
                                                torch.FloatTensor([curr_orientation]).unsqueeze(0).to(vlp.device), 
                                                decayed_num_propose)

                        # Particles not from the proposer - sample with replacement from existing set
                        idx = torch.multinomial(normalized_weights, vlp.num_par_pf - decayed_num_propose,
                                            replacement=True).detach().cpu().numpy()
                    
                    # For de-localization problem: don't propose so many particles
                    elif vlp.pp_std:
                        # Only propose particles if existing particle stdev is above threshold
                        std_norm = np.linalg.norm(np.std(par_states, axis=0))
                        if std_norm >= vlp.std_thres:
                            # Propose in an amount proportional to the stdev, capped at num_par_propose
                            std_num_propose = int(min(num_par_propose, vlp.std_alpha*std_norm))
                            proposal_state = model.pp_net(curr_obs_tensor.unsqueeze(0).to(vlp.device), 
                                                torch.FloatTensor([curr_orientation]).unsqueeze(0).to(vlp.device), 
                                                std_num_propose)
                        else: # Otherwise don't propose particles 
                            proposal_state = None
                            std_num_propose = 0

                        # Particles not from the proposer - sample with replacement from existing set
                        idx = torch.multinomial(normalized_weights, vlp.num_par_pf - std_num_propose,
                                            replacement=True).detach().cpu().numpy() 

                    # For de-localization problem: don't propose so many particles
                    elif vlp.pp_effective:
                        # Only propose particles if the "effective number of particles" is below 
                        # a threshold
                        # Formula comes from Gustafsson 2010
                        effective_num_particles = vlp.num_par_pf/(1 + vlp.num_par_pf**2 * 
                                                    torch.var(normalized_weights.detach()))
                        if effective_num_particles < vlp.effective_thres:
                            # Propose in an amount inversely proportional to the effective num particles, 
                            # capped at num_par_propose
                            effective_num_propose = int(min(num_par_propose, 
                                                    vlp.effective_alpha / effective_num_particles))
                            proposal_state = model.pp_net(curr_obs_tensor.unsqueeze(0).to(vlp.device), 
                                                torch.FloatTensor([curr_orientation]).unsqueeze(0).to(vlp.device), 
                                                effective_num_propose)    
                        else: # Otherwise don't propose particles 
                            proposal_state = None
                            effective_num_propose = 0

                        # Particles not from the proposer - sample with replacement from existing set
                        idx = torch.multinomial(normalized_weights, vlp.num_par_pf - effective_num_propose,
                                        replacement=True).detach().cpu().numpy() 
                    
                    # Otherwise propose fixed amount of particles
                    else:
                        proposal_state = model.pp_net(curr_obs_tensor.unsqueeze(0).to(vlp.device), 
                                                torch.FloatTensor([curr_orientation]).unsqueeze(0).to(vlp.device), 
                                                num_par_propose)
                        # Particles not from the proposer - sample with replacement from existing set
                        idx = torch.multinomial(normalized_weights, vlp.num_par_pf - num_par_propose,
                                            replacement=True).detach().cpu().numpy()
                    # Particles not from the proposer - sample with replacement from existing set
                    resample_state = par_states[idx]  # [num_par_pf - number of particles proposed, dim_state]
                    
                    if proposal_state is not None:
                        # Keep proposals within environment bounds
                        proposal_state[:, 0] = torch.clamp(proposal_state[:, 0], env.xrange[0], env.xrange[1])
                        proposal_state[:, 1] = torch.clamp(proposal_state[:, 1], env.yrange[0], env.yrange[1])
                        proposal_state = proposal_state.detach().cpu().numpy()

                        par_states = np.concatenate(
                            (resample_state, proposal_state), 0)  # [num_par_pf, dim_state]
                    else:
                        par_states = resample_state  # [num_par_pf, dim_state]

                # No proposer
                else:
                    idx = torch.multinomial(normalized_weights, vlp.num_par_pf, replacement=True).detach().cpu().numpy()
                    par_states = par_states[idx]

                par_weight = torch.log(torch.ones((vlp.num_par_pf)).to(vlp.device) * (1.0 / float(vlp.num_par_pf)))
                normalized_weights = torch.softmax(par_weight, -1)  # [num_par_pf]

            mean_state = model.get_mean_state(
                par_states, normalized_weights).detach().cpu().numpy()
            filter_rmse = math.sqrt(pow(mean_state[0] - curr_state[0], 2) + pow(mean_state[1] - curr_state[1], 2))
            rmse_per_step[step] += filter_rmse
            filter_dist += filter_rmse

            toc = time.perf_counter()

            #######################################
            if vlp.show_traj and episode % vlp.display_iter == 0:
                if step < 10:
                    file_name = 'im00' + str(step)
                elif step < 100:
                    file_name = 'im0' + str(step)
                else:
                    file_name = 'im' + str(step)
                frm_name = traj_dir + '/' + file_name + '_par' + sep.fig_format

                # if vlp.pp_exist and step % vlp.pf_resample_step == 0:
                if vlp.pp_exist: 
                    xlim = env.xrange
                    ylim = env.yrange
                    goal = [env.target_x[0], env.target_y[0], 
                            env.target_x[1]-env.target_x[0], env.target_y[1]-env.target_y[0]]
                    trap1_x = env.trap_x[0]
                    trap2_x = env.trap_x[1]
                    trap1 = [trap1_x[0], env.trap_y[0], 
                            trap1_x[1]-trap1_x[0], env.trap_y[1]-env.trap_y[0]]
                    trap2 = [trap2_x[0], env.trap_y[0], 
                            trap2_x[1]-trap2_x[0], env.trap_y[1]-env.trap_y[0]]
                    dark = [env.xrange[0], env.yrange[0], env.xrange[1]-env.xrange[0], env.dark_line-env.yrange[0]]
                    
                    state_iterate = mean_state.reshape((1, 2))
                    planned_traj = state_iterate
                    for action in future_actions:
                        state_iterate, _, _, _ = env.transition(state_iterate, None, action)
                        state_iterate = state_iterate[:, :2]
                        planned_traj = np.vstack([planned_traj, state_iterate])
                    planned_traj = np.array(planned_traj)
                    planned_traj = planned_traj.reshape((planned_traj.shape[0], 1, planned_traj.shape[1]))
                    
                    if not train and test_env_is_diff:
                        test_trap1_x = env.test_trap_x[0]
                        test_trap2_x = env.test_trap_x[1]
                        test_trap1_y = env.test_trap_y[0]
                        test_trap2_y = env.test_trap_y[1]
                        test_trap1 = [test_trap1_x[0], test_trap1_y[0], 
                            test_trap1_x[1]-test_trap1_x[0], test_trap1_y[1]-test_trap1_y[0]]
                        test_trap2 = [test_trap2_x[0], test_trap2_y[0], 
                            test_trap2_x[1]-test_trap2_x[0], test_trap2_y[1]-test_trap2_y[0]]
                        test_trap_plot_params = [test_trap1, test_trap2]
                    
                        check_path(img_path + "/traj/")
                        st = img_path + "/traj/" + str(episode) + "-" + file_name + "-trj" + sep.fig_format
                        plot_maze(xlim, ylim, goal, [trap1, trap2], test_trap_plot_params,
                          dark, figure_name=st, states=np.array(trajectory))
                        plot_par(xlim, ylim, goal, [trap1, trap2], test_trap_plot_params, 
                                 dark, frm_name, curr_state, mean_state, resample_state, 
                                 normalized_weights.detach().cpu().numpy(), proposal_state, planned_traj)
                    else:
                        check_path(img_path + "/traj/")
                        st = img_path + "/traj/" + str(episode) + "-" + file_name + "-trj" + sep.fig_format
                        plot_maze(xlim, ylim, goal, [trap1, trap2], None,
                          dark, figure_name=st, states=np.array(trajectory))
                        plot_par(xlim, ylim, goal, [trap1, trap2], None, 
                                 dark, frm_name, curr_state, mean_state, resample_state, 
                                 normalized_weights.detach().cpu().numpy(), proposal_state, planned_traj)

            #######################################
            # Update the environment
            reward = env.step(action * sep.step_range, action_is_vector=True)
            next_state = env.state
            next_orientation = env.orientation
            if not train and test_img_is_diff:
                next_obs, next_pre_norm, _, _, _ = env.get_observation(normalization_data=normalization_data, occlusion=True)
            else:  
                next_obs, next_pre_norm, _, _, _ = env.get_observation(normalization_data=normalization_data)
            #######################################
            if train:
                model.replay_buffer.push(curr_state, action, reward, next_state, env.done, curr_obs_tensor,
                                         curr_s, mean_state, states_init, curr_orientation)
                if len(model.replay_buffer) > vlp.batch_size:
                    p_loss, z_loss, obs_gen_loss = \
                        model.online_training()

                    step_P_loss.append(p_loss.item())
                    step_Z_loss.append(z_loss.item())
                    step_G_loss.append(obs_gen_loss.item())

            #######################################
            # Transition Model
            next_par_states, _, _, _ = env.transition(par_states, normalized_weights.detach().cpu().numpy(), action)
            par_states = next_par_states[:, :sep.dim_state]    
            #######################################            
            curr_state = next_state
            curr_orientation = next_orientation
            curr_obs = next_obs
            curr_pre_norm = next_pre_norm
            trajectory.append(next_state)
            # Recording data
            time_this_step = toc - tic
            time_list_step.append(time_this_step)
            reward_list_step.append(reward)

            if env.done:
                break

        # TODO TRY BOTH MEAN AND NOT FOR THE LOSS PLOTS
        # Get the average loss of each model for this episode if we are training
        if train:
            if len(model.replay_buffer) > vlp.batch_size:
                episode_P_loss.append(np.mean(step_P_loss))
                episode_Z_loss.append(np.mean(step_Z_loss))
                episode_G_loss.append(np.mean(step_G_loss))

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

        #if episode >= dlp.summary_iter:
            #step_list.pop(0)
            #dist_list.pop(0)
        
        reach = np.array(step_list) < (sep.max_steps - 1)
        
        # Take only the statistics for successful episodes
        reach_steps = np.array(step_list)[reach] 
        reach_rewards = np.array(reward_list_episode)[reach]
        reach_times = np.array(time_list_episode)[reach]
        reach_dists = np.array(dist_list)[reach]

        if episode % vlp.save_iter == 0 and train:
            model.save_model(model_path + "/vts_online", shared_enc=shared_enc, indep_enc=independent_enc)
            print("Saving online trained models to %s" % model_path)


        if episode % vlp.display_iter == 0:
            st2 = img_path + "/"
            episode_list = [episode_P_loss, episode_Z_loss, episode_G_loss]
            name_list = ['particle_loss', 'observation_loss', 'generative_loss']
            if train:
                visualize_learning(st2, episode_list, time_list_episode, step_list, reward_list_episode, episode, name_list)
            else:
                visualize_learning(st2, None, time_list_episode, step_list, reward_list_episode, episode, name_list)

            if len(reach_steps) == 0:  # No episodes were successful - return a null value
                rs = [-1, -1]
            else:
                rs = [np.mean(reach_steps), np.std(reach_steps)]
            if len(reach_rewards) == 0:  # No episodes were successful - return a null value
                rr = [-1, -1]
            else:
                rr = [np.mean(reach_rewards), np.std(reach_rewards)]
            if len(reach_times) == 0:  # No episodes were successful - return a null value
                rt = [-1, -1]
            else:
                rt = [np.mean(reach_times), np.std(reach_times)]
            if len(reach_dists) == 0:  # No episodes were successful - return a null value
                rd = [-1, -1]
            else:
                rd = [np.mean(reach_dists), np.std(reach_dists)]
            interaction = 'Episode %s: cumulative success rate = %s, mean/stdev steps taken = %s / %s, reward = %s / %s, avg_plan_time = %s / %s, avg_dist = %s / %s' % (
                episode, np.mean(reach), rs[0], rs[1], rr[0], rr[1],
                rt[0], rt[1], rd[0], rd[1])
            print('\r{}'.format(interaction))
            file2.write('\n{}'.format(interaction))
            file2.flush()

            
        # Plot every trajectory
        check_path(img_path + "/traj/")
        st = img_path + "/traj/" + str(episode) + "-trj" + sep.fig_format
        xlim = env.xrange
        ylim = env.yrange
        goal = [env.target_x[0], env.target_y[0], 
                env.target_x[1]-env.target_x[0], env.target_y[1]-env.target_y[0]]
        trap1_x = env.trap_x[0]
        trap2_x = env.trap_x[1]
        trap1 = [trap1_x[0], env.trap_y[0], 
                trap1_x[1]-trap1_x[0], env.trap_y[1]-env.trap_y[0]]
        trap2 = [trap2_x[0], env.trap_y[0], 
                trap2_x[1]-trap2_x[0], env.trap_y[1]-env.trap_y[0]]
        dark = [env.xrange[0], env.yrange[0], env.xrange[1]-env.xrange[0], env.dark_line-env.yrange[0]]

        if not train and test_env_is_diff:
            test_trap1_x = env.test_trap_x[0]
            test_trap2_x = env.test_trap_x[1]
            test_trap1_y = env.test_trap_y[0]
            test_trap2_y = env.test_trap_y[1]
            test_trap1 = [test_trap1_x[0], test_trap1_y[0], 
                test_trap1_x[1]-test_trap1_x[0], test_trap1_y[1]-test_trap1_y[0]]
            test_trap2 = [test_trap2_x[0], test_trap2_y[0], 
                test_trap2_x[1]-test_trap2_x[0], test_trap2_y[1]-test_trap2_y[0]]
            test_trap_plot_params = [test_trap1, test_trap2]
          
            plot_maze(xlim, ylim, goal, [trap1, trap2], test_trap_plot_params,
                          dark, figure_name=st, states=np.array(trajectory))
        else:
            plot_maze(xlim, ylim, goal, [trap1, trap2], None, 
                         dark, figure_name=st, states=np.array(trajectory))


        # Repeat the above code block for writing to the text file every episode instead of every 10
        interaction = 'Episode %s: cumulative success rate = %s, steps = %s, reward = %s, avg_plan_time = %s, avg_dist = %s' % (
            episode, np.mean(reach), step, tot_reward, avg_time_this_episode, filter_dist)
        print('\r{}'.format(interaction))
        file1.write('\n{}'.format(interaction))
        file1.flush()

    rmse_per_step = rmse_per_step / num_loops
    file1.close()
    file2.close()


def vts_lightdark_driver(shared_enc=False, independent_enc=False, load_paths=None, pre_training=True, save_pretrained_model=True,
                   end_to_end=True, save_online_model=True, test=True, test_env_is_diff=False,
                   test_img_is_diff=False):

    torch.manual_seed(vlp.torch_seed)
    random.seed(vlp.random_seed)
    np.random.seed(vlp.np_random_seed)

    if independent_enc and not shared_enc:
        raise Exception("If the encoder is independent, it must also be shared!")

    # This block of code creates the folders for plots
    experiment_id = "vts_lightdark" + get_datetime()
    foldername = "data/pretraining/" + experiment_id
    check_path(foldername)
    model_path = "nets/" + experiment_id
    check_path(model_path)

    check_path("data")
    check_path("nets")

    # Create a model and environment object
    model = VTS(shared_enc, independent_enc)
    env = StanfordEnvironment(disc_thetas=True) 

    # Let the user load in a previous model
    if load_paths is not None:
        cwd = os.getcwd()
        if len(load_paths) > 1:
            model.load_model(cwd + "/nets/" + load_paths[0] + "/vts_pre_trained", load_g=False,
                shared_enc=shared_enc, indep_enc=independent_enc) # Load Z/P
            model.load_model(cwd + "/nets/" + load_paths[1] + "/vts_pre_trained", load_zp=False,
                shared_enc=shared_enc, indep_enc=independent_enc) # Load G
        else:
            model.load_model(cwd + "/nets/" + load_paths[0] + "/vts_pre_trained",
                shared_enc=shared_enc, indep_enc=independent_enc) # Load all models

    if pre_training:
        tic = time.perf_counter()
        print("Pretraining observation density model, particle proposer, and observation generator")
        print_freq = 50
        measure_loss = []
        proposer_loss = []
        generator_loss = []
        encoder_loss = []

        steps_per_epoch = int(np.ceil(vlp.num_training_data/vlp.batch_size))
        normalization_data = env.preprocess_data()


        ###################
        # Train independent encoder 
        ###################

        # For training the independent encoder with noisy images
        # Pre-generate the corrupted indices in the image
        # Noise in the image plane
        if vlp.num_epochs_e != 0:
            diff_pattern = vlp.diff_pattern
            s_vs_p = sep.salt_vs_pepper
            image_plane_size = sep.img_size**2
            num_salt = np.ceil(sep.noise_amount * image_plane_size * s_vs_p)
            num_pepper = np.ceil(sep.noise_amount * image_plane_size * (1. - s_vs_p))
            if diff_pattern: # Same noise pattern in all dark images or a different noise pattern per image?
                # Pre-generate the corrupted indices per image in the training data
                noise_list = []
                for i in range(len(env.training_data_files)):
                    noise_list.append(np.random.choice(image_plane_size, int(num_salt + num_pepper), replace=False)) 
            else:
                noise_list = np.random.choice(image_plane_size, int(num_salt + num_pepper), replace=False) 
            noise_list = np.array(noise_list)


        steps = []
        for epoch in range(vlp.num_epochs_e):
            data_files_indices = env.shuffle_dataset()

            for step in range(steps_per_epoch):

                states, orientations, images, _ = env.get_training_batch(vlp.batch_size, data_files_indices, 
                                                        step, normalization_data, vlp.num_par_pf, 
                                                        noise_list=noise_list, noise_amount=sep.noise_amount)
                states = torch.from_numpy(states).float()
                images = torch.from_numpy(images).float()
                images = images.permute(0, 3, 1, 2)  # [batch_size, in_channels, 32, 32]
                state_batch = states
                obs_batch = images  

                E_loss = model.pretraining_independent_encoder(state_batch, orientations, obs_batch)
                encoder_loss.append(E_loss.item())

                # Print loss and stuff for the last $print_freq batches
                if step % print_freq == 0:
                    print("Epoch: ", epoch, "Step: ", step, ", E loss: ", np.mean(encoder_loss[-print_freq:]))
                    steps.append(epoch * steps_per_epoch + step)
        
        if vlp.num_epochs_e != 0:
            plt.figure()
            plt.plot(steps, np.array(encoder_loss)[steps], label="Independent Encoder Training Loss")
            plt.xlabel("Training Steps")
            plt.ylabel("Total Loss")
            plt.legend()
            plt.savefig(foldername + "/indep_e_loss.png")

        if save_pretrained_model:
            model.save_model(model_path + "/vts_pre_trained", shared_enc=shared_enc, indep_enc=independent_enc)
            print("Saving pre-trained (independent) encoder model to %s" % model_path)

        tocq = time.perf_counter()
        time_this_step = tocq - tic
        print("Time elapsed for pre-training independent encoder: ", time_this_step, "seconds.")

        
        ###################
        # Train Z and P 
        ###################

        steps = [] 
        for epoch in range(vlp.num_epochs_zp):
            noise_amount = 0
            if epoch >= vlp.num_epochs_zp/4:
                noise_amount = sep.noise_amount/4
            if epoch >= vlp.num_epochs_zp/2:
                noise_amount = sep.noise_amount/2
            if epoch >= 3*vlp.num_epochs_zp/4:
                noise_amount = sep.noise_amount

            data_files_indices = env.shuffle_dataset()

            for step in range(steps_per_epoch):

                states, orientations, images, par_batch = env.get_training_batch(vlp.batch_size, data_files_indices, 
                                                                                step, normalization_data, vlp.num_par_pf,
                                                                                noise_amount=noise_amount)
                states = torch.from_numpy(states).float()
                images = torch.from_numpy(images).float()
                images = images.permute(0, 3, 1, 2)  # [batch_size, in_channels, 32, 32]
                state_batch = states
                obs_batch = images  

                Z_loss, P_loss = model.pretraining_zp(
                    state_batch, orientations, obs_batch, par_batch)
                measure_loss.append(Z_loss.item())
                proposer_loss.append(P_loss.item())

                # Print loss and stuff for the last $print_freq batches
                if step % print_freq == 0:
                    print("Epoch: ", epoch, ", Step: ", step, ", Z loss: ", np.mean(
                        measure_loss[-print_freq:]), ", P loss: ", np.mean(proposer_loss[-print_freq:]))
                    steps.append(epoch * steps_per_epoch + step)
        
        plt.figure()
        plt.plot(steps, np.array(measure_loss)[steps], label="Observation Model Training Loss")
        plt.xlabel("Training Steps")
        plt.ylabel("Total Loss")
        plt.legend()
        plt.savefig(foldername + "/z_loss.png")

        plt.figure()
        plt.plot(steps, np.array(proposer_loss)[steps], label="Proposer Training Loss")
        plt.xlabel("Training Steps")
        plt.ylabel("Total Loss")
        plt.legend()
        plt.savefig(foldername + "/p_loss.png")

        
        if save_pretrained_model:
            model.save_model(model_path + "/vts_pre_trained", shared_enc=shared_enc, indep_enc=independent_enc)
            print("Saving pre-trained Z, P models to %s" % model_path)

        tocc = time.perf_counter()
        time_this_step = tocc - tic
        print("Time elapsed for pre-training Z and P: ", time_this_step, "seconds.")


        ###################
        # Train generator 
        ###################

        # For training the generator with noisy images
        # Pre-generate the corrupted indices in the image
        # Noise in the image plane

        diff_pattern = vlp.diff_pattern  
        s_vs_p = sep.salt_vs_pepper
        image_plane_size = sep.img_size**2
        num_salt = np.ceil(sep.noise_amount * image_plane_size * s_vs_p)
        num_pepper = np.ceil(sep.noise_amount * image_plane_size * (1. - s_vs_p))
        if diff_pattern: # Same noise pattern in all dark images or a different noise pattern per image?
            # Pre-generate the corrupted indices per image in the training data
            noise_list = []
            for i in range(len(env.training_data_files)):
                noise_list.append(np.random.choice(image_plane_size, int(num_salt + num_pepper), replace=False)) 
        else:
            noise_list = np.random.choice(image_plane_size, int(num_salt + num_pepper), replace=False) 
        noise_list = np.array(noise_list)

        steps = []
        for epoch in range(vlp.num_epochs_g):
            data_files_indices = env.shuffle_dataset()

            for step in range(steps_per_epoch):
                states, orientations, images, _ = env.get_training_batch(vlp.batch_size, data_files_indices, 
                                                        step, normalization_data, vlp.num_par_pf, 
                                                        noise_list=noise_list, noise_amount=sep.noise_amount)

                states = torch.from_numpy(states).float()
                images = torch.from_numpy(images).float()
                images = images.permute(0, 3, 1, 2)  # [batch_size, in_channels, 32, 32]
                state_batch = states
                obs_batch = images  

                G_loss = model.pretraining_g(
                    state_batch, orientations, obs_batch)
                generator_loss.append(G_loss.item())

                # Print loss and stuff for the last $print_freq batches
                if step % print_freq == 0:
                    print("Epoch: ", epoch, "Step: ", step, ", G loss: ", np.mean(generator_loss[-print_freq:]))
                    steps.append(epoch * steps_per_epoch + step)

        plt.figure()
        plt.plot(steps, np.array(generator_loss)[steps], label="Generator Training Loss")
        plt.xlabel("Training Steps")
        plt.ylabel("Total Loss")
        plt.legend()
        plt.savefig(foldername + "/g_loss.png")

        if save_pretrained_model:
            model.save_model(model_path + "/vts_pre_trained", shared_enc=shared_enc, indep_enc=independent_enc)
            print("Saving pre-trained Z, P, G models to %s" % model_path)
        
        toc = time.perf_counter()
        time_this_step = toc - tic
        print("Time elapsed for pre-training all models: ", time_this_step, "seconds.")

        train_save_path = sep.ckpt + experiment_id + "/train"
        check_path(train_save_path)
        train_file_str = experiment_id + "_train.txt"
        train_file = open(train_save_path + "/" + train_file_str, 'w+')
        training_time = 'Time elapsed for pre-training: %s ' % (time_this_step)
        train_file.write('\n{}'.format(training_time))
        train_file.flush()
        

    if end_to_end:
        train = True
        # After pretraining move into the end to end training
        vts_lightdark(model, experiment_id,
            train, model_path, shared_enc, independent_enc,
            test_env_is_diff, test_img_is_diff)

    if save_online_model:
        # Save the model
        model.save_model(model_path + "/vts_online_trained", shared_enc=shared_enc, indep_enc=independent_enc)
        print("Saving online trained Z, P models to %s" % model_path)

    if test:
        train = False
        vts_lightdark(model, experiment_id,
            train, model_path, shared_enc, independent_enc,
            test_env_is_diff, test_img_is_diff)


if __name__ == "__main__":
    if vlp.model_name == 'vts_lightdark':
        if len(sys.argv) > 1 and sys.argv[1] == "--pretrain":
            # Just pre-training
            vts_lightdark_driver(end_to_end=False, save_online_model=False, test=False)
        elif len(sys.argv) > 1 and sys.argv[1] == "--pretrain-test":
            # Pre-training immediately followed by testing
            vts_lightdark_driver(end_to_end=False, save_online_model=False)
        elif len(sys.argv) > 1 and sys.argv[1] == "--test":
            # Just testing
            load_path = sys.argv[2]
            vts_lightdark_driver(load_paths=[load_path], 
                pre_training=False, end_to_end=False, save_online_model=False)
        elif len(sys.argv) > 1 and sys.argv[1] == "--online-train-test":
            # Right into online learning & testing
            vts_lightdark_driver(load_paths="test500k", pre_training=False)
        elif len(sys.argv) > 1 and sys.argv[1] == "--test-traps":
            # Generalization Experiment 1
            load_path = sys.argv[2]
            vts_lightdark_driver(load_paths=[load_path], 
                       pre_training=False, end_to_end=False, save_online_model=False, test_env_is_diff=True)
        elif len(sys.argv) > 1 and sys.argv[1] == "--occlusions":
            # Generalization Experiment 2
            load_path = sys.argv[2]
            vts_lightdark_driver(load_paths=[load_path], 
                       pre_training=False, end_to_end=False, save_online_model=False, test_env_is_diff=False, 
                       test_img_is_diff=True)
        elif len(sys.argv) > 1:
            print("Unknown argument!")
        else:
            # Everything
            vts_lightdark_driver()


