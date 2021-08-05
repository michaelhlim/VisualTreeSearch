# author: @wangyunbo, @liubo
import math
import os.path
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


def vts_lightdark(model, experiment_id, train, model_path):
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


    env = StanfordEnvironment()

    # Begin main dualSMC loop
    for episode in range(num_loops):
        episode += 1

        if episode != 1:
            env.reset_environment()

        filter_dist = 0
        trajectory = []
        time_list_step = []
        reward_list_step = []

        hidden = np.zeros((vlp.num_lstm_layer, 1, vlp.dim_lstm_hidden))
        cell = np.zeros((vlp.num_lstm_layer, 1, vlp.dim_lstm_hidden))

        curr_state = env.state
        curr_orientation = env.orientation
        curr_obs, _, _, _ = env.get_observation() 
        trajectory.append(curr_state)

        par_states, par_orientations = env.make_pars(vlp.num_par_pf)
        par_weight = torch.log(torch.ones((vlp.num_par_pf)).to(vlp.device) * (1.0 / float(vlp.num_par_pf)))
        normalized_weights = torch.softmax(par_weight, -1)
        mean_state = model.get_mean_state(par_states, normalized_weights).detach().cpu().numpy()

        pft_planner = PFTDPW(env, model.measure_net, model.generator)

        if vlp.show_traj and episode % vlp.display_iter == 0:
            check_path(img_path + "/iters/")
            traj_dir = img_path + "/iters/" + "/iter-" + str(episode)
            if os.path.exists(traj_dir):
                shutil.rmtree(traj_dir)
            os.mkdir(traj_dir)


        num_par_propose = int(vlp.num_par_pf * vlp.pp_ratio)

        for step in range(sep.max_steps):
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
                lik, _, _ = model.measure_net.m_model(     # [1, num_par]
                    torch.FloatTensor(par_states).to(vlp.device),
                    torch.FloatTensor(par_orientations).to(vlp.device),
                    curr_obs_tensor.unsqueeze(0).to(vlp.device),
                    torch.FloatTensor(hidden).to(vlp.device),
                    torch.FloatTensor(cell).to(vlp.device))
            else:
                lik, _, _ = model.measure_net.m_model(     # [1, num_par]
                    torch.FloatTensor(par_states).to(vlp.device),
                    torch.FloatTensor(np.tile([curr_orientation], (vlp.num_par_pf, 1))).to(vlp.device),
                    curr_obs_tensor.unsqueeze(0).to(vlp.device),
                    torch.FloatTensor(hidden).to(vlp.device),
                    torch.FloatTensor(cell).to(vlp.device))
            par_weight += lik.squeeze()  # [num_par_pf]
            normalized_weights = torch.softmax(par_weight, -1)

            if vlp.show_distr and episode % vlp.display_iter == 0:
                check_path(img_path + "/distrs/")
                if step < 10:
                    file_name = 'im00' + str(step)
                elif step < 100:
                    file_name = 'im0' + str(step)
                else:
                    file_name = 'im' + str(step)
                #frm_name = traj_dir + '/' + file_name + '_distr' + sep.fig_format
                frm_name = img_path + "/distrs/" + file_name + '_distr' + sep.fig_format  
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
            #######################################
            
            # Resampling
            if step % vlp.pf_resample_step == 0:
                #if False:
                if vlp.pp_exist:
                    idx = torch.multinomial(normalized_weights, vlp.num_par_pf - num_par_propose,
                                            replacement=True).detach().cpu().numpy()
                    resample_state = par_states[idx]  # [num_par_pf - num_par_propose, dim_state]
                    proposal_state = model.pp_net(curr_obs_tensor.unsqueeze(0).to(vlp.device), 
                                                torch.FloatTensor([curr_orientation]).unsqueeze(0).to(vlp.device), 
                                                num_par_propose)
                    proposal_state[:, 0] = torch.clamp(proposal_state[:, 0], env.xrange[0], env.xrange[1])
                    proposal_state[:, 1] = torch.clamp(proposal_state[:, 1], env.yrange[0], env.yrange[1])
                    proposal_state = proposal_state.detach().cpu().numpy()
                    par_states = np.concatenate((resample_state, proposal_state), 0)  # [num_par_pf, dim_state]
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

                if vlp.pp_exist and step % vlp.pf_resample_step == 0:
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
                    plot_par(xlim, ylim, goal, [trap1, trap2], dark, frm_name, curr_state, 
                            mean_state, resample_state, normalized_weights.cpu().numpy(), proposal_state, None)
                    #plot_par(xlim, ylim, goal, [trap1, trap2], dark, frm_name, curr_state, 
                    #        mean_state, par_states, normalized_weights.cpu().numpy(), None, None)

            #######################################
            # Update the environment
            reward = env.step(action * sep.step_range)
            next_state = env.state
            next_orientation = env.orientation
            next_obs, _, _, _ = env.get_observation()
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
            hidden = 0
            cell = 0
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

        if episode % vlp.save_iter == 0 and train:
            model.save_model(model_path + "/vts_online")
            print("Saving online trained models to %s" % model_path)

        if episode % vlp.display_iter == 0:
            st2 = img_path + "/"
            episode_list = [episode_P_loss, episode_Z_loss, episode_G_loss]
            name_list = ['particle_loss', 'observation_loss', 'generative_loss']
            if train:
                visualize_learning(st2, episode_list, time_list_episode, step_list, reward_list_episode, episode, name_list)
            else:
                visualize_learning(st2, None, time_list_episode, step_list, reward_list_episode, episode, name_list)

            interaction = 'Episode %s: mean/stdev steps taken = %s / %s, reward = %s / %s, avg_plan_time = %s / %s, avg_dist = %s / %s' % (
                episode, np.mean(step_list), np.std(step_list), np.mean(reward_list_episode), np.std(reward_list_episode),
                np.mean(time_list_episode), np.std(time_list_episode), np.mean(dist_list), np.std(dist_list))
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
        plot_maze(xlim, ylim, goal, [trap1, trap2], dark, figure_name=st, states=np.array(trajectory))


        # Repeat the above code block for writing to the text file every episode instead of every 10
        interaction = 'Episode %s: steps = %s, reward = %s, avg_plan_time = %s, avg_dist = %s' % (
            episode, step, tot_reward, avg_time_this_episode, filter_dist)
        print('\r{}'.format(interaction))
        file1.write('\n{}'.format(interaction))
        file1.flush()

    rmse_per_step = rmse_per_step / num_loops
    # print(rmse_per_step) - not sure why this is relevant...
    file1.close()
    file2.close()


def vts_lightdark_driver(load_path=None, gen_load_path=None, pre_training=True, save_pretrained_model=True,
                   end_to_end=True, save_online_model=True, test=True):
    # This block of code creates the folders for plots
    experiment_id = "vts_lightdark" + get_datetime()
    foldername = "data/" + experiment_id
    check_path(foldername)
    model_path = "nets/" + experiment_id
    check_path(model_path)

    check_path("data")
    check_path("nets")

    # Create a model and environment object
    model = VTS()
    env = StanfordEnvironment() 

    #observation_generator = ObservationGenerator()

    # Let the user load in a previous model
    if load_path is not None:
        cwd = os.getcwd()
        model.load_model(cwd + "/nets/" + load_path + "/vts_pre_trained", load_g=False)
        # observation_generator.load_model(
        #     cwd + "/nets/" + gen_load_path + "/gen_pre_trained")

    if pre_training:
        tic = time.perf_counter()
        print("Pretraining observation density model, particle proposer, and observation generator")
        print_freq = 50
        measure_loss = []
        proposer_loss = []
        generator_loss = []

        steps_per_epoch = int(np.ceil(vlp.num_training_data/vlp.batch_size))
        normalization_data = env.preprocess_data()
        
        steps = []
        # Train Z and P 
        for epoch in range(vlp.num_epochs_zp):
            data_files_indices = env.shuffle_dataset()

            for step in range(steps_per_epoch):

                states, orientations, images, par_batch = env.get_training_batch(vlp.batch_size, data_files_indices, 
                                                                                step, normalization_data, vlp.num_par_pf)
                states = torch.from_numpy(states).float()
                images = torch.from_numpy(images).float()
                images = images.permute(0, 3, 1, 2)  # [batch_size, in_channels, 32, 32]
                state_batch = states
                obs_batch = images  
                #par_batch = env.get_par_batch(states)

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

        
        steps = []
        # Train G
        for epoch in range(vlp.num_epochs_g):
            data_files_indices = env.shuffle_dataset()

            for step in range(steps_per_epoch):

                states, orientations, images, _ = env.get_training_batch(vlp.batch_size, data_files_indices, 
                                                        step, normalization_data, vlp.num_par_pf)
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

        # Observation generative model
        # training_time = observation_generator.pretrain(save_pretrained_model, model_path)

        if save_pretrained_model:
            model.save_model(model_path + "/vts_pre_trained")
            print("Saving pre-trained Z, P, G models to %s" % model_path)
        
        toc = time.perf_counter()
        time_this_step = toc - tic
        print("Time elapsed for pre-training: ", time_this_step, "seconds.")

    if end_to_end:
        train = True
        # After pretraining move into the end to end training
        vts_lightdark(model, experiment_id,
            train, model_path)

    if save_online_model:
        # Save the model
        model.save_model(model_path + "/vts_online_trained")
        print("Saving online trained Z, P models to %s" % model_path)

    if test:
        train = False
        vts_lightdark(model, experiment_id,
            train, model_path)


if __name__ == "__main__":
    if vlp.model_name == 'vts_lightdark':
        # Right into online learning & testing
        # vts_lightdark_driver(load_path="test500k",
                #    gen_load_path="test500k", pre_training=False)

        # Just pre-training
        #vts_lightdark_driver(load_path="vts_lightdark08-04-14_14_35", end_to_end=False, save_online_model=False, test=False)
        vts_lightdark_driver(end_to_end=False, save_online_model=False, test=False)

        # Pre-training immediately followed by testing
        # vts_lightdark_driver(end_to_end=False, save_online_model=False)

        # Just testing
        # vts_lightdark_driver(load_path="test500k",
        #           gen_load_path="test500k", pre_training=False, end_to_end=False, save_online_model=False)

        # Everything
        # vts_lightdark_driver()

