# author: @wangyunbo, @liubo
import math
import os.path
import shutil
from statistics import mean, stdev
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import time
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
from utils.utils import *

from configs.environments.stanford import *
from configs.solver.dualsmc_lightdark import *
from plotting.stanford import *
from src.environments.stanford import *
from src.solvers.dualsmc_lightdark import DualSMC

dlp = DualSMC_LightDark_Params()
sep = Stanford_Environment_Params()
 

def dualsmc(model, experiment_id, train, model_path):
    ################################
    # Create variables necessary for tracking diagnostics
    ################################
    step_list = []
    dist_list = []
    time_list_episode = []
    reward_list_episode = []
    episode_Z_loss = []
    episode_P_loss = []
    episode_T_loss = []
    episode_q1_loss = []
    episode_q2_loss = []
    rmse_per_step = np.zeros((sep.max_steps))
    tot_time = 0

    ################################
    # Create logs for diagnostics
    ################################
    if train:
        print("=========================\nTraining for iterations:", dlp.max_episodes_train)
        experiment_str = experiment_id + "/train"
        num_loops = dlp.max_episodes_train
    else:
        print("=========================\nTesting for iterations:", dlp.max_episodes_test)
        experiment_str = experiment_id + "/test"
        num_loops = dlp.max_episodes_test

    save_path = sep.ckpt + experiment_str
    img_path = sep.img + experiment_str
    check_path(save_path)
    check_path(img_path)
    txt_path = experiment_id + ".txt"
    txt_path_10 = experiment_id + "every_10_eps" + ".txt"
    file1 = open(save_path + "/" + txt_path, 'w+')
    file2 = open(save_path + "/" + txt_path_10, 'w+')

    real_display_iter = dlp.display_iter
    if train:
        training_tic = time.perf_counter()
        #real_display_iter *= 10

    env = StanfordEnvironment()

    normalization_data = env.preprocess_data()
    # Begin main dualSMC loop
    #tnow = time.time()
    for episode in range(num_loops):
        # tnext = time.time()
        # print("***EPISODE TIME***", tnext - tnow)
        # tnow = tnext

        #t0 = time.time()

        episode += 1

        if episode != 1:
            env.reset_environment()
        
        filter_dist = 0
        trajectory = []
        time_list_step = []
        reward_list_step = []

        hidden = np.zeros((dlp.num_lstm_layer, 1, dlp.dim_lstm_hidden))
        cell = np.zeros((dlp.num_lstm_layer, 1, dlp.dim_lstm_hidden))

        curr_state = env.state
        curr_orientation = env.orientation
        #tg = time.time()
        curr_obs, _, _, _ = env.get_observation(normalization_data=normalization_data) 
        #tf = time.time()
        #print("TIME BEFORE STEP GET_OBS", tf-tg)
        trajectory.append(curr_state)

        par_states, par_orientations = env.make_pars(dlp.num_par_pf)
        par_weight = torch.log(torch.ones((dlp.num_par_pf)).to(dlp.device) * (1.0 / float(dlp.num_par_pf)))
        normalized_weights = torch.softmax(par_weight, -1)
        mean_state = model.get_mean_state(par_states, normalized_weights).detach().cpu().numpy()

        if dlp.show_traj and episode % real_display_iter == 0:
            check_path(img_path + "/iters/")
            traj_dir = img_path + "/iters/" + "/iter-" + str(episode)
            if os.path.exists(traj_dir):
                shutil.rmtree(traj_dir)
            os.mkdir(traj_dir)
        
        if dlp.show_distr and episode % dlp.display_iter == 0:
            check_path(img_path + "/distrs/")
            distr_dir = img_path + "/distrs/" + "/distr-" + str(episode)
            if os.path.exists(distr_dir):
                shutil.rmtree(distr_dir)
            os.mkdir(distr_dir)

        num_par_propose = int(dlp.num_par_pf * dlp.pp_ratio)

        #t1 = time.time()
        #print("TIME BEFORE STEP:", t1-t0)

        #t0 = time.time()
        for step in range(sep.max_steps):
            #tstep = time.time()
            # 1. observation model
            # 2. planning
            # 3. re-sample
            # 4. transition model
            step_Z_loss = []
            step_P_loss = []
            step_T_loss = []
            step_q1_loss = []
            step_q2_loss = []
            #######################################
            # Observation model
            curr_obs_tensor = torch.FloatTensor(curr_obs).permute(2, 0, 1)  # [in_channels, img_size, img_size]
            #tm = time.time()
            if step == 0:
                lik, next_hidden, next_cell = model.measure_net.m_model(   ######### TODO!!! #########
                    torch.FloatTensor(par_states).to(dlp.device),
                    torch.FloatTensor(par_orientations).to(dlp.device),
                    curr_obs_tensor.unsqueeze(0).to(dlp.device),
                    torch.FloatTensor(hidden).to(dlp.device),
                    torch.FloatTensor(cell).to(dlp.device))
            else:
                lik, next_hidden, next_cell = model.measure_net.m_model(
                    torch.FloatTensor(par_states).to(dlp.device),
                    torch.FloatTensor(np.tile([curr_orientation], (dlp.num_par_pf, 1))).to(dlp.device),
                    curr_obs_tensor.unsqueeze(0).to(dlp.device),
                    torch.FloatTensor(hidden).to(dlp.device),
                    torch.FloatTensor(cell).to(dlp.device))
            #tm1 = time.time()
            #print("MEASURE MODEL", tm1-tm)
            par_weight += lik.squeeze()  # [num_par_pf]
            normalized_weights = torch.softmax(par_weight, -1)

            if dlp.show_distr and episode % real_display_iter == 0:
                if step < 10:
                    file_name = 'im00' + str(step)
                elif step < 100:
                    file_name = 'im0' + str(step)
                else:
                    file_name = 'im' + str(step)
                #frm_name = traj_dir + '/' + file_name + '_distr' + sep.fig_format
                #frm_name = img_path + "/distrs/" + file_name + '_distr' + sep.fig_format 
                frm_name = distr_dir + '/' + file_name + '_par' + sep.fig_format
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
            if dlp.smcp_mode == 'topk':
                # For some reason not using the normalized weights
                # Sample top 3 states and corresponding weights from our particle set
                weight_init, idx = torch.topk(par_weight, dlp.num_par_smc_init)  # both are [num_par_smc_init]
                idx = idx.detach().cpu().numpy()
            elif dlp.smcp_mode == 'samp':
                idx = torch.multinomial(normalized_weights, dlp.num_par_smc_init, replacement=True).detach().cpu().numpy()
                weight_init = par_weight[idx]
            # Make 10 copies of each starting weight so we now have 30 trajectory "samples" 
            weight_init = torch.softmax(weight_init, -1).unsqueeze(1).repeat(1, dlp.num_par_smc)  # [num_par_smc_init, num_par_smc]
            states_init = par_states[idx]  # [num_par_smc_init, dim_state]
            states_init_ = np.reshape(states_init, (1, dlp.num_par_smc_init, 1, sep.dim_state))  # [1, num_par_smc_init, 1, dim_state]
            # Make 10 copies of each starting state so we now have 30 trajectory "samples"
            smc_states = np.tile(states_init_, (dlp.horizon, 1, dlp.num_par_smc, 1))  # [horizon, num_par_smc_init, num_par_smc, dim_state]
            # For each starting state, apply 10 different actions 
            smc_action = np.zeros((dlp.horizon, dlp.num_par_smc, sep.dim_action))  # [horizon, num_par_smc, dim_action]
            # Weights on the actions 
            smc_weight = torch.log(torch.ones((dlp.num_par_smc)).to(dlp.device) * (1.0 / float(dlp.num_par_smc)))  # [num_par_smc]
            mean_state = np.reshape(mean_state, (1, 1, sep.dim_state))  # [1, 1, dim_state]
            # Also make 10 copies of the mean state in the particle set
            smc_mean_state = np.tile(mean_state, (dlp.horizon, dlp.num_par_smc, 1))  # [horizon, num_par_smc, dim_state]
            prev_q = 0

            for i in range(dlp.horizon):
                #thor = time.time()
                curr_smc_state = torch.FloatTensor(smc_states[i]).to(dlp.device)  # [num_par_smc_init, num_par_smc, dim_state]
                #tpolicy = time.time()
                # Policy takes the mean state and all of the 30 states and outputs 10 actions
                action, log_prob = model.policy.get_action(
                    torch.FloatTensor(smc_mean_state[i]).to(dlp.device),  
                    torch.transpose(curr_smc_state, 0, 1).contiguous().view(dlp.num_par_smc, -1))  # action [num_par_smc, dim_action]  log_prob [1]
                #tpolicy1 = time.time()
                #print("POLICY", tpolicy1-tpolicy)
                action_tile = action.unsqueeze(0).repeat(dlp.num_par_smc_init, 1, 1).view(-1, sep.dim_action)  # [num_par_smc * num_par_smc_init, dim_action]

                #tdynv = time.time()
                # Apply those 10 actions to each copy of the top 3 states
                next_smc_state = model.dynamic_net.t_model(
                    torch.FloatTensor(smc_states[i]).to(dlp.device).view(-1, sep.dim_state), 
                    action_tile * sep.step_range)  # [num_par_smc * num_par_smc_init, dim_state]
                #tdynv1 = time.time()
                #print("DYNAMIC1", tdynv1-tdynv)
                next_smc_state[:, 0] = torch.clamp(next_smc_state[:, 0], env.xrange[0], env.xrange[1])
                next_smc_state[:, 1] = torch.clamp(next_smc_state[:, 1], env.yrange[0], env.yrange[1])
                #next_smc_state[:, 2] = torch.clamp(next_smc_state[:, 2], env.thetas[0], env.thetas[1])
                next_smc_state = next_smc_state.view(dlp.num_par_smc_init, dlp.num_par_smc, sep.dim_state)

                #tdynw = time.time()
                # Apply the action to the mean state too
                mean_par = model.dynamic_net.t_model(
                    torch.FloatTensor(smc_mean_state[i]).to(dlp.device), action * sep.step_range)  # [num_par_smc, dim_state]
                #tdynw1 = time.time()
                #print("DYNAMIC2", tdynw1-tdynw)
                mean_par[:, 0] = torch.clamp(mean_par[:, 0], env.xrange[0], env.xrange[1])
                mean_par[:, 1] = torch.clamp(mean_par[:, 1], env.yrange[0], env.yrange[1])

                if i < dlp.horizon - 1:
                    smc_action[i] = action.detach().cpu().numpy()
                    smc_states[i + 1] = next_smc_state.detach().cpu().numpy()
                    smc_mean_state[i + 1] = mean_par.detach().cpu().numpy()

                #tq = time.time()
                # Get the Q-values for each of the 30 trajectories
                q = model.get_q(curr_smc_state.view(-1, sep.dim_state), action_tile).view(dlp.num_par_smc_init, -1)  # [num_par_smc_init, num_par_smc]
                #tq1 = time.time()
                #print("Q", tq1-tq)
                advantage = q - prev_q - log_prob.unsqueeze(0).repeat(dlp.num_par_smc_init, 1)  # [num_par_smc_init, num_par_smc]
                # Update the weights on each trajectory "sample" by the advantage
                # Sum over the 3 top states
                advantage = torch.sum(weight_init * advantage, 0).squeeze()  # [num_par_smc]
                smc_weight += advantage
                prev_q = q
                normalized_smc_weight = F.softmax(smc_weight, -1)  # [num_par_smc]

                if dlp.smcp_resample and (i % dlp.smcp_resample_step == 0):
                    # Resample 10 actions based on the action/trajectory weights 
                    idx = torch.multinomial(normalized_smc_weight, dlp.num_par_smc, replacement=True).detach().cpu().numpy()
                    smc_action = smc_action[:, idx, :]
                    smc_states = smc_states[:, :, idx, :]
                    smc_mean_state = smc_mean_state[:, idx, :]
                    smc_weight = torch.log(torch.ones((dlp.num_par_smc)).to(dlp.device) * (1.0 / float(dlp.num_par_smc)))
                    normalized_smc_weight = F.softmax(smc_weight, -1)  # [num_par_smc]
                
                #thor1 = time.time()
                #print("TIME TO PLAN HORIZON 1 STEP", thor1-thor)

            # For plotting 
            smc_xy = np.reshape(smc_states[:, :, :, :2], (-1, dlp.num_par_smc_init * dlp.num_par_smc, 2))

            # The action is randomly chosen out of the 10 --- not sure why the horizon is involved here
            if dlp.smcp_resample and (dlp.horizon % dlp.smcp_resample_step == 0):
                n = np.random.randint(dlp.num_par_smc, size=1)[0]
            else:
                n = Categorical(normalized_smc_weight).sample().detach().cpu().item()
            action = smc_action[0, n, :]
            #######################################
            if step % dlp.pf_resample_step == 0:
                #if False:
                if dlp.pp_exist:
                    idx = torch.multinomial(normalized_weights, dlp.num_par_pf - num_par_propose,
                                            replacement=True).detach().cpu().numpy()
                    resample_state = par_states[idx]  # [num_par_pf - num_par_propose, dim_state]
                    proposal_state = model.pp_net(curr_obs_tensor.unsqueeze(0).to(dlp.device), 
                                                torch.FloatTensor([curr_orientation]).unsqueeze(0).to(dlp.device), 
                                                num_par_propose)
                    proposal_state[:, 0] = torch.clamp(proposal_state[:, 0], env.xrange[0], env.xrange[1])
                    proposal_state[:, 1] = torch.clamp(proposal_state[:, 1], env.yrange[0], env.yrange[1])
                    #proposal_state[:, 2] = torch.clamp(proposal_state[:, 2], env.thetas[0], env.thetas[1])
                    proposal_state = proposal_state.detach().cpu().numpy()
                    par_states = np.concatenate((resample_state, proposal_state), 0)  # [num_par_pf, dim_state]
                else:
                    idx = torch.multinomial(normalized_weights, dlp.num_par_pf, replacement=True).detach().cpu().numpy()
                    par_states = par_states[idx]

                par_weight = torch.log(torch.ones((dlp.num_par_pf)).to(dlp.device) * (1.0 / float(dlp.num_par_pf)))
                normalized_weights = torch.softmax(par_weight, -1)  # [num_par_pf]

            mean_state = model.get_mean_state(par_states, normalized_weights).detach().cpu().numpy()
            filter_rmse = math.sqrt(pow(mean_state[0] - curr_state[0], 2) + pow(mean_state[1] - curr_state[1], 2))
            rmse_per_step[step] += filter_rmse
            filter_dist += filter_rmse

            toc = time.perf_counter()
            #print("TIME TO PLAN", toc-tic)
            #######################################
            
            if dlp.show_traj and episode % real_display_iter == 0:
                if step < 10:
                    file_name = 'im00' + str(step)
                elif step < 100:
                    file_name = 'im0' + str(step)
                else:
                    file_name = 'im' + str(step)
                frm_name = traj_dir + '/' + file_name + '_par' + sep.fig_format

                if dlp.pp_exist and step % dlp.pf_resample_step == 0:
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
                    # plot_par(xlim, ylim, goal, [trap1, trap2], dark, frm_name, curr_state, 
                    #        mean_state, resample_state, proposal_state, smc_xy)
                    # plot_par(xlim, ylim, goal, [trap1, trap2], dark, frm_name, curr_state, 
                    #         mean_state, par_states, normalized_weights.cpu().numpy(), None, smc_xy)
                    plot_par(xlim, ylim, goal, [trap1, trap2], dark, frm_name, curr_state, 
                           mean_state, resample_state, normalized_weights.cpu().numpy(), proposal_state, smc_xy)
            
            #######################################
            # Update the environment
            reward = env.step(action * sep.step_range)
            next_state = env.state
            next_orientation = env.orientation
            #ts = time.time()
            next_obs, _, _, _ = env.get_observation(normalization_data=normalization_data)
            #tt = time.time()
            #print("TIME DURING STEP GET_OBS", tt-ts)
            #######################################
            #ttrain = time.time()
            if train:
                model.replay_buffer.push(curr_state, action, reward, next_state, env.done, curr_obs_tensor,
                                         curr_s, mean_state, hidden, cell, states_init, curr_orientation)
                if len(model.replay_buffer) > dlp.batch_size:
                    critic_update = (step % dlp.critic_update == 0)  
                    p_loss, t_loss, z_loss, q1_loss, q2_loss = model.soft_q_update(critic_update)

                    step_P_loss.append(p_loss.item())
                    step_T_loss.append(t_loss.item())
                    step_Z_loss.append(z_loss.item())
                    step_q1_loss.append(q1_loss.item())
                    step_q2_loss.append(q2_loss.item())
            #ttrain1 = time.time()
            #print("TRAINING TIME", ttrain1-ttrain)
            #######################################
            # Transition Model
            #tdyn = time.time()
            par_states = model.dynamic_net.t_model(torch.FloatTensor(par_states).to(dlp.device),
                                                   torch.FloatTensor(action * sep.step_range).to(dlp.device))
            #tdyn1 = time.time()
            #print("DYNAMIC MODEL", tdyn1-tdyn)
            par_states[:, 0] = torch.clamp(par_states[:, 0], env.xrange[0], env.xrange[1])
            par_states[:, 1] = torch.clamp(par_states[:, 1], env.yrange[0], env.yrange[1])
            #par_states[:, 2] = torch.clamp(par_states[:, 2], env.thetas[0], env.thetas[1])
            par_states = par_states.detach().cpu().numpy()

            #######################################
            curr_state = next_state
            curr_orientation = next_orientation
            curr_obs = next_obs
            hidden = next_hidden.detach().cpu().numpy()
            cell = next_cell.detach().cpu().numpy()
            trajectory.append(next_state)
            # Recording data
            time_this_step = toc - tic
            time_list_step.append(time_this_step)
            reward_list_step.append(reward)
            
            if env.done:
                break

            #tstep1 = time.time()
            #print("TIME FOR ONE STEP", tstep1-tstep)

        #t1 = time.time()
        #print("TIME AFTER STEPS", t1-t0)

        # Get the average loss of each model for this episode if we are training
        if train:
            if len(model.replay_buffer) > dlp.batch_size:
                episode_P_loss.append(np.mean(step_P_loss))
                episode_T_loss.append(np.mean(step_T_loss))
                episode_Z_loss.append(np.mean(step_Z_loss))
                episode_q1_loss.append(np.mean(step_q1_loss))
                episode_q2_loss.append(np.mean(step_q2_loss))
        

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

        if episode >= dlp.summary_iter:
            step_list.pop(0)
            dist_list.pop(0)
        
        reach = np.array(step_list) < (sep.max_steps - 1)

        if episode % dlp.save_iter == 0 and train:
            model.save_model(model_path + "/dpf_lightdark_online")
            print("Saving online trained models to %s" % model_path)

        if episode % real_display_iter == 0:
            episode_list = [episode_P_loss, episode_T_loss, episode_Z_loss, episode_q1_loss, episode_q2_loss]
            st2 = img_path + "/"
            name_list = ['P_loss', 'transition_loss', 'Z_loss', 'sac_1_loss', 'sac_2_loss']
            if train:
                visualize_learning(st2, episode_list, time_list_episode, step_list, reward_list_episode, episode, name_list)
            else:
                visualize_learning(st2, None, time_list_episode, step_list, reward_list_episode, episode, name_list)
            
            interaction = 'Episode %s: cumulative success rate = %s, mean/stdev steps taken = %s / %s, reward = %s / %s, avg_plan_time = %s / %s, avg_dist = %s / %s' % (
                episode, np.mean(reach), np.mean(step_list), np.std(step_list), np.mean(reward_list_episode), np.std(reward_list_episode),
                np.mean(time_list_episode), np.std(time_list_episode), np.mean(dist_list), np.std(dist_list))
            print('\r{}'.format(interaction))
            file2.write('\n{}'.format(interaction))
            file2.flush()

        if (train and episode % dlp.display_iter == 0) or (not train):
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
        
        interaction = 'Episode %s: cumulative success rate = %s, steps = %s, reward = %s, avg_plan_time = %s, avg_dist = %s' % (
            episode, np.mean(reach), step, tot_reward, avg_time_this_episode, filter_dist)
        print('\r{}'.format(interaction))
        file1.write('\n{}'.format(interaction))
        file1.flush()


    if train:
        training_toc = time.perf_counter()
        training_time = training_toc - training_tic
        training_time_str = 'Time elapsed for online training:  %s seconds.' % (
            training_time)
        print('\r{}'.format(training_time_str))
        file1.write('\n{}'.format(training_time_str))
        file1.flush()


    rmse_per_step = rmse_per_step / num_loops
    # print(rmse_per_step) - not sure why this is relevant...
    file1.close()
    file2.close()



def dualsmc_driver(load_path=None, end_to_end=True, save_model=True, test=True):
    # This block of code creates the folders for plots
    experiment_id = "dualsmc_lightdark" + get_datetime()
    model_path = "nets/" + experiment_id
    check_path(model_path)

    check_path("data")
    check_path("nets")

    # Create a model object
    model = DualSMC()

    # Let the user load in a previous model
    if load_path is not None:
        cwd = os.getcwd()
        model.load_model(cwd + "/nets/" + load_path)

    if end_to_end:
        train = True
        # After pretraining move into the end to end training
        dualsmc(model, experiment_id, train, model_path)

    if save_model:
        # Save the model
        model.save_model(model_path + "/dpf_online")
        print("Saving online trained models to %s" % model_path)

    if test:
        train = False
        dualsmc(model, experiment_id, train, model_path)


if __name__ == "__main__":
    if dlp.model_name == 'dualsmc_lightdark':
        dualsmc_driver(load_path=None, end_to_end=True, save_model=True, test=True)