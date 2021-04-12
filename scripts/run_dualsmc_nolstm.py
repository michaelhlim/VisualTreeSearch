# author: @wangyunbo, @liubo
# from experiments.configs import *
import sys
import shutil
import math
import time
import random
from utils.utils import *
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from statistics import mean, stdev
# DualSMC w/ No LSTM
from src.solvers.dualsmc_nolstm import DualSMC
from src.environments.env import *
from plotting.floor import *
from configs.environments.floor import *
from configs.solver.dualsmc import *


def dualsmc(model, experiment_id, foldername, train):
    ################################
    # Create variables necessary for tracking diagnostics
    ################################
    step_list = []
    dist_list = []
    time_list_step = []
    time_list_episode = []
    reward_list_step = []
    reward_list_episode = []
    episode_Z_loss = []
    episode_P_loss = []
    episode_T_loss = []
    episode_q1_loss = []
    episode_q2_loss = []
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
            step_T_loss = []
            step_q1_loss = []
            step_q2_loss = []
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
            if SMCP_MODE == 'topk':
                weight_init, idx = torch.topk(par_weight, NUM_PAR_SMC_INIT)
                idx = idx.detach().cpu().numpy()
            elif SMCP_MODE == 'samp':
                idx = torch.multinomial(normalized_weights, NUM_PAR_SMC_INIT, replacement=True).detach().cpu().numpy()
                weight_init = par_weight[idx]
            weight_init = torch.softmax(weight_init, -1).unsqueeze(1).repeat(1, NUM_PAR_SMC)  # [M, N]
            states_init = par_states[idx]  # [K, C] -> [M, C]
            states_init_ = np.reshape(states_init, (1, NUM_PAR_SMC_INIT, 1, DIM_STATE))  # [1, M, 1, C]
            smc_states = np.tile(states_init_, (HORIZON, 1, NUM_PAR_SMC, 1))  # [1, M, 1, C] -> [T, M, N, C]
            smc_action = np.zeros((HORIZON, NUM_PAR_SMC, DIM_ACTION))  # [T, N, dim_a]
            smc_weight = torch.log(torch.ones((NUM_PAR_SMC)).to(device) * (1.0 / float(NUM_PAR_SMC)))  # [N]
            mean_state = np.reshape(mean_state, (1, 1, DIM_STATE))  # [1, 1, C]
            smc_mean_state = np.tile(mean_state, (HORIZON, NUM_PAR_SMC, 1))  # [T, N, C]
            prev_q = 0

            for i in range(HORIZON):
                curr_smc_state = torch.FloatTensor(smc_states[i]).to(device)  # [M, N, C]
                action, log_prob = model.policy.get_action(
                    torch.FloatTensor(smc_mean_state[i]).to(device),  # [N, C]
                    torch.transpose(curr_smc_state, 0, 1).contiguous().view(NUM_PAR_SMC, -1))  # [N, M * C]
                action_tile = action.unsqueeze(0).repeat(NUM_PAR_SMC_INIT, 1, 1).view(-1, DIM_ACTION)

                next_smc_state = model.dynamic_net.t_model(
                    torch.FloatTensor(smc_states[i]).to(device).view(-1, DIM_STATE), action_tile * STEP_RANGE)
                next_smc_state[:, 0] = torch.clamp(next_smc_state[:, 0], 0, 2)
                next_smc_state[:, 1] = torch.clamp(next_smc_state[:, 1], 0, 1)
                next_smc_state = next_smc_state.view(NUM_PAR_SMC_INIT, NUM_PAR_SMC, DIM_STATE)

                mean_par = model.dynamic_net.t_model(
                    torch.FloatTensor(smc_mean_state[i]).to(device), action * STEP_RANGE)
                mean_par[:, 0] = torch.clamp(mean_par[:, 0], 0, 2)
                mean_par[:, 1] = torch.clamp(mean_par[:, 1], 0, 1)

                if i < HORIZON - 1:
                    smc_action[i] = action.detach().cpu().numpy()
                    smc_states[i + 1] = next_smc_state.detach().cpu().numpy()
                    smc_mean_state[i + 1] = mean_par.detach().cpu().numpy()

                q = model.get_q(curr_smc_state.view(-1, DIM_STATE), action_tile).view(NUM_PAR_SMC_INIT, -1)
                advantage = q - prev_q - log_prob.unsqueeze(0).repeat(NUM_PAR_SMC_INIT, 1)  # [M, N]
                advantage = torch.sum(weight_init * advantage, 0).squeeze()  # [N]
                smc_weight += advantage
                prev_q = q
                normalized_smc_weight = F.softmax(smc_weight, -1)  # [N]

                if SMCP_RESAMPLE and (i % SMCP_RESAMPLE_STEP == 1):
                    idx = torch.multinomial(normalized_smc_weight, NUM_PAR_SMC, replacement=True).detach().cpu().numpy()
                    smc_action = smc_action[:, idx, :]
                    smc_states = smc_states[:, :, idx, :]
                    smc_mean_state = smc_mean_state[:, idx, :]
                    smc_weight = torch.log(torch.ones((NUM_PAR_SMC)).to(device) * (1.0 / float(NUM_PAR_SMC)))
                    normalized_smc_weight = F.softmax(smc_weight, -1)  # [N]

            smc_xy = np.reshape(smc_states[:, :, :, :2], (-1, NUM_PAR_SMC_INIT * NUM_PAR_SMC, 2))

            if SMCP_RESAMPLE and (HORIZON % SMCP_RESAMPLE_STEP == 0):
                n = np.random.randint(NUM_PAR_SMC, size=1)[0]
            else:
                n = Categorical(normalized_smc_weight).sample().detach().cpu().item()
            action = smc_action[0, n, :]
            toc = time.perf_counter()
            #######################################
            if step % PF_RESAMPLE_STEP == 0:
                if PP_EXIST:
                    idx = torch.multinomial(normalized_weights, NUM_PAR_PF - num_par_propose,
                                            replacement=True).detach().cpu().numpy()
                    resample_state = par_states[idx]
                    proposal_state = model.pp_net(torch.FloatTensor(curr_obs).unsqueeze(0).to(device), num_par_propose)
                    proposal_state[:, 0] = torch.clamp(proposal_state[:, 0], 0, 2)
                    proposal_state[:, 1] = torch.clamp(proposal_state[:, 1], 0, 1)
                    proposal_state = proposal_state.detach().cpu().numpy()
                    par_states = np.concatenate((resample_state, proposal_state), 0)
                else:
                    idx = torch.multinomial(normalized_weights, NUM_PAR_PF, replacement=True).detach().cpu().numpy()
                    par_states = par_states[idx]

                par_weight = torch.log(torch.ones((NUM_PAR_PF)).to(device) * (1.0 / float(NUM_PAR_PF)))
                normalized_weights = torch.softmax(par_weight, -1)

            mean_state = model.get_mean_state(par_states, normalized_weights).detach().cpu().numpy()
            filter_rmse = math.sqrt(pow(mean_state[0] - curr_state[0], 2) + pow(mean_state[1] - curr_state[1], 2))
            rmse_per_step[step] += filter_rmse
            filter_dist += filter_rmse

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
            reward = env.step(action * STEP_RANGE)
            next_state = env.state
            next_obs = env.get_observation()

            #######################################
            if train:
                model.replay_buffer.push(curr_state, action, reward, next_state, env.done, curr_obs,
                                         curr_s, mean_state, states_init)
                if len(model.replay_buffer) > BATCH_SIZE:
                    p_loss, t_loss, z_loss, q1_loss, q2_loss = model.soft_q_update()

                    step_P_loss.append(p_loss.item())
                    step_T_loss.append(t_loss.item())
                    step_Z_loss.append(z_loss.item())
                    step_q1_loss.append(q1_loss.item())
                    step_q2_loss.append(q2_loss.item())
            #######################################
            # Transition Model
            par_states = model.dynamic_net.t_model(torch.FloatTensor(par_states).to(device),
                                                   torch.FloatTensor(action * STEP_RANGE).to(device))
            par_states[:, 0] = torch.clamp(par_states[:, 0], 0, 2)
            par_states[:, 1] = torch.clamp(par_states[:, 1], 0, 1)
            par_states = par_states.detach().cpu().numpy()

            #######################################
            curr_state = next_state
            curr_obs = next_obs
            hidden = 0
            cell = 0
            trajectory.append(next_state)
            if env.done:
                break

            time_this_step = toc - tic
            time_list_step.append(time_this_step)
            reward_list_step.append(reward)

        # TODO TRY BOTH MEAN AND NOT FOR THE LOSS PLOTS
        # Get the average loss of each model for this episode if we are training
        if train:
            episode_P_loss.append(mean(step_P_loss))
            episode_T_loss.append(mean(step_T_loss))
            episode_Z_loss.append(mean(step_Z_loss))
            episode_q1_loss.append(mean(step_q1_loss))
            episode_q2_loss.append(mean(step_q2_loss))

        # Get the sum of the episode time
        tot_time = sum(time_list_step)
        # Get the running average of the time
        avg_time_this_episode = tot_time / len(time_list_step)
        time_list_episode.append(avg_time_this_episode)

        # Get the average reward this episode
        tot_reward = sum(reward_list_step)
        avg_reward_this_episode = tot_reward / len(reward_list_step)
        reward_list_episode.append(avg_reward_this_episode)
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
            episode_list = [episode_P_loss, episode_T_loss, episode_Z_loss, episode_q1_loss, episode_q2_loss]
            st2 = img_path + "/" + str(episode)
            if train:
                visualize_learning(st2, episode_list, time_list_episode, step_list, reward_list_episode, episode)
            else:
                visualize_learning(st2, None, time_list_episode, step_list, reward_list_episode, episode)
            st = img_path + "/" + str(episode) + "-trj" + FIG_FORMAT
            print("plotting ... save to %s" % st)
            plot_maze(figure_name=st, states=np.array(trajectory))

            if episode >= SUMMARY_ITER:
                total_iter = SUMMARY_ITER
            else:
                total_iter = episode
            reach = np.array(step_list) < (MAX_STEPS - 1)
            num_reach = sum(reach)
            step_reach = step_list * reach

            interaction = 'Episode %s: mean/stdev steps taken = %s / %s, reward = %s / %s, avg_plan_time = %s / %s, avg_dist = %s / %s' % (
                episode, mean(step_list), stdev(step_list), mean(reward_list_episode), stdev(reward_list_episode),
                mean(time_list_episode), stdev(time_list_episode), mean(dist_list), stdev(dist_list))
            file2.write('\n{}'.format(interaction))
            file2.flush()

        # Repeat the above code block for writing to the text file every episode instead of every 10
        if episode >= SUMMARY_ITER:
            total_iter = SUMMARY_ITER
        else:
            total_iter = episode

        interaction = 'Episode %s: steps = %s, reward = %s, avg_plan_time = %s, avg_dist = %s' % (
            episode, step, avg_reward_this_episode, avg_time_this_episode, sum(dist_list) / total_iter)
        file1.write('\n{}'.format(interaction))
        file1.flush()

    rmse_per_step = rmse_per_step / MAX_EPISODES
    print(rmse_per_step)
    file1.close()
    file2.close()


def dualsmc_driver(load_path=None, pre_training=True, save_pretrained_model=True,
                   end_to_end=True, save_model=True, test=True):
    # This block of code creates the folders for plots
    settings = "dualsmc_nolstm_indv"
    foldername = settings + get_datetime()
    os.mkdir(foldername)
    experiment_id = "dualsmc" + get_datetime()
    save_path = CKPT + experiment_id

    # Create a model and environment object
    model = DualSMC()
    env = Environment()

    # Let the user load in a previous model
    if load_path is not None:
        model.load_model(load_path)

    # This is where we need to perform individual training (if the user wants).
    # The process for this is to (1) create a observation and state batch.
    # Then (2) only train Z and P in an adversarial manner using the custom
    # soft_q_update function

    if pre_training:
        measure_loss = []
        proposer_loss = []
        # First we'll do train individually for 64 batches
        for batch in range(5000):
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
            measure_loss.append(Z_loss)
            proposer_loss.append(P_loss)

        if save_pretrained_model:
            model_path = save_path + "pre_trained_" + str(pre_training)
            model.save_model(model_path)
            print("saving pre-trained model to %s" % model_path)

    if end_to_end:
        train = True
        # After pretraining move into the end to end training
        dualsmc(model, experiment_id, foldername, train)

    if save_model:
        # Save the model
        model.save_model(save_path)
        print("saving model to %s" % save_path)

    if test:
        train = False
        dualsmc(model, experiment_id, foldername, train)


if __name__ == "__main__":
    if MODEL_NAME == 'dualsmc':
        dualsmc_driver()

