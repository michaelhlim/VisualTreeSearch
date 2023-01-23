# author: @sdeglurkar, @jatucker4, @michaelhlim

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

def vts(model, observation_generator, experiment_id, train, model_path):
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
    if train:
        print("=========================\nTraining for iterations:", MAX_EPISODES_TRAIN)
        experiment_str = experiment_id + "/train"
        num_loops = MAX_EPISODES_TRAIN
    else:
        print("=========================\nTesting for iterations:", MAX_EPISODES_TEST)
        experiment_str = experiment_id + "/test"
        num_loops = MAX_EPISODES_TEST

    save_path = CKPT + experiment_str
    img_path = IMG + experiment_str
    check_path(save_path)
    check_path(img_path)
    str123 = experiment_id + ".txt"
    str1234 = experiment_id + "every_10_eps" + ".txt"
    file1 = open(save_path + "/" + str123, 'w+')
    file2 = open(save_path + "/" + str1234, 'w+')

    # Begin main dualSMC loop
    for episode in range(num_loops):
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
            # action = pft_planner.solve(par_states, normalized_weights.detach().cpu().numpy())
            action, future_actions = pft_planner.solve_viz(par_states, normalized_weights.detach().cpu().numpy())

            # Resampling
            if step % PF_RESAMPLE_STEP == 0:
                if PP_EXIST:
                    # For de-localization problem: don't propose so many particles
                    if PP_DECAY:
                        # Exponential decay in proposed particles according to decay rate
                        decayed_num_propose = int(num_par_propose * DECAY_RATE**step)
                        if decayed_num_propose <= 0:
                            proposal_state = None
                        else:
                            proposal_state = model.pp_net(torch.FloatTensor(
                                curr_obs).unsqueeze(0).to(device), decayed_num_propose)

                        # Particles not from the proposer - sample with replacement from existing set
                        idx = torch.multinomial(normalized_weights, NUM_PAR_PF - decayed_num_propose,
                                            replacement=True).detach().cpu().numpy()
                    
                    # For de-localization problem: don't propose so many particles
                    elif PP_STD:
                        # Only propose particles if existing particle stdev is above threshold
                        std_norm = np.linalg.norm(np.std(par_states, axis=0))
                        if std_norm >= STD_THRES:
                            # Propose in an amount proportional to the stdev, capped at num_par_propose
                            std_num_propose = int(min(num_par_propose, STD_ALPHA*std_norm))
                            proposal_state = model.pp_net(torch.FloatTensor(
                                curr_obs).unsqueeze(0).to(device), std_num_propose)
                        else: # Otherwise don't propose particles 
                            proposal_state = None
                            std_num_propose = 0

                        # Particles not from the proposer - sample with replacement from existing set
                        idx = torch.multinomial(normalized_weights, NUM_PAR_PF - std_num_propose,
                                            replacement=True).detach().cpu().numpy() 
                    
                    # For de-localization problem: don't propose so many particles
                    elif PP_EFFECTIVE:
                        # Only propose particles if the "effective number of particles" is below 
                        # a threshold
                        # Formula comes from Gustafsson 2010
                        effective_num_particles = NUM_PAR_PF/(1 + NUM_PAR_PF**2 * 
                                                    torch.var(normalized_weights.detach()))
                        if effective_num_particles < EFFECTIVE_THRES:
                            # Propose in an amount inversely proportional to the effective num particles, 
                            # capped at num_par_propose
                            effective_num_propose = int(min(num_par_propose, 
                                                    EFFECTIVE_ALPHA / effective_num_particles))
                            proposal_state = model.pp_net(torch.FloatTensor(
                                curr_obs).unsqueeze(0).to(device), effective_num_propose)
                        else: # Otherwise don't propose particles 
                            proposal_state = None
                            effective_num_propose = 0

                        # Particles not from the proposer - sample with replacement from existing set
                        idx = torch.multinomial(normalized_weights, NUM_PAR_PF - effective_num_propose,
                                        replacement=True).detach().cpu().numpy() 

                    # Otherwise propose fixed amount of particles
                    else:
                        proposal_state = model.pp_net(torch.FloatTensor(
                            curr_obs).unsqueeze(0).to(device), num_par_propose)

                        # Particles not from the proposer - sample with replacement from existing set
                        idx = torch.multinomial(normalized_weights, NUM_PAR_PF - num_par_propose,
                                            replacement=True).detach().cpu().numpy()

                    # Particles not from the proposer - sample with replacement from existing set
                    resample_state = par_states[idx]
                    
                    if proposal_state is not None:
                        # Keep proposals within environment bounds
                        proposal_state[:, 0] = torch.clamp(proposal_state[:, 0], 0, 2)
                        proposal_state[:, 1] = torch.clamp(proposal_state[:, 1], 0, 1)
                        proposal_state = proposal_state.detach().cpu().numpy()

                        par_states = np.concatenate(
                            (resample_state, proposal_state), 0)
                    else:
                        par_states = resample_state
                
                # No proposer
                else:  
                    idx = torch.multinomial(
                        normalized_weights, NUM_PAR_PF, replacement=True).detach().cpu().numpy()
                    par_states = par_states[idx]

                par_weight = torch.log(torch.ones((NUM_PAR_PF)).to(
                    device) * (1.0 / float(NUM_PAR_PF)))
                normalized_weights = torch.softmax(par_weight, -1)

            mean_state = model.get_mean_state(
                par_states, normalized_weights).detach().cpu().numpy()
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
                    check_path(img_path + "/traj/")
                    st = img_path + "/traj/" + str(episode) + "-" + file_name + "-trj" + FIG_FORMAT
                    plot_maze(figure_name=st, states=np.array(trajectory))
                    state_iterate = mean_state.reshape((1, 2))
                    planned_traj = state_iterate
                    for action in future_actions:
                        state_iterate, _, _, _ = env.transition(state_iterate, None, action)
                        planned_traj = np.vstack([planned_traj, state_iterate])
                    planned_traj = np.array(planned_traj)
                    planned_traj = planned_traj.reshape((planned_traj.shape[0], 1, planned_traj.shape[1]))
                    plot_par(frm_name, curr_state, mean_state, resample_state, proposal_state, planned_traj)

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
            par_states, _, _, _ = env.transition(par_states, normalized_weights.detach().cpu().numpy(), action)

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

            if env.done:
                break

        # TODO TRY BOTH MEAN AND NOT FOR THE LOSS PLOTS
        # Get the average loss of each model for this episode if we are training
        if train:
            if len(model.replay_buffer) > BATCH_SIZE:
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

        reach = np.array(step_list) < (MAX_STEPS - 1) # List of booleans for successful episodes

        if episode % SAVE_ITER == 0 and train:
            model.save_model(model_path + "/dpf_online")
            print("Saving online trained models to %s" % model_path)

        # Take only the statistics for successful episodes
        reach_steps = np.array(step_list)[reach] 
        reach_rewards = np.array(reward_list_episode)[reach]
        reach_times = np.array(time_list_episode)[reach]
        reach_dists = np.array(dist_list)[reach]

        if episode % DISPLAY_ITER == 0:
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
        st = img_path + "/traj/" + str(episode) + "-trj" + FIG_FORMAT
        plot_maze(figure_name=st, states=np.array(trajectory))

        # Repeat the above code block for writing to the text file every episode instead of every 10
        interaction = 'Episode %s: steps = %s, reward = %s, avg_plan_time = %s, avg_dist = %s' % (
            episode, step, tot_reward, avg_time_this_episode, filter_dist)
        print('\r{}'.format(interaction))
        file1.write('\n{}'.format(interaction))
        file1.flush()

    rmse_per_step = rmse_per_step / num_loops
    file1.close()
    file2.close()


def vts_driver(load_path=None, pre_training=True, save_pretrained_model=True,
                   end_to_end=True, save_online_model=True, test=True):
    torch.manual_seed(torch_seed)
    random.seed(random_seed)
    np.random.seed(np_random_seed)

    # This block of code creates the folders for plots
    experiment_id = "vts" + get_datetime()
    foldername = "data/" + experiment_id
    check_path(foldername)
    model_path = "nets/" + experiment_id
    check_path(model_path)

    check_path("data")
    check_path("nets")

    # Create a model and environment object
    model = VTS()
    env = Environment()

    observation_generator = ObservationGenerator()

    # Let the user load in a previous model
    if load_path is not None:
        cwd = os.getcwd()
        model.load_model(cwd + "/nets/" + load_path + "/dpf_pre_trained", observation_generator)

    # This is where we need to perform individual training (if the user wants).
    # The process for this is to (1) create a observation and state batch.
    # Then (2) only train Z and P in an adversarial manner using the custom
    # soft_q_update function

    if pre_training:
        tic = time.perf_counter()
        print("Pretraining observation density and particle proposer")
        print_freq = 100
        measure_loss = []
        proposer_loss = []
        # First we'll do train individually for 64 batches
        for batch in range(PRETRAIN):
            walls_arr = [0.1, 0.4, 0.6, 0.9, 0,
                         0, 0, 0]  # wall 0 means no wall
            state_batch, obs_batch, par_batch = env.make_batch_multiple_walls(64, walls_arr)

            # Train Z and P using the soft q update function
            Z_loss, P_loss = model.soft_q_update_individual(
                state_batch, obs_batch, par_batch)
            measure_loss.append(Z_loss.item())
            proposer_loss.append(P_loss.item())

            # Print loss and stuff for the last $print_freq batches
            if batch % print_freq == 0:
                print("Step: ", batch, ", Z loss: ", np.mean(
                    measure_loss[-print_freq:]), ", P loss: ", np.mean(proposer_loss[-print_freq:]))

        # Observation generative model
        training_time = observation_generator.pretrain(save_pretrained_model, model_path)

        if save_pretrained_model:
            model.save_model(model_path + "/dpf_pre_trained", observation_generator)
            print("Saving pre-trained Z, P models to %s" % model_path)
        
        toc = time.perf_counter()
        time_this_step = toc - tic
        print("Time elapsed for pre-training: ", time_this_step, "seconds.")
        print("Time elapsed for pre-training generator:", training_time, "seconds.")

        train_save_path = CKPT + experiment_id + "/train"
        check_path(train_save_path)
        train_file_str = experiment_id + "_train.txt"
        train_file = open(train_save_path + "/" + train_file_str, 'w+')
        training_time = 'Time elapsed for pre-training: %s, Time for pre-training generator: %s ' % \
                        (time_this_step, training_time)
        train_file.write('\n{}'.format(training_time))
        train_file.flush()

    if end_to_end:
        train = True
        # After pretraining move into the end to end training
        vts(model, observation_generator, experiment_id,
            train, model_path)

    if save_online_model:
        # Save the model
        model.save_model(model_path + "/dpf_online_trained")
        print("Saving online trained Z, P models to %s" % model_path)

    if test:
        train = False
        vts(model, observation_generator, experiment_id,
            train, model_path)


if __name__ == "__main__":
    if MODEL_NAME == 'dualsmc':
        if len(sys.argv) > 1 and sys.argv[1] == "--pretrain":
            # Just pre-training
            vts_driver(end_to_end=False, save_online_model=False, test=False)
        elif len(sys.argv) > 1 and sys.argv[1] == "--pretrain-test":
            # Pre-training immediately followed by testing
            vts_driver(end_to_end=False, save_online_model=False)
        elif len(sys.argv) > 1 and sys.argv[1] == "--test":
            # Just testing
            load_path = sys.argv[2]
            vts_driver(load_path=load_path, pre_training=False, end_to_end=False, save_online_model=False)
        elif len(sys.argv) > 1 and sys.argv[1] == "--online-train-test":
            # Right into online learning & testing
            vts_driver(load_path="test500k", pre_training=False)
        elif len(sys.argv) > 1:
            print("Unknown argument!")
        else:
            # Everything
            vts_driver()

