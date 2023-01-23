# author: @sdeglurkar, @jatucker4, @michaelhlim

import matplotlib

from src.environments.stanford import StanfordEnvironment
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

from configs.environments.stanford import *

sep = Stanford_Environment_Params()


def plot_maze(xlim, ylim, goal, trap, test_trap, dark, figure_name='default', states=None, highres=False):
    plt.figure(figure_name)
    ax = plt.axes()

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    from matplotlib.patches import Rectangle
    # goal: [start_x, start_y, width, height]
    ax.add_patch(Rectangle((goal[0], goal[1]), goal[2], goal[3], facecolor='green'))
    trap1 = trap[0]
    trap2 = trap[1]
    # trap i: [start_x, start_y, width, height]
    ax.add_patch(Rectangle((trap1[0], trap1[1]), trap1[2], trap1[3], facecolor='orange'))
    ax.add_patch(Rectangle((trap2[0], trap2[1]), trap2[2], trap2[3], facecolor='orange'))
    # dark region
    ax.add_patch(Rectangle((dark[0], dark[1]), dark[2], dark[3], facecolor='black', alpha=0.15))
    # additional wall
    ax.add_patch(Rectangle((0, trap1[1]), 
        trap1[0], trap1[3], facecolor='black', alpha=0.2))
    ax.add_patch(Rectangle((trap1[0]+trap1[2], trap1[1]), 
        goal[0]-(trap1[0]+trap1[2]), trap1[3], facecolor='black', alpha=0.2))
    ax.add_patch(Rectangle((goal[0]+goal[2], trap1[1]), 
        trap2[0]-(goal[0]+goal[2]), trap1[3], facecolor='black', alpha=0.2))
    ax.add_patch(Rectangle((trap2[0]+trap2[2], trap1[1]), 
        xlim[1]-(trap2[0]+trap2[2]), trap1[3], facecolor='black', alpha=0.2))
    
    if test_trap is not None:
         # trap i: [start_x, start_y, width, height]
         test_trap1 = test_trap[0]
         test_trap2 = test_trap[1]
         ax.add_patch(Rectangle((test_trap1[0], test_trap1[1]), test_trap1[2], test_trap1[3], facecolor='orange'))
         ax.add_patch(Rectangle((test_trap2[0], test_trap2[1]), test_trap2[2], test_trap2[3], facecolor='orange'))

    if type(states) is np.ndarray:
        xy = states[:,:2]
        x, y = zip(*xy)
        ax.plot(x[0], y[0], 'bo')
        # Iterate through x and y with a colormap
        colorvec = np.linspace(0, 1, len(x))
        viridis = cm.get_cmap('YlGnBu', len(colorvec))
        for i in range(len(x)):
            if i == 0:
                continue
            plt.plot(x[i], y[i], color=viridis(colorvec[i]), marker='o')

    ax.set_aspect('equal')
    if highres:
        plt.savefig(figure_name, bbox_inches='tight', dpi=1000)
    else:
        plt.savefig(figure_name, bbox_inches='tight')
    plt.close()


def plot_par(xlim, ylim, goal, trap, test_trap, dark, figure_name='default', true_state=None, mean_state=None, 
            pf_state=None, pf_weights=None, pp_state=None, smc_traj=None, highres=False):
    plt.figure(figure_name)
    ax = plt.axes()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    from matplotlib.patches import Rectangle
    # goal: [start_x, start_y, width, height]
    ax.add_patch(Rectangle((goal[0], goal[1]), goal[2], goal[3], facecolor='green'))
    trap1 = trap[0]
    trap2 = trap[1]
    # trap i: [start_x, start_y, width, height]
    ax.add_patch(Rectangle((trap1[0], trap1[1]), trap1[2], trap1[3], facecolor='orange'))
    ax.add_patch(Rectangle((trap2[0], trap2[1]), trap2[2], trap2[3], facecolor='orange'))
    # dark region
    ax.add_patch(Rectangle((dark[0], dark[1]), dark[2], dark[3], facecolor='black', alpha=0.15))
    # additional wall
    ax.add_patch(Rectangle((0, trap1[1]), 
        trap1[0], trap1[3], facecolor='black', alpha=0.2))
    ax.add_patch(Rectangle((trap1[0]+trap1[2], trap1[1]), 
        goal[0]-(trap1[0]+trap1[2]), trap1[3], facecolor='black', alpha=0.2))
    ax.add_patch(Rectangle((goal[0]+goal[2], trap1[1]), 
        trap2[0]-(goal[0]+goal[2]), trap1[3], facecolor='black', alpha=0.2))
    ax.add_patch(Rectangle((trap2[0]+trap2[2], trap1[1]), 
        xlim[1]-(trap2[0]+trap2[2]), trap1[3], facecolor='black', alpha=0.2))

    
    if test_trap is not None:
         # trap i: [start_x, start_y, width, height]
         test_trap1 = test_trap[0]
         test_trap2 = test_trap[1]
         ax.add_patch(Rectangle((test_trap1[0], test_trap1[1]), test_trap1[2], test_trap1[3], facecolor='orange'))
         ax.add_patch(Rectangle((test_trap2[0], test_trap2[1]), test_trap2[2], test_trap2[3], facecolor='orange'))

    ax.plot(mean_state[0], mean_state[1], 'ko')
    ax.plot(true_state[0], true_state[1], 'ro')

    xy = pf_state[:, :2]
    x, y = zip(*xy)
    heuristic_alpha_mult = 10
    for j in range(len(x)):
        ax.plot(x[j], y[j], 'gx', alpha=min(pf_weights[j]*heuristic_alpha_mult, 1.0))

    if pp_state is not None:
        xy = pp_state[:, :2]
        x, y = zip(*xy)
        ax.plot(x, y, 'bx')
    
    # planning trajectories
    if smc_traj is not None:
        if smc_traj.any():
            if len(smc_traj.shape) == 2:
                ax.plot(smc_traj[:, 0], smc_traj[:, 1], lw=3, color=(0.5, 0.5, 1.0))  # RGB
            else:
                num_par_smc = smc_traj.shape[1]
                for k in range(num_par_smc):
                    points = smc_traj[:, k, :]
                    ax.plot(*points.T, lw=3, color=(0.5, 0.5, 0.5))  # RGB
                if len(points) > 1:
                    plt.arrow(points[-2, 0], points[-2, 1], 
                            points[-1, 0] - points[-2, 0],
                            points[-1, 1] - points[-2, 1], 
                            head_width = 0.12, width = 0.013, color=(0.5, 0.5, 0.5))

    ax.set_aspect('equal')

    if highres:
        plt.savefig(figure_name, bbox_inches='tight', dpi=1000)
    else:
        plt.savefig(figure_name, bbox_inches='tight')
    plt.close()


def plot_crosses(data, color, ax):
    xy = data
    x, y = zip(*xy)
    for j in range(len(x)):
        ax.plot(x[j], y[j], color)
def plot_crosses_with_alphas(data, color, alphas, ax):
    xy = data
    x, y = zip(*xy)
    if max(alphas) < 0.5:
        heuristic_alpha = 0.5
    else:
        heuristic_alpha = 0
    for j in range(len(x)):
        ax.plot(x[j], y[j], color, alpha=min(alphas[j] + heuristic_alpha, 1.0))

def vts_pretraining_analysis(xlim, ylim, goal, trap, dark, figure_names, 
            true_state, true_orientation, random_states, likelihoods_list,
            #likelihoods, likelihoods_blur, likelihoods_gen,
            proposed_states):
    '''
    Plot on the environment the state and proposed states for when the input is the true image
    versus when it is a generated image.
    '''

    ## Plot all the proposed particles
    plt.figure(figure_names[0])
    ax = plt.axes()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Plot goal, traps, walls, and other features of the environment
    from matplotlib.patches import Rectangle
    # goal: [start_x, start_y, width, height]
    ax.add_patch(Rectangle((goal[0], goal[1]), goal[2], goal[3], facecolor='green'))
    trap1 = trap[0]
    trap2 = trap[1]
    # trap i: [start_x, start_y, width, height]
    ax.add_patch(Rectangle((trap1[0], trap1[1]), trap1[2], trap1[3], facecolor='orange'))
    ax.add_patch(Rectangle((trap2[0], trap2[1]), trap2[2], trap2[3], facecolor='orange'))
    # dark region
    ax.add_patch(Rectangle((dark[0], dark[1]), dark[2], dark[3], facecolor='black', alpha=0.15))
    # additional wall
    ax.add_patch(Rectangle((0, trap1[1]), 
        trap1[0], trap1[3], facecolor='black', alpha=0.2))
    ax.add_patch(Rectangle((trap1[0]+trap1[2], trap1[1]), 
        goal[0]-(trap1[0]+trap1[2]), trap1[3], facecolor='black', alpha=0.2))
    ax.add_patch(Rectangle((goal[0]+goal[2], trap1[1]), 
        trap2[0]-(goal[0]+goal[2]), trap1[3], facecolor='black', alpha=0.2))
    ax.add_patch(Rectangle((trap2[0]+trap2[2], trap1[1]), 
        xlim[1]-(trap2[0]+trap2[2]), trap1[3], facecolor='black', alpha=0.2))

    # Plot the true state and orientation
    ax.plot(true_state[0], true_state[1], 'ro')
    ax.quiver(true_state[0], true_state[1], np.cos(true_orientation), np.sin(true_orientation))

    # Plot the proposed states for the true image as input
    plot_crosses(proposed_states[:, :2], 'gx', ax)
    
    # Plot the proposed states for the blurred image as input
    # plot_crosses(proposed_states_blur[:, :2], 'yx', ax)

    # Plot the proposed states for the generated image as input
    # plot_crosses(proposed_states_gen[:, :2], 'bx', ax) 
    
    ax.set_aspect('equal')

    plt.savefig(figure_names[0])
    plt.close()


    def plot_dummys(likelihoods, fig_name):
        plt.figure(fig_name)
        ax = plt.axes()
        # Plot the true state and orientation
        ax.plot(true_state[0], true_state[1], 'ro')
        ax.quiver(true_state[0], true_state[1], np.cos(true_orientation), np.sin(true_orientation))
        plot_crosses_with_alphas(random_states[:, :2], 'bx', likelihoods, ax)
        ax.set_aspect('equal')
        plt.savefig(fig_name)
        plt.close()

    
    num_bins = 25
    for i in range(len(likelihoods_list)):
        lik = likelihoods_list[i]
        fig_name = figure_names[i+1]
        plot_dummys(lik, fig_name)
        plt.figure()
        plt.hist(lik, bins=np.linspace(min(lik), max(lik), num_bins))
        plt.savefig(fig_name + "_logits")
        plt.close()

    # ## Plot the dummy particles for the true image as input
    # plot_dummys(likelihoods, figure_names[1])

    # ## Plot the dummy particles for the blurred image as input
    # plot_dummys(likelihoods_blur, figure_names[2])

    # ## Plot the dummy particles for the generated image as input
    # plot_dummys(likelihoods_gen, figure_names[3])

        

    # plt.hist(likelihoods, bins=np.linspace(min(likelihoods), max(likelihoods), 25), label='dummy_logits')
    # plt.hist(likelihoods_gen, bins=np.linspace(min(likelihoods_gen), max(likelihoods_gen), 25), label='dummy_logits_gen')
    # ax1.set_xlim(min(min(likelihoods), min(likelihoods_gen)), max(max(likelihoods), max(likelihoods_gen)))
    # plt.legend()
    # plt.savefig("likelihoods")
    # plt.close()
    

def vts_pretraining_analysis_old(xlim, ylim, goal, trap, dark, figure_name='pretraining_analysis', 
            true_state=None, true_orientation=None, proposed_states=None, likelihoods=None,
            proposed_states_gen=None, likelihoods_gen = None):
    '''
    Plot on the environment the state and proposed states for when the input is the true image
    versus when it is a generated image.
    '''

    plt.figure(figure_name)
    ax = plt.axes()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Plot goal, traps, walls, and other features of the environment
    from matplotlib.patches import Rectangle
    # goal: [start_x, start_y, width, height]
    ax.add_patch(Rectangle((goal[0], goal[1]), goal[2], goal[3], facecolor='green'))
    trap1 = trap[0]
    trap2 = trap[1]
    # trap i: [start_x, start_y, width, height]
    ax.add_patch(Rectangle((trap1[0], trap1[1]), trap1[2], trap1[3], facecolor='orange'))
    ax.add_patch(Rectangle((trap2[0], trap2[1]), trap2[2], trap2[3], facecolor='orange'))
    # dark region
    ax.add_patch(Rectangle((dark[0], dark[1]), dark[2], dark[3], facecolor='black', alpha=0.15))
    # additional wall
    ax.add_patch(Rectangle((0, trap1[1]), 
        trap1[0], trap1[3], facecolor='black', alpha=0.2))
    ax.add_patch(Rectangle((trap1[0]+trap1[2], trap1[1]), 
        goal[0]-(trap1[0]+trap1[2]), trap1[3], facecolor='black', alpha=0.2))
    ax.add_patch(Rectangle((goal[0]+goal[2], trap1[1]), 
        trap2[0]-(goal[0]+goal[2]), trap1[3], facecolor='black', alpha=0.2))
    ax.add_patch(Rectangle((trap2[0]+trap2[2], trap1[1]), 
        xlim[1]-(trap2[0]+trap2[2]), trap1[3], facecolor='black', alpha=0.2))


    # Plot the true state and orientation
    ax.plot(true_state[0], true_state[1], 'ro')
    ax.quiver(true_state[0], true_state[1], np.cos(true_orientation), np.sin(true_orientation))

    # Plot the proposed states with alpha value based on their likelihoods (given by Z)
    # These are proposed states and likelihoods for the true image as input
    xy = proposed_states[:, :2]
    x, y = zip(*xy)
    heuristic_alpha_mult = 1
    for j in range(len(x)):
        ax.plot(x[j], y[j], 'gx', alpha=min(likelihoods[j]*heuristic_alpha_mult, 1.0))

    plt.savefig(figure_name)
    plt.close()


def plot_crosses(data, color, ax):
    xy = data
    x, y = zip(*xy)
    for j in range(len(x)):
        ax.plot(x[j], y[j], color)
def plot_crosses_with_alphas(data, color, alphas, ax):
    xy = data
    x, y = zip(*xy)
    if max(alphas) < 0.5:
        heuristic_alpha = 0.5
    else:
        heuristic_alpha = 0
    for j in range(len(x)):
        ax.plot(x[j], y[j], color, alpha=min(alphas[j] + heuristic_alpha, 1.0))

def vts_pretraining_analysis(xlim, ylim, goal, trap, dark, figure_names, 
            true_state, true_orientation, random_states, likelihoods_list,
            proposed_states):
    '''
    Plot on the environment the state and proposed states for when the input is the true image
    versus when it is a generated image.
    '''

    ## Plot all the proposed particles
    plt.figure(figure_names[0])
    ax = plt.axes()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Plot goal, traps, walls, and other features of the environment
    from matplotlib.patches import Rectangle
    # goal: [start_x, start_y, width, height]
    ax.add_patch(Rectangle((goal[0], goal[1]), goal[2], goal[3], facecolor='green'))
    trap1 = trap[0]
    trap2 = trap[1]
    # trap i: [start_x, start_y, width, height]
    ax.add_patch(Rectangle((trap1[0], trap1[1]), trap1[2], trap1[3], facecolor='orange'))
    ax.add_patch(Rectangle((trap2[0], trap2[1]), trap2[2], trap2[3], facecolor='orange'))
    # dark region
    ax.add_patch(Rectangle((dark[0], dark[1]), dark[2], dark[3], facecolor='black', alpha=0.15))
    # additional wall
    ax.add_patch(Rectangle((0, trap1[1]), 
        trap1[0], trap1[3], facecolor='black', alpha=0.2))
    ax.add_patch(Rectangle((trap1[0]+trap1[2], trap1[1]), 
        goal[0]-(trap1[0]+trap1[2]), trap1[3], facecolor='black', alpha=0.2))
    ax.add_patch(Rectangle((goal[0]+goal[2], trap1[1]), 
        trap2[0]-(goal[0]+goal[2]), trap1[3], facecolor='black', alpha=0.2))
    ax.add_patch(Rectangle((trap2[0]+trap2[2], trap1[1]), 
        xlim[1]-(trap2[0]+trap2[2]), trap1[3], facecolor='black', alpha=0.2))

    # Plot the true state and orientation
    ax.plot(true_state[0], true_state[1], 'ro')
    ax.quiver(true_state[0], true_state[1], np.cos(true_orientation), np.sin(true_orientation))

    # Plot the proposed states for the true image as input
    plot_crosses(proposed_states[:, :2], 'gx', ax)
    
    # Plot the proposed states for the blurred image as input
    # plot_crosses(proposed_states_blur[:, :2], 'yx', ax)

    # Plot the proposed states for the generated image as input
    # plot_crosses(proposed_states_gen[:, :2], 'bx', ax) 
    
    ax.set_aspect('equal')

    plt.savefig(figure_names[0])
    plt.close()


    def plot_dummys(likelihoods, fig_name):
        plt.figure(fig_name)
        ax = plt.axes()
        # Plot the true state and orientation
        ax.plot(true_state[0], true_state[1], 'ro')
        ax.quiver(true_state[0], true_state[1], np.cos(true_orientation), np.sin(true_orientation))
        plot_crosses_with_alphas(random_states[:, :2], 'bx', likelihoods, ax)
        ax.set_aspect('equal')
        plt.savefig(fig_name)
        plt.close()

    
    num_bins = 25
    for i in range(len(likelihoods_list)):
        lik = likelihoods_list[i]
        fig_name = figure_names[i+1]
        plot_dummys(lik, fig_name)
        plt.figure()
        plt.hist(lik, bins=np.linspace(min(lik), max(lik), num_bins))
        plt.savefig(fig_name + "_logits")
        plt.close()

    # ## Plot the dummy particles for the true image as input
    # plot_dummys(likelihoods, figure_names[1])

    # ## Plot the dummy particles for the blurred image as input
    # plot_dummys(likelihoods_blur, figure_names[2])

    # ## Plot the dummy particles for the generated image as input
    # plot_dummys(likelihoods_gen, figure_names[3])

        

    # plt.hist(likelihoods, bins=np.linspace(min(likelihoods), max(likelihoods), 25), label='dummy_logits')
    # plt.hist(likelihoods_gen, bins=np.linspace(min(likelihoods_gen), max(likelihoods_gen), 25), label='dummy_logits_gen')
    # ax1.set_xlim(min(min(likelihoods), min(likelihoods_gen)), max(max(likelihoods), max(likelihoods_gen)))
    # plt.legend()
    # plt.savefig("likelihoods")
    # plt.close()

    

def vts_pretraining_analysis_old(xlim, ylim, goal, trap, dark, figure_name='pretraining_analysis', 
            true_state=None, true_orientation=None, proposed_states=None, likelihoods=None,
            proposed_states_gen=None, likelihoods_gen = None):
    '''
    Plot on the environment the state and proposed states for when the input is the true image
    versus when it is a generated image.
    '''

    plt.figure(figure_name)
    ax = plt.axes()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Plot goal, traps, walls, and other features of the environment
    from matplotlib.patches import Rectangle
    # goal: [start_x, start_y, width, height]
    ax.add_patch(Rectangle((goal[0], goal[1]), goal[2], goal[3], facecolor='green'))
    trap1 = trap[0]
    trap2 = trap[1]
    # trap i: [start_x, start_y, width, height]
    ax.add_patch(Rectangle((trap1[0], trap1[1]), trap1[2], trap1[3], facecolor='orange'))
    ax.add_patch(Rectangle((trap2[0], trap2[1]), trap2[2], trap2[3], facecolor='orange'))
    # dark region
    ax.add_patch(Rectangle((dark[0], dark[1]), dark[2], dark[3], facecolor='black', alpha=0.15))
    # additional wall
    ax.add_patch(Rectangle((0, trap1[1]), 
        trap1[0], trap1[3], facecolor='black', alpha=0.2))
    ax.add_patch(Rectangle((trap1[0]+trap1[2], trap1[1]), 
        goal[0]-(trap1[0]+trap1[2]), trap1[3], facecolor='black', alpha=0.2))
    ax.add_patch(Rectangle((goal[0]+goal[2], trap1[1]), 
        trap2[0]-(goal[0]+goal[2]), trap1[3], facecolor='black', alpha=0.2))
    ax.add_patch(Rectangle((trap2[0]+trap2[2], trap1[1]), 
        xlim[1]-(trap2[0]+trap2[2]), trap1[3], facecolor='black', alpha=0.2))


    # Plot the true state and orientation
    ax.plot(true_state[0], true_state[1], 'ro')
    ax.quiver(true_state[0], true_state[1], np.cos(true_orientation), np.sin(true_orientation))

    # Plot the proposed states with alpha value based on their likelihoods (given by Z)
    # These are proposed states and likelihoods for the true image as input
    xy = proposed_states[:, :2]
    x, y = zip(*xy)
    heuristic_alpha_mult = 1
    for j in range(len(x)):
        ax.plot(x[j], y[j], 'gx', alpha=min(likelihoods[j]*heuristic_alpha_mult, 1.0))

    # Plot the proposed states with alpha value based on their likelihoods (given by Z)
    # These are proposed states and likelihoods for the generated image as input
    xy = proposed_states_gen[:, :2]
    x, y = zip(*xy)
    heuristic_alpha_mult = 1
    for j in range(len(x)):
        ax.plot(x[j], y[j], 'bx', alpha=min(likelihoods_gen[j]*heuristic_alpha_mult, 1.0))

    ax.set_aspect('equal')

    plt.savefig(figure_name)
    plt.close()


    fig1, ax1 = plt.subplots()
    plt.hist(likelihoods, bins=np.linspace(min(likelihoods), max(likelihoods), 25), label='likelihoods')
    plt.hist(likelihoods_gen, bins=np.linspace(min(likelihoods_gen), max(likelihoods_gen), 25), label='likelihoods_gen')
    ax1.set_xlim(min(min(likelihoods), min(likelihoods_gen)), max(max(likelihoods), max(likelihoods_gen)))
    plt.legend()
    plt.savefig("likelihoods")
    plt.close()
    

def vts_rollout_analysis(xlim, ylim, goal, trap, dark, figure_name, s, ss, ws, vec):
    
    plt.figure(figure_name)
    ax = plt.axes()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Plot goal, traps, walls, and other features of the environment
    from matplotlib.patches import Rectangle
    # goal: [start_x, start_y, width, height]
    ax.add_patch(Rectangle((goal[0], goal[1]), goal[2], goal[3], facecolor='green'))
    trap1 = trap[0]
    trap2 = trap[1]
    # trap i: [start_x, start_y, width, height]
    ax.add_patch(Rectangle((trap1[0], trap1[1]), trap1[2], trap1[3], facecolor='orange'))
    ax.add_patch(Rectangle((trap2[0], trap2[1]), trap2[2], trap2[3], facecolor='orange'))
    # dark region
    ax.add_patch(Rectangle((dark[0], dark[1]), dark[2], dark[3], facecolor='black', alpha=0.15))
    # additional wall
    ax.add_patch(Rectangle((0, trap1[1]), 
        trap1[0], trap1[3], facecolor='black', alpha=0.2))
    ax.add_patch(Rectangle((trap1[0]+trap1[2], trap1[1]), 
        goal[0]-(trap1[0]+trap1[2]), trap1[3], facecolor='black', alpha=0.2))
    ax.add_patch(Rectangle((goal[0]+goal[2], trap1[1]), 
        trap2[0]-(goal[0]+goal[2]), trap1[3], facecolor='black', alpha=0.2))
    ax.add_patch(Rectangle((trap2[0]+trap2[2], trap1[1]), 
        xlim[1]-(trap2[0]+trap2[2]), trap1[3], facecolor='black', alpha=0.2))
    
    ax.plot(s[0], s[1], 'ro')
    ax.arrow(s[0], s[1], vec[0], vec[1], head_width=0.1, head_length=0.2)
    plot_crosses_with_alphas(ss, 'bx', ws, ax)

    ax.set_aspect('equal')

    plt.savefig(figure_name)
    plt.close()


def visualize_learning(figure_name, episode_loss_list, time_list, step_list, reward_list, num_episodes, name_list):
    '''
    :param figure_name: path to save the figure in
    :param episode_loss_list: List of lists that contains the loss for each network. None type if testing
    :param time_list: List that contains the mean time to plan for each episode
    :param step_list: List that contains the number of steps for each episode
    :param reward_list: List that contains the reward for each episode
    :param num_episodes: The number of episodes trained to this point
    :return: Plots of loss, mean training time, number of steps, and reward vs episode
    '''
    ##################
    # Loss plot block
    ##################
    if episode_loss_list is not None:
        for i in range(len(name_list)):
            num_records = len(episode_loss_list[i])
            episode_number_list = np.linspace(1, num_records, num_records)

            path_str = figure_name + name_list[i] + sep.fig_format
            plt.figure(path_str)
            ax = plt.axes()
            ax.plot(episode_number_list, episode_loss_list[i])
            plt.xlabel("Episodes")
            plt.ylabel("Average Loss")
            plt.savefig(path_str)
            plt.close()

    ##################
    # Time plot block
    ##################

    episode_number_list = np.linspace(1, len(time_list), len(time_list))
    path_str = figure_name + "time_plot" + sep.fig_format 
    plt.figure(path_str)
    ax = plt.axes()
    ax.plot(episode_number_list, time_list)
    plt.xlabel("Episodes")
    plt.ylabel("Average Time to Plan")
    plt.savefig(path_str)
    plt.close()

    ##################
    # Step plot block
    ##################

    episode_number_list = np.linspace(1, len(step_list), len(step_list))
    path_str = figure_name + "step_plot" + sep.fig_format
    plt.figure(path_str)
    ax = plt.axes()
    ax.plot(episode_number_list, step_list)
    plt.xlabel("Episodes")
    plt.ylabel("Number of Steps")
    plt.savefig(path_str)
    plt.close()

    ##################
    # Reward plot block
    ##################

    episode_number_list = np.linspace(1, len(reward_list), len(reward_list))
    path_str = figure_name + "reward_plot" + sep.fig_format
    plt.figure(path_str)
    ax = plt.axes()
    ax.plot(episode_number_list, reward_list)
    plt.xlabel("Episodes")
    plt.ylabel("Average Reward")
    plt.savefig(path_str)
    plt.close()
