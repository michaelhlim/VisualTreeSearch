import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

from configs.environments.stanford import *

sep = Stanford_Environment_Params()


def plot_maze(xlim, ylim, goal, trap, figure_name='default', states=None):
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

    # goals
    # cir = plt.Circle(goal, 0.07, color='orange')
    # ax.add_artist(cir)

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
    plt.savefig(figure_name)
    plt.close()


def plot_par(xlim, ylim, goal, trap, figure_name='default', true_state=None, mean_state=None, pf_state=None,
             pp_state=None, smc_traj=None):
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

    # goals
    # cir = plt.Circle(goal, 0.07, color='orange')
    # ax.add_artist(cir)

    # planning trajectories
    if smc_traj.any():
        num_par_smc = smc_traj.shape[1]
        for k in range(num_par_smc):
            points = smc_traj[:, k, :]
            ax.plot(*points.T, lw=1, color=(0.5, 0.5, 0.5))  # RGB

    ax.plot(mean_state[0], mean_state[1], 'ko')
    ax.plot(true_state[0], true_state[1], 'ro')
    #ax.quiver(true_state[0], true_state[1], np.cos(true_state[2]), np.sin(true_state[2]))

    xy = pf_state[:, :2]
    x, y = zip(*xy)
    ax.plot(x, y, 'gx')

    xy = pp_state[:, :2]
    x, y = zip(*xy)
    ax.plot(x, y, 'bx')

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
