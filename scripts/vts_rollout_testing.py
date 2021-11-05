import numpy as np
from torch import std

from src.environments.stanford import *

from configs.environments.stanford import *
from configs.solver.vts_lightdark import *

from plotting.stanford import *

vlp = VTS_LightDark_Params()
sep = Stanford_Environment_Params()


env = StanfordEnvironment() 

discount = vlp.discount

def generate_particles_normal(num_rand_states, mean, std):
    ss = mean + np.random.normal(0, std, size=(num_rand_states, sep.dim_state))
    ws = np.random.rand(num_rand_states)
    ws = ws/sum(ws)
    return ss, ws

def generate_particles_custom():
    ss = np.array([[1., 1.3],
                [7., 0.45],
                [3.5, 0.75],
                [5.6, 0.1],
                [2.2, 0.3],
                [6., 1.5],
                [4.8, 0.4]])
    ws = np.array([0.5, 0.15, 0.01, 0.01, 0.03, 0.2, 0.1])

    return ss, ws


num_rand_states = vlp.num_par_pftdpw
stdev = 0.1
#mean_states = [[1.5, 0.25], [2., 0.5], [2.5, 1.], [3., 1.25], [3.5, 0.75], [4., 0.5]]
#mean_states = [[2., 0.25], [2., 0.4], [2., 0.5], [2., 0.6], [2., 0.75], [2., 0.85], [2., 1.], [2., 1.25], [2., 1.35]]
mean_states = [[2., 0.45], [2., 0.47], [2., 0.5], [2., 0.53], [2., 0.55]]

#ss, ws = generate_particles_normal(num_rand_states=50, mean=np.array([3., 1.2]), std=0.1)
ss, ws = generate_particles_normal(num_rand_states, mean=np.array([0., 0.]), std=stdev)  

for i in range(len(mean_states)):
        mean_state = mean_states[i]
        ss_copy = ss + mean_state

        num_tests = 200
        rewards = []
        for _ in range(num_tests):
                index = np.random.choice(len(ws), 1, p = ws)
                s = ss_copy[index][0]
                reward, vec = env.rollout(s, ss_copy, ws, discount)
                rewards.append(reward)
                #print("Chosen state:", s, "Weight of that state:", ws[index], "Reward:", reward, "Vector:", vec, "Weights:", ws, "\n")

        print("Mean reward:", np.mean(np.array(rewards)))

        fig1, ax1 = plt.subplots()
        plt.hist(ws, bins=np.linspace(min(ws), max(ws), 15), label='Weights')
        plt.legend()
        plt.savefig("rollout_belief_weights" + str(i))
        plt.close()

        fig1, ax1 = plt.subplots()
        plt.hist(rewards, bins=np.linspace(min(rewards), max(rewards), 15), label='Rewards')
        plt.legend()
        plt.savefig("rollout_rewards" + str(i))
        plt.close()

        # Plotting the states and vector to the goal in the environment
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
        figure_name = "rollout_analysis" + str(i)
        vts_rollout_analysis(xlim, ylim, goal, [trap1, trap2], dark, figure_name, s, ss_copy, ws, vec)
        print("Distance:", env.distance_to_goal(mean_state)[1])

