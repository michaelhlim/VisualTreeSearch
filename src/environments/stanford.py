# author: @wangyunbo
import cv2
import numpy as np
import os
import random

from src.environments.abstract import AbstractEnvironment
from utils.utils import *

from configs.environments.stanford import *
from configs.solver.dualsmc_lightdark import *

dlp = DualSMC_LightDark_Params()
sep = Stanford_Environment_Params()

from examples.examples import *


class StanfordEnvironment(AbstractEnvironment):
    def __init__(self):
        self.done = False
        self.xrange = [24, 32.5]
        self.yrange = [23, 24.5]
        self.thetas = [0.0, 2*np.pi]
        self.trap_x = self.xrange  # but excluding the target
        self.trap_y = [23, 23.25]
        self.target_x = [27.5, 28.5]
        self.target_y = [23, 23.25]
        self.init_strip_x = self.xrange 
        self.init_strip_y = [23.25, 23.5]
        self.state = np.random.rand(sep.dim_state)
        self.orientation = np.random.rand()
        self.state[0] = self.state[0] * (self.xrange[1] - self.xrange[0]) + self.xrange[0]
        self.state[1] = self.state[1] * (23.5 - 23.25) + 23.25
        #self.state[2] = self.state[2] * (self.thetas[1] - self.thetas[0]) + self.thetas[0]
        self.orientation = self.orientation * (self.thetas[1] - self.thetas[0]) + self.thetas[0]
        self.dark_line = (self.yrange[0] + self.yrange[1])/2

        path = os.getcwd() + '/temp/'
        os.mkdir(path)
        img_path, traversible, dx_m = self.get_observation(path=path)
        os.remove(img_path)
        os.rmdir(path)
        self.traversible = traversible
        self.dx = dx_m
        self.map_origin = [0, 0]


    def get_observation(self, state=None, path=None):
        if state == None:
            state_arr = np.array([[self.state[0], self.state[1], self.orientation]])
        else:
            state_arr = np.array([state])

        if path == None:
            path = path = os.getcwd() + '/images/' 

        img_path, traversible, dx_m = generate_observation(state_arr, path)
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        out = image
        # cv2.imwrite(img_path[:-4] + "_ORIGINAL.png", out)

        if state_arr[0][1] <= self.dark_line: 
            # Dark observation - add salt & pepper noise
            
            row,col,ch = image.shape
            s_vs_p = 0.5
            amount = 0.15
            out = np.copy(image)
            # Salt mode
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt))
                    for i in image.shape]
            out[coords] = 255

            # # Salt mode
            # num_salt = np.ceil(amount * image.size * s_vs_p)
            # coords = [np.repeat(np.random.randint(0, i - 1, int(num_salt)), 3)
            #         for i in image.shape]
            # coords[2] = np.tile([0, 1, 2], int(num_salt))
            # out[coords] = 255

            # Pepper mode
            num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper))
                    for i in image.shape]
            out[coords] = 0

            # # Pepper mode
            # num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
            # coords = [np.repeat(np.random.randint(0, i - 1, int(num_pepper)), 3)
            #         for i in image.shape]
            # coords[2] = np.tile([0, 1, 2], int(num_salt))
            # out[coords] = 0

        
        cv2.imwrite(img_path, out)

        # obs_orig = self.read_observation(img_path[:-4] + "_ORIGINAL.png", "_DENORMORIG.png", normalize=True)
        # obs_noisy = self.read_observation(img_path, "_DENORMNOISE.png", normalize=True)

        return img_path, traversible, dx_m
    

    def read_observation(self, img_path, normalize):
        obs = cv2.imread(img_path, cv2.IMREAD_COLOR)
        obs = obs[:,:,::-1]   ## CV2 works in BGR space instead of RGB!! So dumb! --- now obs is in RGB
        if normalize:
            obs = (obs - obs.mean())/obs.std()  # "Normalization" -- TODO
        
        # obs = obs * 100 + (255./2)
        # obs = obs[:, :, ::-1]
        # cv2.imwrite(img_path[:-4] + str, obs)

        os.remove(img_path)

        return obs
        
    
    def point_to_map(self, pos_2, cast_to_int=True):
        """
        Convert pos_2 in real world coordinates
        to a point on the map.
        """
        map_pos_2 = pos_2/self.dx - self.map_origin
        if cast_to_int:
            map_pos_2 = map_pos_2.astype(np.int32)
        return map_pos_2


    def detect_collision(self, state):
        # Returns true if you collided
        # Map value 0 means there's an obstacle there

        # Don't hit rectangle boundaries
        if state[0] < self.xrange[0] or state[0] > self.xrange[1]:
            return True 
        if state[1] < self.yrange[0] or state[1] > self.yrange[1]:
            return True

        map_state = self.point_to_map(np.array(state[:2]))
        map_value = self.traversible[map_state[1], map_state[0]]
        return map_value == 0

    
    def in_trap(self, state):
        # Returns true if in trap
        trap = (state[0] >= self.trap_x[0] and state[0] <= self.trap_x[1]) and \
                (state[1] >= self.trap_y[0] and state[1] <= self.trap_y[1]) 
        
        return trap and not self.in_goal(state)
    
    def in_goal(self, state):
        # Returns true if in goal
        goal = (state[0] >= self.target_x[0] and state[0] <= self.target_x[1]) and \
                (state[1] >= self.target_y[0] and state[1] <= self.target_y[1])
        return goal


    def step(self, action):
        self.done = False
        curr_state = self.state
        new_theta = action[0] * np.pi + np.pi
        vector = np.array([np.cos(new_theta), np.sin(new_theta)]) * sep.velocity  # Go in the direction the new theta is
        next_state = curr_state + vector
    
        cond_hit = self.detect_collision(next_state)

        if self.in_goal(next_state):
            self.state = next_state
            self.orientation = new_theta
            self.done = True
        elif cond_hit == False:
            self.state = next_state
            self.orientation = new_theta
        reward = sep.epi_reward * self.done

        cond_false = self.in_trap(next_state)
        reward -= sep.epi_reward * cond_false

        return reward


    def is_terminal(self, s):
        # Check if a given state tensor is a terminal state
        s = s[:, :2]
        targets = np.tile(self.target, (s.shape[0], 1))
        
        true_dist = l2_distance_np(s, targets)
        
        return all(true_dist <= sep.end_range)

    
    def make_pars(self, batch_size):
        thetas = np.random.rand(batch_size, 1) * (self.thetas[1] - self.thetas[0]) + self.thetas[0]
        xs = np.random.rand(batch_size, 1) * (self.xrange[1] - self.xrange[0]) + self.xrange[0]
        ys = np.random.rand(batch_size, 1) * (self.init_strip_y[1] - self.init_strip_y[0]) + self.init_strip_y[0]

        # thetas = np.random.rand(batch_size, 1) * (self.thetas[-1] - self.thetas[0]) + self.thetas[0]
        # xs = np.random.rand(batch_size, 1) * (self.xrange[1] - self.xrange[0]) + self.xrange[0]
        # ys = np.random.rand(batch_size, 1) * (self.yrange[1] - self.yrange[0]) + self.yrange[0]

        #par_batch = np.concatenate((xs, ys, thetas), 1)
        par_batch = np.concatenate((xs, ys), 1)
        
        return par_batch, thetas
    

    def make_batch(self, batch_size):
        return
        path = path = os.getcwd() + '/temp/' 
        os.mkdir(path)

        states_batch = []
        obs_batch = []
        for i in range(batch_size):
            # theta = self.thetas[np.random.randint(len(self.thetas))]
            theta = np.random.rand() * (self.thetas[-1] - self.thetas[0]) + self.thetas[0]
            x = np.random.rand() * (self.xrange[1] - self.xrange[0]) + self.xrange[0]
            y = np.random.rand() * (self.yrange[1] - self.yrange[0]) + self.yrange[0]
            state = [x, y, theta]
            
            img_path, _, _ = self.get_observation(state, path)
            obs = self.read_observation(img_path, normalize=True) 

            par_vec_x = np.random.normal(state[0], sep.obs_std, dlp.num_par_pf)
            par_vec_y = np.random.normal(state[1], sep.obs_std, dlp.num_par_pf)
            par_vec_theta = np.random.rand(dlp.num_par_pf) * (self.thetas[-1] - self.thetas[0]) + self.thetas[0]
            # par_vec_theta = self.thetas[np.random.randint(len(self.thetas), size=(dlp.num_par_pf))]
            states_batch.append(state)
            obs_batch.append(obs)
            middle_var = np.stack((par_vec_x, par_vec_y, par_vec_theta), 1)

            if i == 0:
                par_batch = middle_var
            else:
                par_batch = np.concatenate((par_batch, middle_var), 0)

            os.remove(img_path)

        os.rmdir(path)

        states_batch = np.array(states_batch)
        obs_batch = np.array(obs_batch)

        return states_batch, obs_batch, par_batch


    def action_sample(self):
        return
        # Gives back a uniformly sampled random action
        rnd = int(random.random()*9)

        # No blank move
        while rnd == 4:
            rnd = int(random.random()*9)

        action = STEP_RANGE * np.array([(rnd % 3) - 1, (rnd // 3) - 1])

        # # just generate completely random
        # action_x = STEP_RANGE * (2 * random.random() - 1)
        # action_y = STEP_RANGE * (2 * random.random() - 1)
        # action = np.array([action_x, action_y])

        return action

    def transition(self, s, w, a):
        return
        # transition each state in state tensor s with actions in action/action tensor a
        next_state = np.copy(s)
        if w is not None:
            weights = np.copy(w)
            next_weights = np.copy(w)
        else:
            # Dummy weight
            weights = np.ones(np.shape(s)[0])
            next_weights = np.ones(np.shape(s)[0])
        sp = s + a
        reward = 0.0

        # Determine targets
        cond = (s[:, 1] <= 0.5)
        true_targets = np.zeros(np.shape(s))
        true_targets[cond, :] = self.target1
        true_targets[~cond, :] = self.target2
        false_targets = np.zeros(np.shape(s))
        false_targets[cond, :] = self.false_target1
        false_targets[~cond, :] = self.false_target2

        # Calculate distances
        next_true_dist = l2_distance_np(sp, true_targets)
        curr_false_dist = l2_distance_np(s, false_targets)
        next_false_dist = l2_distance_np(sp, false_targets)

        # Check collision & goal
        cond_hit = detect_collision(s, sp)
        goal_achieved = (next_true_dist <= END_RANGE)
        step_ok = (~cond_hit | goal_achieved)

        # Check false goal
        false_goal = (curr_false_dist > END_RANGE) * (next_false_dist <= END_RANGE)
        normal_step = ~(goal_achieved | false_goal)

        # If goal reached
        next_state[step_ok, :] = sp[step_ok, :]
        next_weights[goal_achieved] = 0.0
        reward += np.sum(weights[goal_achieved]) * EPI_REWARD

        # If false goal reached
        reward -= np.sum(weights[false_goal]) * EPI_REWARD

        # Else
        reward += np.sum(weights[normal_step]) * STEP_REWARD

        # Is the transition terminal?
        is_terminal = all(goal_achieved)

        if is_terminal:
            # Dummy weight
            next_weights = np.array([1/len(next_weights)] * len(next_weights))  
        else:
            # Reweight
            next_weights = next_weights / np.sum(next_weights)
                
        return next_state, next_weights, reward, is_terminal

    def rollout(self, s, ss, ws):
        return
        # Roll out from state s, calculating the naive distance & reward to the goal, then check how it would do for all other particles
        cond = (s[1] <= 0.5)
        target = cond * self.target1 + (1 - cond) * self.target2
        dist = np.sqrt(l2_distance(s, target))
        steps = int(np.floor(dist/STEP_RANGE))
        
        # Going stepping number of times will provide the following intermediate rewards (geometric series result)
        gamma = np.power(DISCOUNT, steps)
        reward = STEP_REWARD * (1.0 - gamma)/(1.0 - DISCOUNT)

        # Basically check if the targets are same (1) or different (-1)
        cond_all = -2 * (cond ^ (ss[:, 1] <= 0.5)) + 1
        reward += gamma * np.dot(cond_all, ws) * EPI_REWARD

        return reward


# stan = StanfordEnvironment()
# stan.make_batch(3)