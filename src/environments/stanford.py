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
        self.state = [24.5, 23.0, np.pi]
        self.target = [28.0, 23.0]
        self.xrange = [24, 32.5]
        self.yrange = [23, 24.5]
        self.thetas = np.array([0.0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4])

        path = '/home/sampada_deglurkar/VisualTreeSearch/temp/'
        os.mkdir(path)
        img_path, traversible, dx_m = self.get_observation(path=path)
        os.remove(img_path)
        os.rmdir(path)
        self.traversible = traversible
        self.dx = dx_m
        self.map_origin = [0, 0]


    def get_observation(self, state=None, path=None):
        if state == None:
            state_arr = np.array([self.state])
        else:
            state_arr = np.array([state])

        if path == None:
            path = '/home/sampada_deglurkar/VisualTreeSearch/images/'

        return generate_observation(state_arr, path)
    

    def read_observation(self, img_path, normalize):
        obs = cv2.imread(img_path, cv2.IMREAD_COLOR)
        obs = obs[:,:,::-1]   ## CV2 works in BGR space instead of RGB!! So dumb! --- now obs is in RGB
        if normalize:
            obs = (obs - obs.mean())/obs.std()  # "Normalization" -- TODO

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
        map_state = self.point_to_map(np.array(state[:2]))
        map_value = self.traversible[map_state[1], map_state[0]]
        return map_value == 0


    def step(self, action):
        self.done = False
        curr_state = self.state
        next_state = curr_state + action

        next_dist = l2_distance(next_state[:2], self.target)
        cond_hit = self.detect_collision(next_state)

        if next_dist <= sep.end_range:
            self.state = next_state
            self.done = True
        elif cond_hit == False:
            self.state = next_state
        reward = sep.epi_reward * self.done

        return reward


    def is_terminal(self, s):
        # Check if a given state tensor is a terminal state
        s = s[:, :2]
        targets = np.tile(self.target, (s.shape[0], 1))
        
        true_dist = l2_distance_np(s, targets)
        
        return all(true_dist <= sep.end_range)
    

    def make_batch(self, batch_size):
        path = '/home/sampada_deglurkar/VisualTreeSearch/temp/'
        os.mkdir(path)

        states_batch = []
        obs_batch = []
        for i in range(batch_size):
            theta = self.thetas[np.random.randint(len(self.thetas))]
            x = np.random.rand() * (self.xrange[1] - self.xrange[0]) + self.xrange[0]
            y = np.random.rand() * (self.yrange[1] - self.yrange[0]) + self.yrange[0]
            state = [x, y, theta]
            
            img_path, _, _ = self.get_observation(state, path)
            obs = self.read_observation(img_path, normalize=True) 

            par_vec_x = np.random.normal(state[0], sep.obs_std, dlp.num_par_pf)
            par_vec_y = np.random.normal(state[1], sep.obs_std, dlp.num_par_pf)
            par_vec_theta = self.thetas[np.random.randint(len(self.thetas), size=(dlp.num_par_pf))]
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