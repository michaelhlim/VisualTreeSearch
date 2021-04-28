# author: @wangyunbo
from utils.utils import *
from configs.environments.floor import *
from src.environments.abstract import AbstractEnvironment
import random
import numpy as np

class Environment(AbstractEnvironment):
    def __init__(self):
        self.done = False
        self.state = np.random.rand(2)
        self.state[0] = self.state[0] * 0.4 + 0.8
        self.state[1] = self.state[1] * 0.3 + 0.1 + np.random.randint(2) * 0.5
        self.target1 = np.array([2, 0.25])
        self.target2 = np.array([0, 0.75])
        self.false_target1 = np.array([0, 0.25])
        self.false_target2 = np.array([2, 0.75])
        self.walls_x = [[1.2, 0.9, 1], [0.4, 0.4, 0.5], [1.2, 0.5, 0.6], [0.4, 0.0, 0.1]]
        self.walls_y = [[0.5, 0, 2]]

    def get_observation(self):
        x = self.state[0]
        y = self.state[1]
        obs_x1 = x
        obs_y1 = y
        obs_x2 = 1 - x
        obs_y2 = 1 - y
        num_wall_x = len(self.walls_x)
        num_wall_y = len(self.walls_y)
        for i in range(num_wall_x):
            wx = self.walls_x[i][0]
            wy1 = self.walls_x[i][1]
            wy2 = self.walls_x[i][2]
            if y > wy1 and y < wy2 and x > wx:
                dist_x1 = x - wx
                obs_x1 = min(obs_x1, dist_x1)
            if y > wy1 and y < wy2 and x < wx:
                dist_x2 = wx - x
                obs_x2 = min(obs_x2, dist_x2)
        for i in range(num_wall_y):
            wy = self.walls_y[i][0]
            wx1 = self.walls_y[i][1]
            wx2 = self.walls_y[i][2]
            if x > wx1 and x < wx2 and y > wy:
                dist_y1 = y - wy
                obs_y1 = min(obs_y1, dist_y1)
            if x > wx1 and x < wx2 and y < wy:
                dist_y2 = wy - y
                obs_y2 = min(obs_y2, dist_y2)
        obs = np.array([obs_x1, obs_y1, obs_x2, obs_y2])
        obs += np.random.normal(0, OBS_STD, DIM_OBS)
        return obs


    def get_observation_batch(self, x, y):
        obs_x1 = x
        obs_y1 = y
        obs_x2 = 1 - x
        obs_y2 = 1 - y
        num_wall_x = len(self.walls_x)
        num_wall_y = len(self.walls_y)
        for i in range(num_wall_x):
            wx = self.walls_x[i][0]
            wy1 = self.walls_x[i][1]
            wy2 = self.walls_x[i][2]
            if y > wy1 and y < wy2 and x > wx:
                dist_x1 = x - wx
                obs_x1 = min(obs_x1, dist_x1)
            if y > wy1 and y < wy2 and x < wx:
                dist_x2 = wx - x
                obs_x2 = min(obs_x2, dist_x2)
        for i in range(num_wall_y):
            wy = self.walls_y[i][0]
            wx1 = self.walls_y[i][1]
            wx2 = self.walls_y[i][2]
            if x > wx1 and x < wx2 and y > wy:
                dist_y1 = y - wy
                obs_y1 = min(obs_y1, dist_y1)
            if x > wx1 and x < wx2 and y < wy:
                dist_y2 = wy - y
                obs_y2 = min(obs_y2, dist_y2)
        obs = np.array([obs_x1, obs_y1, obs_x2, obs_y2])
        obs += np.random.normal(0, OBS_STD, DIM_OBS)
        return obs


    def step(self, action):
        self.done = False
        curr_state = self.state
        next_state = curr_state + action

        cond = (curr_state[1] <= 0.5)
        target = cond * self.target1 + (1 - cond) * self.target2

        next_dist = l2_distance(next_state, target)
        cond_hit = detect_collision(curr_state, next_state)

        if next_dist <= END_RANGE:
            self.state = next_state
            self.done = True
        elif cond_hit == False:
            self.state = next_state
        reward = EPI_REWARD * self.done

        false_target = cond * self.false_target1 + (1 - cond) * self.false_target2
        curr_false_dist = l2_distance(curr_state, false_target)
        next_false_dist = l2_distance(next_state, false_target)
        cond_false = (curr_false_dist >= END_RANGE) * (next_false_dist < END_RANGE)
        reward -= EPI_REWARD * cond_false
        return reward

    def is_terminal(self, s):
        # Check if a given state tensor is a terminal state
        cond = (s[:, 1] <= 0.5)
        true_targets = np.zeros(np.shape(s))
        true_targets[cond, :] = self.target1
        true_targets[~cond, :] = self.target2
        
        true_dist = l2_distance_np(s, true_targets)
        
        return all(true_dist <= END_RANGE)

    def action_sample(self):
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
        # transition each state in state tensor s with actions in action/action tensor a
        next_state = np.copy(s)
        next_weights = np.copy(w)
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
        reward += np.sum(w[goal_achieved]) * EPI_REWARD

        # If false goal reached
        reward -= np.sum(w[false_goal]) * EPI_REWARD

        # Else
        reward += np.sum(w[normal_step]) * STEP_REWARD

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

    def make_batch(self, batch_size):
        states_batch = []
        obs_batch = []
        for i in range(batch_size):
            state = np.random.rand(2)
            state[0] = state[0] * 2
            state[1] = state[1]

            obs = self.get_observation_batch(state[0], state[1])

            par_vec_x = np.random.normal(state[0], OBS_STD, NUM_PAR_PF)
            par_vec_y = np.random.normal(state[1], OBS_STD, NUM_PAR_PF)
            states_batch.append(state)
            obs_batch.append(obs)
            middle_var = np.stack((par_vec_x, par_vec_y), 1)

            if i == 0:
                par_batch = middle_var
            else:
                par_batch = np.concatenate((par_batch, middle_var), 0)

        states_batch = np.array(states_batch)
        obs_batch = np.array(obs_batch)

        return states_batch, obs_batch, par_batch


    def make_batch_single_state(self, batch_size):
        # Used for testing observation generative model

        state = np.random.rand(2)
        state[0] = state[0] * 2
        state[1] = state[1]

        obs_batch = np.array([self.get_observation_batch(state[0], state[1]) for _ in range(batch_size)])

        return state, obs_batch


    def make_batch_wall(self, batch_size, wall):
        # Make a whole batch from just one state in one of the 4 walls

        state = np.random.rand(2)
        state[0] = state[0] * 2
        if wall == 0.1:
            state[1] = state[1] * 0.1
        elif wall == 0.4:
            state[1] = state[1] * 0.1 + 0.4
        elif wall == 0.6:
            state[1] = state[1] * 0.1 + 0.5
        elif wall == 0.9:
            state[1] = state[1] * 0.1 + 0.9

        obs_batch = np.array([self.get_observation_batch(state[0], state[1]) for _ in range(batch_size)])
        states_batch = np.tile(state, (batch_size, 1))

        return states_batch, obs_batch


    def make_batch_multiple_walls(self, batch_size, walls_arr):
        # Make a batch from just wall states, mixture from all 4 walls

        states_batch = []
        obs_batch = []
        for i in range(batch_size):
            state = np.random.rand(2)
            state[0] = state[0] * 2

            index = np.random.randint(len(walls_arr))
            wall = walls_arr[index]

            if wall == 0.1:
                state[1] = state[1] * 0.1
            elif wall == 0.4:
                state[1] = state[1] * 0.1 + 0.4
            elif wall == 0.6:
                state[1] = state[1] * 0.1 + 0.5
            elif wall == 0.9:
                state[1] = state[1] * 0.1 + 0.9

            obs = self.get_observation_batch(state[0], state[1])

            par_vec_x = np.random.normal(state[0], OBS_STD, NUM_PAR_PF)
            par_vec_y = np.random.normal(state[1], OBS_STD, NUM_PAR_PF)
            states_batch.append(state)
            obs_batch.append(obs)
            middle_var = np.stack((par_vec_x, par_vec_y), 1)

            if i == 0:
                par_batch = middle_var
            else:
                par_batch = np.concatenate((par_batch, middle_var), 0)

        states_batch = np.array(states_batch)
        obs_batch = np.array(obs_batch)

        return states_batch, obs_batch, par_batch
