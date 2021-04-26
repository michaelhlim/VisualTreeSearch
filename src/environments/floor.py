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
        obs += np.random.normal(0, 0.01, DIM_OBS)
        return obs

    def step(self, action):
        self.done = False
        curr_state = self.state
        next_state = curr_state + action

        cond = (curr_state[1] <= 0.5)
        target = cond * self.target1 + (1 - cond) * self.target2

        next_dist = l2_distance(next_state, target)
        cond_hit = detect_collison(curr_state, next_state)

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
        # Check if a given state/state tensor is a terminal state
        cond = (self.state[1] <= 0.5)
        target = cond * self.target1 + (1 - cond) * self.target2

        return all(l2_distance_np(s, target) <= END_RANGE)

    def action_sample(self):
        # Gives back a uniformly sampled random action
        rnd = int(random.random()*9)

        # No blank move
        while rnd == 5:
            rnd = int(random.random()*9)

        action = STEP_RANGE * np.array([(rnd % 3) - 1, (rnd // 3) - 1])

        return action

    def reward(self, s):
        # Check if a given state/state tensor is a terminal state and give corresponding reward
        cond = (self.state[1] <= 0.5)
        target = cond * self.target1 + (1 - cond) * self.target2
        false_target = cond * self.false_target1 + (1 - cond) * self.false_target2

        dist = l2_distance_np(s, target)
        false_dist = l2_distance_np(s, false_target)

        cond_true = (dist <= END_RANGE)
        cond_false = (false_dist <= END_RANGE)

        reward = EPI_REWARD * cond_true - EPI_REWARD * cond_false + STEP_REWARD * np.logical_not((cond_true | cond_false))

        return reward

    def rollout(self, s):
        # Roll out from state s, calculating the naive distance & reward to the goal       