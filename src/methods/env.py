import numpy as np

DIM_OBS = 4

EPI_REWARD = 100
END_RANGE = 0.005
STEP_RANGE = 0.05


def detect_collison(curr_state, next_state):
    if len(curr_state.shape) == 1:
        cx = curr_state[0]
        cy = curr_state[1]
        nx = next_state[0]
        ny = next_state[1]
    elif len(curr_state.shape) == 2:
        cx = curr_state[:, 0]
        cy = curr_state[:, 1]
        nx = next_state[:, 0]
        ny = next_state[:, 1]
    cond_hit = (nx < 0) + (nx > 2) + (ny < 0) + (ny > 1)  # hit the surrounding walls
    cond_hit += (cy <= 0.5) * (ny > 0.5)  # cross the middle wall
    cond_hit += (cy >= 0.5) * (ny < 0.5)  # cross the middle wall
    cond_hit += (cx <= 1.2) * (nx > 1.2) * ((cy > 0.9) * (cy < 1) + (ny > 0.9) * (ny < 1))
    cond_hit += (cx >= 1.2) * (nx < 1.2) * ((cy > 0.9) * (cy < 1) + (ny > 0.9) * (ny < 1))
    cond_hit += (cx <= 0.4) * (nx > 0.4) * ((cy > 0.4) * (cy < 0.5) + (ny > 0.4) * (ny < 0.5))
    cond_hit += (cx >= 0.4) * (nx < 0.4) * ((cy > 0.4) * (cy < 0.5) + (ny > 0.4) * (ny < 0.5))
    cond_hit += (cx <= 1.2) * (nx > 1.2) * ((cy > 0.5) * (cy < 0.6) + (ny > 0.5) * (ny < 0.6))
    cond_hit += (cx >= 1.2) * (nx < 1.2) * ((cy > 0.5) * (cy < 0.6) + (ny > 0.5) * (ny < 0.6))
    cond_hit += (cx <= 0.4) * (nx > 0.4) * ((cy > 0.0) * (cy < 0.1) + (ny > 0.0) * (ny < 0.1))
    cond_hit += (cx >= 0.4) * (nx < 0.4) * ((cy > 0.0) * (cy < 0.1) + (ny > 0.0) * (ny < 0.1))
    return cond_hit

def l2_distance(state, goal):
    if len(state.shape) == 1:
        dist = np.power((state[0] - goal[0]), 2) + np.power((state[1] - goal[1]), 2) + const
    elif len(state.shape) == 2:
        dist = (state[:, 0] - goal[:, 0]).pow(2) + (state[:, 1] - goal[:, 1]).pow(2) + const
    elif len(state.shape) == 3:
        dist = (state[:, :, 0] - goal[:, :, 0]).pow(2) + (state[:, :, 1] - goal[:, :, 1]).pow(2) + const
    return dist

class Environment(object):
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
        #self.walls_x = []
        #self.walls_y = []

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
        ######################## I CHANGED THIS #########################
        #obs += np.random.normal(0, 0.01, DIM_OBS)
        obs += np.random.normal(0, 0.1, DIM_OBS)
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