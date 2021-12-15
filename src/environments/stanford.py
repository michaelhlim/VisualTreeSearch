# author: @wangyunbo
import cv2
import glob
import numpy as np
import os
import pickle
import random

from src.environments.abstract import AbstractEnvironment
from utils.utils import *

from configs.environments.stanford import *
from configs.solver.dualsmc_lightdark import *

dlp = DualSMC_LightDark_Params()
sep = Stanford_Environment_Params()

from examples.examples import *  # generate_observation


class StanfordEnvironment(AbstractEnvironment):
    def __init__(self):
        self.done = False
        self.true_env_corner = [24.0, 23.0] 
        self.xrange = [0, 8.5]
        self.yrange = [0, 1.5]
        self.thetas = [0.0, 2*np.pi]
        self.trap_x = [[1.5, 2], [6.5, 7]] 
        self.trap_y = [0, 0.25]
        self.target_x = [4, 4.5] 
        self.target_y = [0, 0.25]
        
        # During test time - have an additional trap region (optional)
        self.test_trap = False
        self.test_trap_is_random = False
        self.test_trap_x = [[3, 3.5], [5, 5.5]] #[[1.5, 2], [6.5, 7]] #[3.5, 5]
        self.test_trap_y = [[0.5, 1], [0.5, 1]] #[0.5, 1] #[0.75, 1.25]

        self.init_strip_x = self.xrange 
        self.init_strip_y = [0.25, 0.5]
        self.state, self.orientation = self.initial_state()
        self.dark_line = (self.yrange[0] + self.yrange[1])/2
        self.dark_line_true = self.dark_line + self.true_env_corner[1]

        # Get the traversible
        try:
            traversible = pickle.load(open("traversible.p", "rb"))
            dx_m = 0.05
        except Exception:
            path = os.getcwd() + '/temp/'
            os.mkdir(path)
            _, _, traversible, dx_m = self.get_observation(normalize=False)  
            pickle.dump(traversible, open("traversible.p", "wb"))

        self.traversible = traversible
        self.dx = dx_m
        self.map_origin = [0, 0]

        # For making training batches
        self.training_data_path = sep.training_data_path
        self.training_data_files = glob.glob(self.training_data_path)
        self.testing_data_path = sep.testing_data_path
        self.testing_data_files = glob.glob(self.testing_data_path)


    def set_test_trap(self, test_trap_is_random=False):
         self.test_trap = True
         self.test_trap_is_random = test_trap_is_random


    def reset_environment(self):
        self.done = False
        self.state, self.orientation = self.initial_state()

        # Randomizing the test traps: 
        # Each trap is size 0.5 by 0.5
        # Trap 1 will be randomly placed between x = 0 and 4 (goal x)
        # Trap 2 will be randomly placed between x = 4.5 and 8 (end of goal x to end of xrange - 0.5)
        # Each trap will be randomly placed between y = 0.5 and 0.75 (dark line)

        if self.test_trap and self.test_trap_is_random:
            trap_size = 0.5
            trap1_x = np.random.rand() * (self.target_x[0] - self.xrange[0]) + self.xrange[0]
            trap2_x = np.random.rand() * (self.xrange[1]-trap_size - self.target_x[1]) + self.target_x[1]
            self.test_trap_x = [[trap1_x, trap1_x+trap_size], [trap2_x, trap2_x+trap_size]] 

            trap1_y = np.random.rand() * (self.dark_line - self.init_strip_y[1]) + self.init_strip_y[1]
            trap2_y = np.random.rand() * (self.dark_line - self.init_strip_y[1]) + self.init_strip_y[1]
            self.test_trap_y = [[trap1_y, trap1_y+trap_size], [trap2_y, trap2_y+trap_size]]


    def initial_state(self):
        orientation = np.random.rand()
        orientation = orientation * (self.thetas[1] - self.thetas[0]) + self.thetas[0]

        valid_state = False
        while not valid_state:
            state = np.random.rand(sep.dim_state)
            temp = state[1]
            state[0] = state[0] * (self.init_strip_x[1] - self.init_strip_x[0]) + self.init_strip_x[0]
            state[1] = temp * (self.trap_y[1] - self.trap_y[0]) + self.trap_y[0]  # Only consider x for in_trap
            if not self.in_trap(state) and not self.in_goal(state):
                valid_state = True
                state[1] = temp * (self.init_strip_y[1] - self.init_strip_y[0]) + self.init_strip_y[0]

        return state, orientation 

    
    def preprocess_data(self):
        # For normalizing the images - per channel mean and std
        print("Preprocessing the data")

        try:
            normalization_data = pickle.load(open("data_normalization.p", "rb"))
            rmean, gmean, bmean, rstd, gstd, bstd = normalization_data
            print("Done preprocessing")
            return rmean, gmean, bmean, rstd, gstd, bstd
        except Exception:
            rmean = 0
            gmean = 0
            bmean = 0
            rstd = 0
            gstd = 0
            bstd = 0
            for i in range(len(self.training_data_files)):
                img_path = self.training_data_files[i]
                src = cv2.imread(img_path, cv2.IMREAD_COLOR)
                src = src[:,:,::-1]   ## CV2 works in BGR space instead of RGB!! So dumb! --- now src is in RGB
                
                rmean += src[:, :, 0].mean()/len(self.training_data_files)
                gmean += src[:, :, 1].mean()/len(self.training_data_files)
                bmean += src[:, :, 2].mean()/len(self.training_data_files)
                
                rstd += src[:, :, 0].std()/len(self.training_data_files)  ## TODO: FIX?
                gstd += src[:, :, 1].std()/len(self.training_data_files)
                bstd += src[:, :, 2].std()/len(self.training_data_files)
            
            normalization_data = [rmean, gmean, bmean, rstd, gstd, bstd]
            pickle.dump(normalization_data, open("data_normalization.p", "wb"))

            print("Done preprocessing")

            return rmean, gmean, bmean, rstd, gstd, bstd


    def noise_image(self, image, state, noise_amount=sep.noise_amount):
        salt = 255
        pepper = 0

        out = image

        if state[1] <= self.dark_line: # Dark observation - add salt & pepper noise
            s_vs_p = 0.5
            amount = noise_amount  
            out = np.copy(image)
            num_salt = np.ceil(amount * image.size * s_vs_p)
            num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
            noise_indices = np.random.choice(image.size, int(num_salt + num_pepper), replace=False) 
            salt_indices = noise_indices[:int(num_salt)]
            pepper_indices = noise_indices[int(num_salt):]
            salt_coords = np.unravel_index(salt_indices, image.shape)
            pepper_coords = np.unravel_index(pepper_indices, image.shape)
            out[salt_coords] = salt
            out[pepper_coords] = pepper
        
        #cv2.imwrite("out_debug.png", out)

        return out
    
    
    def noise_image_plane(self, image, state, noise_amount=sep.noise_amount):
        # Corrupts the R, G, and B channels of noise_amount * (32 x 32) pixels

        salt = 255
        pepper = 0

        out = image

        image_plane_size = image.shape[0] * image.shape[1]
        image_plane_shape = image.shape[:2]
        if state[1] <= self.dark_line: # Dark observation - add salt & pepper noise
            s_vs_p = sep.salt_vs_pepper
            amount = noise_amount  
            out = np.copy(image)
            num_salt = np.ceil(amount * image_plane_size * s_vs_p)
            num_pepper = np.ceil(amount * image_plane_size * (1. - s_vs_p))
            noise_indices = np.random.choice(image_plane_size, int(num_salt + num_pepper), replace=False) 
            
            salt_indices = noise_indices[:int(num_salt)]
            pepper_indices = noise_indices[int(num_salt):]
            #salt_coords = np.unravel_index(salt_indices, image_plane_shape)
            #pepper_coords = np.unravel_index(pepper_indices, image_plane_shape)
            salt_coords = (np.array([int(elem) for elem in salt_indices/image_plane_shape[1]]), 
                            salt_indices%image_plane_shape[1])
            pepper_coords = (np.array([int(elem) for elem in pepper_indices/image_plane_shape[1]]), 
                            pepper_indices%image_plane_shape[1])
            
            if num_salt != 0:
                out[salt_coords[0], salt_coords[1], :] = salt
            # for i in range(len(salt_coords[0])):  # salt_coords[0] is row indices, salt_coords[1] is col indices
            #     row = salt_coords[0][i]
            #     col = salt_coords[1][i]
            #     for j in range(3):
            #         out[row, col, j] = salt
            if num_pepper != 0:
                out[pepper_coords[0], pepper_coords[1], :] = pepper
            # for i in range(len(pepper_coords[0])):  # pepper_coords[0] is row indices, pepper_coords[1] is col indices
            #     row = pepper_coords[0][i]
            #     col = pepper_coords[1][i]
            #     for j in range(3):
            #         out[row, col, j] = pepper
        
        #cv2.imwrite("out_debug1.png", out)

        return out

    
    def noise_image_occlusion(self, image, state, occlusion_amount=sep.occlusion_amount):
        # Turns a randomly chosen occlusion_amount x occlusion_amount square in the image to black 

        out = image

        if state[1] <= self.dark_line: # Dark observation - add occlusion
            amount = occlusion_amount

            start_index_x = np.random.randint(image.shape[1] - occlusion_amount)  # column index
            start_index_y = np.random.randint(image.shape[0] - occlusion_amount)  # row index
            start_point = (start_index_x, start_index_y)
            end_point = (start_index_x + amount, start_index_y + amount)
            color = (0, 0, 0)
            thickness = -1

            out = np.copy(image)
            out = cv2.rectangle(out, start_point, end_point, color, thickness)
        
        #cv2.imwrite("out_debug2.png", out)

        return out


    def get_observation(self, state=None, normalize=True, normalization_data=None, occlusion=False):
        if state == None:
            state_temp = self.state
            state = self.state + self.true_env_corner
            state_arr = np.array([[state[0], state[1], self.orientation]])
        else:
            state_temp = state
            state = state + self.true_env_corner
            state_arr = np.array([state])

        path = os.getcwd() + '/images/' 
        #os.mkdir(path)
        check_path(path)

        img_path, traversible, dx_m = generate_observation(state_arr, path)
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        
        if occlusion:
            out = self.noise_image_occlusion(image, state_temp)
        else:
            #out = self.noise_image(image, state_temp)
            out = self.noise_image_plane(image, state_temp)
            
        out = out[:, :, ::-1]  ## CV2 works in BGR space instead of RGB!! So dumb! --- now out is in RGB
        out = np.ascontiguousarray(out)

        if normalize:
            if normalization_data is None: raise Exception("Normalization data is None!")

            rmean, gmean, bmean, rstd, gstd, bstd = normalization_data
            img_rslice = (out[:, :, 0] - rmean)/rstd
            img_gslice = (out[:, :, 1] - gmean)/gstd
            img_bslice = (out[:, :, 2] - bmean)/bstd

            out = np.stack([img_rslice, img_gslice, img_bslice], axis=-1)

            #out = (out - out.mean())/out.std()  # "Normalization" -- TODO

        os.remove(img_path)
        os.rmdir(path)

        return out, img_path, traversible, dx_m
    

    # Don't think this is used
    def read_observation(self, img_path, normalize):
        obs = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        #obs = cv2.imread(img_path, cv2.IMREAD_COLOR)
        obs = obs[:,:,::-1]   ## CV2 works in BGR space instead of RGB!! So dumb! --- now obs is in RGB
        # if normalize:
        #     obs = (obs - obs.mean())/obs.std()  # "Normalization" -- TODO
        
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

        # Check if state y-value is the same as trap/goal but it's not in the trap or goal
        if (state[1] >= self.trap_y[0] and state[1] <= self.trap_y[1]) and \
             not self.in_trap(state) and not self.in_goal(state):
             return True

        map_state = self.point_to_map(np.array(state[:2] + self.true_env_corner))
        map_value = self.traversible[map_state[1], map_state[0]]
        collided = (map_value == 0)
        return collided

    
    def in_trap(self, state):
        # Returns true if in trap
        first_trap = self.trap_x[0]
        first_trap_x = (state[0] >= first_trap[0] and state[0] <= first_trap[1])
        second_trap = self.trap_x[1]
        second_trap_x = (state[0] >= second_trap[0] and state[0] <= second_trap[1])
        trap_x = first_trap_x or second_trap_x

        trap = trap_x and (state[1] >= self.trap_y[0] and state[1] <= self.trap_y[1]) 

        # Traps that may appear during test time -- optional
        if self.test_trap:
            first_test_trap = self.test_trap_x[0]
            first_test_trap_x = (state[0] >= first_test_trap[0] and state[0] <= first_test_trap[1])
            second_test_trap = self.test_trap_x[1]
            second_test_trap_x = (state[0] >= second_test_trap[0] and state[0] <= second_test_trap[1])
            #test_trap_x = first_test_trap_x or second_test_trap_x
            
            first_test_trap = self.test_trap_y[0]
            first_test_trap_y = (state[1] >= first_test_trap[0] and state[1] <= first_test_trap[1])
            second_test_trap = self.test_trap_y[1]
            second_test_trap_y = (state[1] >= second_test_trap[0] and state[1] <= second_test_trap[1])

            test_trap = (first_test_trap_x and first_test_trap_y) or \
                        (second_test_trap_x and second_test_trap_y)

            # test_trap = test_trap_x and (state[1] >= self.test_trap_y[0] and state[1] <= self.test_trap_y[1])
        
            return (trap or test_trap) and not self.in_goal(state)

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
        return
        # Check if a given state tensor is a terminal state
        s = s[:, :2]
        targets = np.tile(self.target, (s.shape[0], 1))
        
        true_dist = l2_distance_np(s, targets)
        
        return all(true_dist <= sep.end_range)

    
    def make_pars(self, batch_size):
        thetas = np.random.rand(batch_size, 1) * (self.thetas[1] - self.thetas[0]) + self.thetas[0]
        # xs = np.random.rand(batch_size, 1) * (self.init_strip_x[1] - self.init_strip_x[0]) + self.init_strip_x[0]
        # ys = np.random.rand(batch_size, 1) * (self.init_strip_y[1] - self.init_strip_y[0]) + self.init_strip_y[0]

        xs = np.zeros((batch_size, 1))
        ys = np.zeros((batch_size, 1))

        for i in range(batch_size):
            [x, y], _ = self.initial_state()
            xs[i, 0] = x
            ys[i, 0] = y

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


# se = StanfordEnvironment()
# se.state = [4, 0.3]
# r = se.step([0.5])
# print("REWARD", r)
# print(se.done)

# trap = se.in_trap([0.5, 0.24])
# print("TRAP", trap)
