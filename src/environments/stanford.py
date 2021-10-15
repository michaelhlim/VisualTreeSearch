# author: @wangyunbo
from configs.environments.floor import DISCOUNT
import cv2
import glob
import numpy as np
import os
import pickle
import random

from src.environments.abstract import AbstractEnvironment
from utils.utils import *

from configs.environments.stanford import *

sep = Stanford_Environment_Params()

from examples.examples import *  # generate_observation


class StanfordEnvironment(AbstractEnvironment):
    def __init__(self):
        self.done = False
        self.true_env_corner = [24.0, 23.0] 
        self.xrange = [0, 8.5]
        self.yrange = [0, 1.5]
        self.thetas = [0.0, 2*np.pi]
        self.discrete_thetas = np.array([0.0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4])
        self.discrete_thetas = self.discrete_thetas.reshape((len(self.discrete_thetas), 1))
        self.trap_x = [[1.5, 2], [6.5, 7]] #[[1.5, 2], [6, 6.5]] #[[0, 0], [8, 8]] #[[0, 1], [7, 8]] 
        self.trap_y = [0, 0.25]
        self.target_x = [4, 4.5] #[3.5, 4.5] #[1.5, 6.5] #[3, 5] #[3.5, 4.5]
        self.target_y = [0, 0.25]
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
            _, _, traversible, dx_m = self.get_observation(path=path)
            pickle.dump(traversible, open("traversible.p", "wb"))

        # if traversible_dump:
        #     path = os.getcwd() + '/temp/'
        #     os.mkdir(path)
        #     _, _, traversible, dx_m = self.get_observation(path=path)
        #     pickle.dump(traversible, open("traversible.p", "wb"))
        # else:
        #     traversible = pickle.load(open("traversible.p", "rb"))
        #     dx_m = 0.05

        self.traversible = traversible
        self.dx = dx_m
        self.map_origin = [0, 0]

        # For making training batches
        self.training_data_path = sep.training_data_path
        self.training_data_files = glob.glob(self.training_data_path)
        self.testing_data_path = sep.testing_data_path
        self.testing_data_files = glob.glob(self.testing_data_path)
        self.normalization = sep.normalization

    
    def reset_environment(self):
        self.done = False
        self.state, self.orientation = self.initial_state()

        # self.state = np.random.rand(sep.dim_state)
        # self.orientation = np.random.rand()
        # self.state[0] = self.state[0] * (self.init_strip_x[1] - self.init_strip_x[0]) + self.init_strip_x[0]
        # self.state[1] = self.state[1] * (self.init_strip_y[1] - self.init_strip_y[0]) + self.init_strip_y[0]
        # self.orientation = self.orientation * (self.thetas[1] - self.thetas[0]) + self.thetas[0]


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


    def get_observation(self, state=None, path=None, normalize=True):
        if state == None:
            state_temp = self.state
            state = self.state + self.true_env_corner
            state_arr = np.array([[state[0], state[1], self.orientation]])
        else:
            state_temp = state
            state = state + self.true_env_corner
            state_arr = np.array([state])

        if path == None:
            path = os.getcwd() + '/images/' 
            #os.mkdir(path)
            check_path(path)

        img_path, traversible, dx_m = generate_observation(state_arr, path)
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        
        out = self.noise_image(image, state_temp)
        out = out[:, :, ::-1]  ## CV2 works in BGR space instead of RGB!! So dumb! --- now image is in RGB
        out = np.ascontiguousarray(out)

        if normalize:
            out = (out - out.mean())/out.std()  # "Normalization" -- TODO

        os.remove(img_path)
        os.rmdir(path)

        return out, img_path, traversible, dx_m
    

    # Currently not used
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

        # Check if state y-value is the same as trap/goal but it's not in the trap or goal - that's a wall
        if self.in_trap([self.trap_x[0][0], state[1]]) and \
            not self.in_trap(state) and not self.in_goal(state):
            return True

        map_state = self.point_to_map(np.array(state[:sep.dim_state] + self.true_env_corner))
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

    
    def make_pars(self, batch_size):
        thetas = self.discrete_thetas[np.random.randint(len(self.discrete_thetas), size=batch_size)]
        # thetas = np.random.rand(batch_size, 1) * (self.thetas[1] - self.thetas[0]) + self.thetas[0]

        xs = np.zeros((batch_size, 1))
        ys = np.zeros((batch_size, 1))

        for i in range(batch_size):
            [x, y], _ = self.initial_state()
            xs[i, 0] = x
            ys[i, 0] = y

        par_batch = np.concatenate((xs, ys), 1)
        
        return par_batch, thetas


    ########## Methods for tree search ###########

    def action_sample(self):
        # Gives back a uniformly sampled random action
        rnd = int(random.random()*9)

        # No blank move
        while rnd == 4:
            rnd = int(random.random()*9)

        action = sep.velocity * np.array([(rnd % 3) - 1, (rnd // 3) - 1])

        # # just generate completely random
        # action_x = STEP_RANGE * (2 * random.random() - 1)
        # action_y = STEP_RANGE * (2 * random.random() - 1)
        # action = np.array([action_x, action_y])

        return action
    

    def is_terminal(self, s):
        # Check if a given state tensor is a terminal state
        terminal = [self.in_goal(state) for state in s]
        return all(terminal)

        # s = s[:, :2]
        # targets = np.tile(self.target, (s.shape[0], 1))

        # true_dist = l2_distance_np(s, targets)

        # return all(true_dist <= sep.end_range)


    def transition(self, s, w, a):
        # transition each state in state tensor s with actions in action/action tensor a
        if w is not None:
            weights = np.copy(w)
            next_weights = np.copy(w)
        else:
            # Dummy weight
            weights = np.ones(np.shape(s)[0])
            next_weights = np.ones(np.shape(s)[0])
        sp = s[:, :sep.dim_state] + a
        action_angle = np.arctan2(a[1], a[0])
        if action_angle < 0:  # Arctan stuff
            action_angle += 2*np.pi
        orientations_next = np.tile(action_angle, (len(sp), 1))  
        sp = np.concatenate((sp, orientations_next), -1) 
        next_state = np.copy(sp)
        reward = 0.0

        cond_hit = np.array([self.detect_collision(state) for state in sp])
        goal_achieved = np.array([self.in_goal(state) for state in sp])
        step_ok = (~cond_hit | goal_achieved)

        trap = np.array([self.in_trap(state) for state in sp])
        normal_step = ~(goal_achieved | trap)

        # If goal reached
        next_state[step_ok, :] = sp[step_ok, :]
        next_weights[goal_achieved] = 0.0
        reward += np.sum(weights[goal_achieved]) * sep.epi_reward

        # If trap reached
        reward -= np.sum(weights[trap]) * sep.epi_reward

        # Else
        reward += np.sum(weights[normal_step]) * sep.step_reward

        # Is the transition terminal?
        is_terminal = all(goal_achieved)

        if is_terminal:
            # Dummy weight
            next_weights = np.array([1/len(next_weights)] * len(next_weights))  
        else:
            # Reweight
            next_weights = next_weights / np.sum(next_weights)
                
        return next_state, next_weights, reward, is_terminal


    def distance_to_goal(self, state):
        '''
        Returns vector that gets the state x, y to the goal (rectangular) and 
        the length of that vector
        '''
        x = state[0]
        y = state[1]

        if self.in_goal(state):
            return np.array([0, 0])

        if x >= self.target_x[0] and x <= self.target_x[1]:
            y_dist_argmin = np.argmin([abs(self.target_y[1] - y), abs(self.target_y[0] - y)])
            vec_dist = [self.target_y[1] - y, self.target_y[0] - y]
            vec = np.array([0, vec_dist[y_dist_argmin]])
        
        elif y >= self.target_y[0] and y <= self.target_y[1]:
            x_dist_argmin = np.argmin([abs(self.target_x[1] - x), abs(self.target_x[0] - x)])
            vec_dist = [self.target_x[1] - x, self.target_x[0] - x]
            vec = np.array([vec_dist[x_dist_argmin], 0])
        
        else:
            corners = np.array([[self.target_x[0], self.target_y[0]], 
                                [self.target_x[0], self.target_y[1]],
                                [self.target_x[1], self.target_y[0]],
                                [self.target_x[1], self.target_y[1]]])
            corner_argmin = np.argmin([np.linalg.norm(np.array([x, y]) - corner) for corner in corners])
            corner_argmin = corners[corner_argmin]
            vec = corner_argmin - np.array([x, y])

        return vec, np.linalg.norm(vec)


    def rollout(self, s, ss, ws):
        '''
        s - one sampled state out of all particles
        ss - all other particles
        ws - weights on the particles
        '''

        # Roll out from state s, calculating the naive distance & reward to the goal, then check how it would do for all other particles
        vec, dist = self.distance_to_goal(s)
        steps = int(np.floor(dist/sep.velocity))
        ss_copy = np.copy(ss)
        ss_copy[:, :sep.dim_state] += vec  # Orientations do not matter for goal/trap checking

        # Going stepping number of times will provide the following intermediate rewards (geometric series result)
        gamma = np.power(sep.discount, steps)
        reward = sep.step_reward * (1.0 - gamma)/(1.0 - sep.discount)

        goal_reached = [self.in_goal(state) for state in ss_copy]
        trap_reached = [self.in_trap(state) for state in ss_copy]
        r = np.dot(goal_reached, ws) * sep.epi_reward + np.dot(trap_reached, ws) * -sep.epi_reward
        reward += gamma * r

        return reward 


    ########## For (pre)training - making batches ##########

    def shuffle_dataset(self):
        data_files_indices = list(range(len(self.training_data_files)))
        np.random.shuffle(data_files_indices)

        return data_files_indices


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
            for i in range(len(self.data_files)):
                img_path = self.data_files[i]
                src = cv2.imread(img_path, cv2.IMREAD_COLOR)
                src = src[:,:,::-1]   ## CV2 works in BGR space instead of RGB!! So dumb! --- now src is in RGB
                
                rmean += src[:, :, 0].mean()/len(self.data_files)
                gmean += src[:, :, 1].mean()/len(self.data_files)
                bmean += src[:, :, 2].mean()/len(self.data_files)
                
                rstd += src[:, :, 0].std()/len(self.data_files)  ## TODO: FIX?
                gstd += src[:, :, 1].std()/len(self.data_files)
                bstd += src[:, :, 2].std()/len(self.data_files)
            
            normalization_data = [rmean, gmean, bmean, rstd, gstd, bstd]
            pickle.dump(normalization_data, open("data_normalization.p", "wb"))

            print("Done preprocessing")

            return rmean, gmean, bmean, rstd, gstd, bstd


    def get_training_batch(self, batch_size, data_files_indices, epoch_step, 
                            normalization_data, num_particles, 
                            noise_amount=sep.noise_amount,
                            percent_blur=0, blur_kernel=3):

        rmean, gmean, bmean, rstd, gstd, bstd = normalization_data

        states = []
        orientations = []
        images = []
        remove = 4
        rounding = 3

        if (epoch_step + 1)*batch_size > len(data_files_indices):
            indices = data_files_indices[epoch_step*batch_size:]
        else:
            indices = data_files_indices[epoch_step*batch_size:(epoch_step + 1)*batch_size]

        num_blurred_images = int(percent_blur * len(indices))
        blur = np.zeros(len(indices))
        blur_indices = np.random.choice(len(indices), num_blurred_images, replace=False)
        blur[blur_indices] = 1.

        for i in range(len(indices)):
            index = indices[i]
            img_path = self.training_data_files[index]

            splits = img_path[:-remove].split('_')
            state = np.array([np.round(float(elem), rounding) for elem in splits[-(sep.dim_state + 1):]])
            state[:sep.dim_state] = state[:sep.dim_state] - self.true_env_corner
            states.append(state[:sep.dim_state])
            orientations.append(state[sep.dim_state])

            src = cv2.imread(img_path, cv2.IMREAD_COLOR)

            src = self.noise_image(src, state, noise_amount)

            if blur[i] == 1:
                blurred = cv2.GaussianBlur(src,(blur_kernel,blur_kernel),cv2.BORDER_DEFAULT)
                src = blurred[:,:,::-1]
            else:
                src = src[:,:,::-1]   ## CV2 works in BGR space instead of RGB!! So dumb! --- now src is in RGB

            if self.normalization:
                img_rslice = (src[:, :, 0] - rmean)/rstd
                img_gslice = (src[:, :, 1] - gmean)/gstd
                img_bslice = (src[:, :, 2] - bmean)/bstd

                img = np.stack([img_rslice, img_gslice, img_bslice], axis=-1)

                images.append(img)
            else:
                src = (src - src.mean())/src.std()
                images.append(src)


            if state[1] < self.dark_line:
                obs_std = sep.obs_std_dark
            else:
                obs_std = sep.obs_std_light
                
            par_vec_x = np.random.normal(state[0], obs_std, num_particles)
            par_vec_y = np.random.normal(state[1], obs_std, num_particles)

            middle_var = np.stack((par_vec_x, par_vec_y), 1)

            if i == 0:
                par_batch = middle_var
            else:
                par_batch = np.concatenate((par_batch, middle_var), 0)
        
        return np.array(states), np.array(orientations), np.array(images), par_batch
    

    def get_testing_batch(self, batch_size, normalization_data):
        rmean, gmean, bmean, rstd, gstd, bstd = normalization_data

        states = []
        orientations = []
        images = []
        blurred_images = []
        remove = 4
        rounding = 3
        
        indices = np.random.choice(range(len(self.testing_data_files)), batch_size, replace=False)
        
        for index in indices:

            img_path = self.testing_data_files[index]

            splits = img_path[:-remove].split('_')
            state = np.array([np.round(float(elem), rounding) for elem in splits[-(sep.dim_state + 1):]])
            state[:sep.dim_state] = state[:sep.dim_state] - self.true_env_corner
            states.append(state[:sep.dim_state])
            orientations.append(state[sep.dim_state])

            src = cv2.imread(img_path, cv2.IMREAD_COLOR)
            src = self.noise_image(src, state)

            blurred = cv2.GaussianBlur(src,(5,5),cv2.BORDER_DEFAULT)
            src = src[:,:,::-1]   ## CV2 works in BGR space instead of RGB!! So dumb! --- now src is in RGB
            blurred = blurred[:,:,::-1]

            if self.normalization:
                img_rslice = (src[:, :, 0] - rmean)/rstd
                img_gslice = (src[:, :, 1] - gmean)/gstd
                img_bslice = (src[:, :, 2] - bmean)/bstd

                img = np.stack([img_rslice, img_gslice, img_bslice], axis=-1)

                images.append(img)

                img_rslice = (blurred[:, :, 0] - rmean)/rstd
                img_gslice = (blurred[:, :, 1] - gmean)/gstd
                img_bslice = (blurred[:, :, 2] - bmean)/bstd

                img = np.stack([img_rslice, img_gslice, img_bslice], axis=-1)

                blurred_images.append(img)
            else:
                src = (src - src.mean())/src.std()
                images.append(src)

                blurred = (blurred - blurred.mean())/blurred.std()
                blurred_images.append(blurred)

        
        return np.array(states), np.array(orientations), np.array(images), np.array(blurred_images)  
    

    # def get_par_batch(self, states, num_particles):
    #     for i in range(len(states)):
    #         state = states[i]

    #         if state[1] < self.dark_line:
    #             obs_std = sep.obs_std_dark
    #         else:
    #             obs_std = sep.obs_std_light
                
    #         par_vec_x = np.random.normal(state[0], obs_std, num_particles)
    #         par_vec_y = np.random.normal(state[1], obs_std, num_particles)
    #         par_vec_theta = np.tile([state[2]], (num_particles, 1))

    #         middle_var = np.stack((par_vec_x, par_vec_y, par_vec_theta), 1)

    #         if i == 0:
    #             par_batch = middle_var
    #         else:
    #             par_batch = np.concatenate((par_batch, middle_var), 0)

    #     return par_batch


 

