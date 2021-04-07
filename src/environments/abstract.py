# author: @wangyunbo
from utils.utils import *


class AbstractEnvironment(object):
    def __init__(self):
        if type(self) is Base:
            raise Exception('Base is an abstract class and cannot be instantiated directly')

    def get_observation(self):
        # Generates observation from the current environment state
        return

    def step(self):
        # Takes a step from the current environment state with a supplied action and updates the environment state
        return

    # New functions to be added
    def transition_state(self, s, a):
        # Given a state and an action, return the next step state and reward
        return

    def transition_tensor(self, s, a):
        # Given a state tensor and an action, return the next step state tensor and rewards
        return

    def is_terminal(self, s):
        # Check if a given state is a terminal state
        return

    def action_sample(self):
        # Gives back a uniformly sampled random action
        return

    def action_sample_centered(self, a):
        # Gives back a random action sampled around a center a
        return