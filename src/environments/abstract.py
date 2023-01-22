# author: @sdeglurkar, @jatucker4, @michaelhlim

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

    def is_terminal(self, s):
        # Check if a given state is a terminal state
        return

    def action_sample(self):
        # Gives back a uniformly sampled random action
        return

    def reward(self, s):
        # Gives back reward for corresponding state
        return