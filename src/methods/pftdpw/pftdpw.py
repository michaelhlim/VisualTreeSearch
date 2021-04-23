import torch
import numpy as np
from numpy import random
from dataclasses import dataclass
from utils.utils import *
from configs.solver.pftdpw import *


@dataclass
class BeliefNode:
	states: list # List of lists of floats
	weights: list # List of floats


@dataclass
class PFTTree:
	# Belief node
	child_actions: dict # belief_id -> [action_id]
	n_b_visits: list # number of visits to belief
	belief_ids: list # id -> belief
	# Action node
	n_act_visits: list # number of visits to action
	q: list # Q value function 
	action_ids: list # id -> action
	transitions: dict # action_id -> [(belief_id, r)] (action_id determines the (b,a) pair since all actions are unique)


class PFTDPW():
	def __init__(self, environment, transition_module, obs_density_module, obs_generator_module, rollout_function, rollout_policy):
		# Initialize tree
		self.initialize_tree()

		# Set up modules
		self.env = environment
		self.T = transition_module
		self.Z = obs_density_module
		self.G = obs_generator_module
		self.rollout_function = rollout_function
		self.rollout_policy = rollout_policy

		# Set up parameters
		self.n_query = NUM_QUERY
		self.n_par = NUM_PAR
		self.depth = HORIZON
		self.c_ucb = UCB_EXPLORATION
		self.k_obs = K_OBSERVATION
		self.alpha_obs = ALPHA_OBSERVATION
		self.k_act = K_ACTION
		self.alpha_act = ALPHA_ACTION
		self.discount = DISCOUNT

	def initialize_tree(self):
		self.tree = PFTTree(child_actions={}, n_b_visits=[], belief_ids=[], n_act_visits=[], q=[], action_ids=[], transitions={})

	def insert_belief_node(self, b):
		# Insert a belief node into the DPW tree
		b_id = len(self.tree.belief_ids)
		self.tree.n_b_visits.append(0)
		self.tree.belief_ids.append(b)
		self.tree.child_actions[b_id] = []

		return b_id

	def insert_action_node(self, b_id, a):
		# Insert an action node stemming from a belief node into the DPW tree
		a_id = len(self.tree.action_ids)
		self.tree.n_act_visits.append(0)
		self.tree.q.append(0)
		self.tree.action_ids.append(a)
		self.tree.child_actions[b_id].append(a_id)

		return a_id

	def insert_transition(self, a_id, bp_id, r):
		# Insert a new transition a_id -> bp_id (Basically inserting (b,a) -> b', r)
		self.tree.transitions[a_id].append((bp_id, r))

	def transition_step(self, b, a):
		# State transitions from the particles in b
		states_tensor = np.array(b.states)
		action = np.array(a)
		next_states = self.T.t_model(
			torch.FloatTensor(states_tensor).to(device), torch.FloatTensor(action).to(device))
		next_states = next_states.detach().cpu().numpy() # Passed as an np array for convenience in particle filter

		# Getting rewards
		rewards = self.env.reward(next_states)

		return next_states, rewards

	def is_terminal(self, b):
		# Check if the belief node is terminal (i.e. all particles are terminal)
		return self.env.is_terminal(b.states)

	def next_action(self, b_id):
		# Generate next action from belief id b_id
		return self.env.action_sample()

	def particle_filter_step(self, b, a):
		# Generate b' from T(b,a) and also insert it into the tree
		sp, rewards = self.transition_step(b.states, a)

		# Generate an observation from a random state
		s_idx = np.random.choice(len(b.states), 1)
		o = self.G.sample(1, torch.FloatTensor(sp[s_idx]).to(device))

		# Generate particle belief set
		lik, _, _ = self.Z.m_model(torch.FloatTensor(sp).to(device), 
				o, self.n_par)
		new_weights = np.multiply(np.array(b.weights), lik.detach().cpu().numpy())
		r = np.dot(rewards, new_weights)

		# Resample states (naive resampling)
		sp = sp[np.random.choice(len(new_weights), len(new_weights), p = new_weights)]
		resample_weights = [1/len(new_weights)] * len(new_weights)
		bp = BeliefNode(states=sp.tolist(), weights=resample_weights)

		return bp, r

	def rollout(self, b):
		# Rollout simulation starting from belief b
		s = b.states[np.random.choice(len(b.weights), 1, p = b.weights)]
		return self.rollout_function(s)

	def plan(self, b):
		# Builds a DPW tree and returns the best next action
		# Construct the DPW tree
		self.initialize_tree()
		b_init = self.insert_belief_node(b)

		# Plan with the tree by querying the tree for n_query number of times
		n_iter = 0
		for i in range(self.n_query):
			n_iter += 1
			self.simulate(b_init, self.depth)

		# Find the best action from the root node
		best_q = -np.inf
		best_a = None
		for child in child_actions[b_init]:
			if self.tree.q[child] > best_q:
				best_q = self.tree.q[child]
				best_a = child

		return best_a

	def simulate(self, b_id, d):
		# Simulates dynamics with a DPW tree
		b = self.tree.belief_ids[b_id]

		# Check if d == 0 (full depth) or belief is terminal
		if d == 0:
			return self.rollout(b)
		elif self.is_terminal(b):
			return 0.0

		# Action PW
		if not self.tree.child_actions[b_id] or len(self.tree.child_actions[b_id]) <= self.k_act * (self.tree.n_b_visits[b_id] ** self.alpha_act):
			# If no action present or PW condition met, do PW
			a = self.next_action(b_id)
			a_id = self.insert_action_node(b_id, a)
		else:
			# Otherwise perform UCB among existing actions
			best_ucb = -np.inf
			a_id = -1
			log_nb = np.log(self.tree.n_b_visits[b_id])

			for a_child in self.tree.child_actions[b_id]:
				n_child = self.tree.n_act_visits[a_child]
				q_child = self.tree.q[a_child]
				if (log_nb <= 0 and n_child == 0) or self.c_ucb ==0:
					ucb_child = q_child
				else:
					ucb_child = q_child + self.c_ucb * np.sqrt(log_nb/n_child)

				# Make sure we do not get a negative infinity
				assert ucb_child != -np.inf

				if ucb_child > best_ucb:
					best_ucb = ucb_child
					a_id = a_child

			# Make sure we get a valid action id
			assert a_id >= 0
			a = self.tree.action_ids[a_id]

		# State PW
		new_node = False
		if not self.tree.transitions[a_id] or len(self.tree.transitions[a_id]) <= self.k_obs * (self.tree.n_act_visits[a_id] ** self.alpha_obs):
			# If no state present or PW condition met, do PW
			bp, r = self.particle_filter_step(b, a)
			bp_id = self.insert_belief_node(bp)
			self.insert_transition(a_id, bp_id, r)
			new_node = True
		else:
			# Otherwise pick a belief node at random
			bp_id, r = self.tree.transitions[a_id][np.random.choice(range(len(self.tree.transitions[a_id])), 1)]

		# Simulate recursively
		if new_node:
			q = r + self.discount * self.rollout(bp_id)
		else:
			q = r + self.discount * self.simulate(bp_id, d-1)

		# Update the counters & quantities
		self.tree.n_b_visits[b_id] += 1
		self.tree.n_act_visits[a_id] += 1
		self.tree.q[a_id] += (q - self.tree.q[a_id])/self.tree.n_act_visits[a_id]

		return q
