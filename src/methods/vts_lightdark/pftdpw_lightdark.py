import numpy as np
from numpy import random
from dataclasses import dataclass
import torch
from utils.utils import *

from configs.solver.vts_lightdark import *

vlp = VTS_LightDark_Params()

@dataclass
class BeliefNode:
	states: list # List of lists of floats (it's ok to put in numpy array)
	weights: list # List of floats (it's ok to put in numpy array)


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
	def __init__(self, environment, obs_density_module, obs_generator_module):
		# Initialize tree
		self.initialize_tree()

		# Set up modules
		self.env = environment
		self.Z = obs_density_module
		self.G = obs_generator_module

		# Set up parameters
		self.n_query = vlp.num_query 
		self.n_par = vlp.num_par_pftdpw 
		self.depth = vlp.horizon 
		self.c_ucb = vlp.ucb_exploration 
		self.k_obs = vlp.k_observation 
		self.alpha_obs = vlp.alpha_observation 
		self.k_act = vlp.k_action 
		self.alpha_act = vlp.alpha_action 
		self.discount = vlp.discount 

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
		if self.tree.transitions.get(a_id, -1) == -1:
			self.tree.transitions[a_id] = []
		self.tree.transitions[a_id].append((bp_id, r))

	def transition_step(self, b, a):
		# State transitions from the particles in b
		states_tensor = b.states
		weights_vector = b.weights
		action = a
		next_states, new_weights, reward, is_terminal = self.env.transition(states_tensor, weights_vector, action)

		return next_states, new_weights, reward, is_terminal

	def is_terminal(self, b):
		# Check if the belief node is terminal (i.e. all particles are terminal)
		return self.env.is_terminal(b.states)

	def next_action(self, b_id):
		# Generate next action from belief id b_id
		return self.env.action_sample()

	def particle_filter_step(self, b, a):
		# Generate b' from T(b,a) and also insert it into the tree
		# NOTE this sp includes orientation!
		sp, new_weights, reward, is_terminal = self.transition_step(b, a)

		if is_terminal:
			dummy_weights = np.array([1/len(new_weights)] * len(new_weights))
			bp = BeliefNode(states=sp, weights=dummy_weights)
		else:
			# Generate an observation from a random state
			s_idx = np.random.choice(len(new_weights), 1, p = new_weights)
			o = self.G.sample(1, torch.FloatTensor(sp[s_idx]).to(vlp.device))  # [1, obs_encode_out]
			o = self.G.conv.decode(o)  # [1, 3, 32, 32]

			# Generate particle belief set
			# lik = self.Z.m_model(torch.FloatTensor(sp[:, :2]).to(vlp.device), 
			# 							torch.FloatTensor(sp[:, 2]).unsqueeze(1).to(vlp.device), 
			# 							o.detach(), self.n_par, obs_is_encoded=True)  # [1, num_par]
			lik = self.Z.m_model(torch.FloatTensor(sp[:, :2]).to(vlp.device), 
										torch.FloatTensor(sp[:, 2]).unsqueeze(1).to(vlp.device), 
										o.detach(), self.n_par)  # [1, num_par]
			new_weights = np.multiply(new_weights, lik.detach().cpu().numpy()).flatten()
			
			# Resample states (naive resampling)
			if np.sum(new_weights) > 0:
				new_weights = new_weights / np.sum(new_weights)
				sp = sp[np.random.choice(len(new_weights), len(new_weights), p = new_weights)]

			resample_weights = np.array([1/len(new_weights)] * len(new_weights))
			bp = BeliefNode(states=sp, weights=resample_weights)

		return bp, reward

	def rollout(self, b):
		# Rollout simulation starting from belief b
		s = b.states[np.random.choice(len(b.weights), 1, p = b.weights)].flatten()
		ss = b.states
		ws = b.weights
		return self.env.rollout(s, ss, ws)

	def solve(self, s, w):
		# call plan when given states and weights
		b = BeliefNode(states=s, weights=w)
		a_id = self.plan(b)

		#traj = self.trajectory(a_id)
		traj = None
		
		if a_id == None:
			return self.env.action_sample(), traj
		else:
			return self.tree.action_ids[a_id], traj


	def trajectory(self, a_id):
		# For visualization purposes: give the whole trajectory outputted by the planner
		# of length (self.depth)

		best_a = a_id
		if best_a == None:
			action_list = [self.env.action_sample()]
		else:
			action_list = [self.tree.action_ids[best_a]]

		i = 0
		while best_a is not None and i < self.depth:
			# Pick a belief node at random
			bp_id, r = self.tree.transitions[best_a][int(np.random.choice(range(len(self.tree.transitions[best_a])), 1))]
			# Find the best action from the new belief node
			if len(self.tree.child_actions[bp_id]) > 0:
				index = np.argmax(np.array([self.tree.q[child] for child in self.tree.child_actions[bp_id]])) 
				best_a = self.tree.child_actions[bp_id][index]
			else:
				best_a = None

			# best_q = -np.inf
			# best_a = None
			# for child in self.tree.child_actions[bp_id]:
			# 	if self.tree.q[child] > best_q:
			# 		best_q = self.tree.q[child]
			# 		best_a = child
			
			# Add the best action to the trajectory
			if best_a == None:
				action_list.append(self.env.action_sample()) 
			else:
				action_list.append(self.tree.action_ids[best_a])
			
			i += 1
		
		return action_list


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
		for child in self.tree.child_actions[b_init]:
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
		if (self.tree.transitions.get(a_id, -1) == -1) or len(self.tree.transitions[a_id]) <= self.k_obs * (self.tree.n_act_visits[a_id] ** self.alpha_obs):
			# If no state present or PW condition met, do PW
			bp, r = self.particle_filter_step(b, a)
			bp_id = self.insert_belief_node(bp)
			self.insert_transition(a_id, bp_id, r)
			new_node = True
		else:
			# Otherwise pick a belief node at random
			bp_id, r = self.tree.transitions[a_id][int(np.random.choice(range(len(self.tree.transitions[a_id])), 1))]

		# Simulate recursively
		if new_node:
			q = r + self.discount * self.rollout(bp)
		else:
			q = r + self.discount * self.simulate(bp_id, d-1)

		# Update the counters & quantities
		self.tree.n_b_visits[b_id] += 1
		self.tree.n_act_visits[a_id] += 1
		self.tree.q[a_id] += (q - self.tree.q[a_id])/self.tree.n_act_visits[a_id]

		return q
