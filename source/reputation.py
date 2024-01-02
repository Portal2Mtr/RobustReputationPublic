"""Reputation Calculations

Main reputation values calculations for nodes.

"""

import numpy as np
from gym import Env, spaces

from numpy.linalg import norm
from scipy.spatial.distance import euclidean


class RepEnv(Env):

    def __init__(self, num_nodes, grad_shape, num_store, self_idx, noise_dev, inner_alpha):
        super(RepEnv, self).__init__()

        self.name = 'RepEnv'
        self.env = self.name
        self.state_space = spaces.Box(low=-np.inf, high=np.inf, shape=(grad_shape + 5,))
        self.reward_threshold = np.inf
        self.trials = 100

        # Choose from available nodes or no confidence and use local
        self.action_space = spaces.Discrete(num_nodes,)
        self.elements = []
        self.state_array = []
        self.weight_record = [[] for _ in range(num_nodes)]
        self.state_record = [np.empty((0, grad_shape + 5)) for _ in range(num_nodes)]
        self.have_data = False
        self.max_store = num_store
        self.self_idx = self_idx
        self.noise_dev = float(noise_dev)
        self.up_mem_cnt = [self.max_store for _ in range(num_nodes)]
        self.rep_record = [0.0 for _ in range(num_nodes)]
        self.dist_record = [1E6 for _ in range(num_nodes)]
        self.min_dist = 1E6
        self.inner_alpha = inner_alpha
        self.var_est = 0.0
        self.local_state = None
        self.done = False
        self.reward = 0
        self.recent_action = 0

    def update_memory(self, vectors, update_weights=False):
        """
        Update memory with new vectors and calculate mean across variables
        :param vectors: State vector record
        :return:
        """

        if update_weights:
            for vector in vectors:
                node_idx = int(vector[0])
                self.weight_record[node_idx] = vector[1:-1]

        else:

            for vector in vectors:
                node_idx = int(vector[0])
                self.state_record[node_idx] = vector

    def check_ready(self,check_weights=False):
        """
        Check if ready to form opinion about network
        :return: Boolean for if ready
        """

        ready = True
        if not check_weights:
            check_record = self.state_record
        else:
            check_record = self.weight_record
        for entry in check_record:

            if not ready:
                continue

            if not np.any(entry):
                ready = False

        return ready

    def update_state(self, new_states, local_state, done):
        """
        Update the local memory state for the outer learner
        :param new_states: Incoming states from network
        :param local_state: Locally generated weight state
        :param done: Checking if done
        :return:
        """

        # Add Gaussian noise to the gradients and loss before storing to improve adversary robustness
        work_states = np.stack(new_states, axis=0)
        noise_val = float(np.random.normal(0, self.noise_dev))
        work_states[:][4:] += noise_val

        self.state_array = work_states
        self.local_state = local_state
        self.state_array = np.append(self.state_array, local_state, axis=0)
        self.update_memory(self.state_array)
        self.done = done

    def step(self, action):
        """
        Take a 'step' in the reputation environment
        :param action: Action to select (node id)
        :return: Weight state, reward (see below), if done (always false)
        """

        process_state = self.state_record[action]
        local_state = self.state_record[self.self_idx]
        dist_first = euclidean(process_state, local_state)
        process_weights = process_state[4:]
        local_weights = local_state[4:]
        # Implement reward boundary created for DPSGD
        dist_second = max(0, norm(np.subtract(process_weights, local_weights)) ** 2 - self.inner_alpha * self.var_est)
        dist = dist_first + dist_second
        self.dist_record[action] = float(dist)
        self.min_dist = min(self.min_dist, min(self.dist_record)) + 0.001
        self.reward = max(0, min(1, self.min_dist/(dist+0.001)))
        self.recent_action = action
        return process_state, self.reward, False

    def reset(self):
        """
        Resets reputation environment (not used beyond initialization)
        :return:
        """

        return self.state_space.sample()
