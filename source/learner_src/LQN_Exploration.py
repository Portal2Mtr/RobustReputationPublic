from .Base_Exploration_Strategy import Base_Exploration_Strategy
import numpy as np
import random
import torch

class LQN_Exploration(Base_Exploration_Strategy):
    """Implements an epsilon greedy exploration strategy"""
    def __init__(self, config,params):
        super().__init__(config,params)
        self.notified_that_exploration_turned_off = False
        self.exploration_cycle_episodes_length = None
        self.random_episodes_to_run = 0
        self.curr_neis = None
        self.neighbors_rep_dict = {}
        self.temperature = self.params['max_temp']
        self.beta = self.params['beta_start']
        self.max_temp_thresh = self.params['max_temp']
        self.max_temp_decay = self.params["max_temp_decay"]

    def update_neighbors_rep(self,rep_dict):

        for key in rep_dict.keys():
            self.neighbors_rep_dict[key] = rep_dict[key]

        if len(rep_dict.keys()) > 1:
            self.curr_neis = [i for i in rep_dict.keys()]

    def update_neighbors_arm(self,arm_dict):
        return

    def perturb_action_for_exploration_purposes(self, action_info):
        """Perturbs the action of the agent to encourage exploration"""
        action_values = action_info["action_values"]
        turn_off_exploration = action_info["turn_off_exploration"]
        episode_number = action_info["episode_number"]
        if turn_off_exploration and not self.notified_that_exploration_turned_off:
            self.notified_that_exploration_turned_off = True
        epsilon = self.temperature ** self.params["epsilon"]
        self.beta = self.beta * self.params['beta_decay']
        self.beta = max(self.beta, self.params['beta_min'])
        self.max_temp_thresh = self.max_temp_decay * self.max_temp_thresh

        if (random.random() < epsilon) and (episode_number >= self.random_episodes_to_run):
            return torch.argmax(action_values).item()
        return  np.random.randint(0, action_values.shape[1])


    def custom_update_law(self,Q_target,Q_current):

        store_neis = []
        for nei in self.curr_neis:
            store_neis.append(self.neighbors_rep_dict[nei])

        store_neis.append(Q_current)
        Q_average = np.average(store_neis, axis=0)

        Q_average = torch.tensor(Q_average)
        Q_average = Q_average.type(torch.FloatTensor)

        Q_target = Q_target.view(len(Q_target))

        Q_target = Q_target + Q_average  # Add average of Q-values

        Q_target.view(len(Q_target), 1)

        return Q_target


    def add_exploration_rewards(self, reward_info):
        """Actions intrinsic rewards to encourage exploration"""
        return reward_info["reward"]

    def reset(self):
        """Resets the noise process"""
        pass
