from .Base_Exploration_Strategy import Base_Exploration_Strategy
import numpy as np
import random
import torch



"""
Implements the QD Consensus strategy.
"""

class QD_Update_Greedy_Exploration(Base_Exploration_Strategy):
    """Implements an epsilon greedy exploration strategy"""
    def __init__(self, config,params):
        super().__init__(config,params)

        num_neighbors = 10
        self.neighbors_rep_dict = {}
        for i in range(num_neighbors):
            self.neighbors_rep_dict[i] = [0.0 for _ in range(num_neighbors)]
        self.notified_that_exploration_turned_off = False
        if "exploration_cycle_episodes_length" in self.config.hyperparameters.keys():
            # print("Using a cyclical exploration strategy")
            self.exploration_cycle_episodes_length = self.config.hyperparameters["exploration_cycle_episodes_length"]
        else:
            self.exploration_cycle_episodes_length = None

        if "random_episodes_to_run" in self.config.hyperparameters.keys():
            self.random_episodes_to_run = self.config.hyperparameters["random_episodes_to_run"]
            # print("Running {} random episodes".format(self.random_episodes_to_run))
        else:
            self.random_episodes_to_run = 0


        # Setup Robust QD

        self.alpha_start = 0.01
        self.alpha = 0.01
        self.alpha_decay = 1
        self.b = 0.1
        self.eps_number = 0.0
        self.curr_neis = None

        # Alpha is decreasing over time, b is constant

    def update_neighbors_rep(self, rep_dict):

        for key in rep_dict.keys():
            self.neighbors_rep_dict[key] = rep_dict[key]

        if len(rep_dict.keys()) > 1:
            self.curr_neis = [i for i in rep_dict.keys()]

    def custom_update_law(self,Q_target,Q_current):

        sum_q_values = [0.0 for _ in range(10)]
        for nei in self.curr_neis:
            diff = [i - j for i,j in zip(Q_current,self.neighbors_rep_dict[nei])]
            sum_q_values = [i+j for i,j in zip(sum_q_values,diff)]

        sum_q_values = torch.tensor(sum_q_values)
        Q_current = torch.tensor(Q_current)

        Q_target = Q_target.view(len(Q_target))

        a_k = self.alpha / (1.0 + self.eps_number)

        Q_target = Q_current - self.b * sum_q_values + a_k * Q_target

        Q_target.view(len(Q_target), 1)

        return Q_target

    def perturb_action_for_exploration_purposes(self, action_info):
        """Perturbs the action of the agent to encourage exploration"""
        action_values = action_info["action_values"]
        turn_off_exploration = action_info["turn_off_exploration"]
        episode_number = action_info["episode_number"]
        if turn_off_exploration and not self.notified_that_exploration_turned_off:
            # print(" ")
            # print("Exploration has been turned OFF")
            # print(" ")
            self.notified_that_exploration_turned_off = True
        epsilon = self.get_updated_epsilon_exploration(action_info)


        if (random.random() > epsilon or turn_off_exploration) and (episode_number >= self.random_episodes_to_run):
            return torch.argmax(action_values).item()
        return  np.random.randint(0, action_values.shape[1])

    def get_updated_epsilon_exploration(self, action_info, epsilon=1.0):
        """Gets the probability that we just pick a random action. This probability decays the more episodes we have seen"""
        episode_number = action_info["episode_number"]
        epsilon_decay_denominator = self.params["epsilon_decay_rate_denominator"]

        epsilon = epsilon / (1.0 + (episode_number / epsilon_decay_denominator))
        self.eps_number = episode_number
        return epsilon


    def add_exploration_rewards(self, reward_info):
        """Actions intrinsic rewards to encourage exploration"""
        return reward_info["reward"]

    def reset(self):
        """Resets the noise process"""
        pass
