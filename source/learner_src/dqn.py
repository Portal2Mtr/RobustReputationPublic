from collections import Counter

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import statistics
import random
import math
from .Base_Agent import Base_Agent
from .Epsilon_Greedy_Exploration import Epsilon_Greedy_Exploration
from .DBTS_Exploration import DBTS_Exploration
from .Replay_Buffer import Replay_Buffer
from .LQN_Exploration import LQN_Exploration


cons_dict = {
    'EPSGREEDY': Epsilon_Greedy_Exploration,
    'MVEDQL': Epsilon_Greedy_Exploration,
    'DBTS': DBTS_Exploration,
    'TS': DBTS_Exploration,
    'LQN': LQN_Exploration,
    'CONTROL': Epsilon_Greedy_Exploration
}


class DQN(Base_Agent):
    """A deep Q learning agent"""
    agent_name = "DQN"
    def __init__(self, config):
        Base_Agent.__init__(self, config)
        self.memory = Replay_Buffer(self.hyperparameters["buffer_size"], self.hyperparameters["batch_size"], config.seed, self.device)
        self.q_network_local = self.create_NN(input_dim=self.state_size, output_dim=self.action_size)
        self.q_network_optimizer = optim.Adam(self.q_network_local.parameters(),
                                              lr=self.hyperparameters["learning_rate"], eps=1e-4)

        self.exploration_strategy = cons_dict[config.consensus_scheme](config,
                                                                       self.config.exploreparams[config.consensus_scheme])


        self.cons_scheme = config.consensus_scheme
        self.cons_intent = config.consensus_intent
        # Change consensus scheme between nodes

        if self.cons_scheme == "TS":
            self.exploration_strategy.is_vanilla_ts = True

        # Reputation variables
        self.minqvalues = [np.inf for _ in range(self.action_size)]
        self.maxqvalues = [0.0 for _ in range(self.action_size)]
        self.retqvalues = [0.0 for _ in range(self.action_size)]
        self.self_idx = config.self_idx
        self.inner_learn = config.hyperparameters['inner_learn']
        self.share_exp = None
        self.share_arm = None
        self.temperature_vals = []

    def reset_game(self):
        super(DQN, self).reset_game()
        self.update_learning_rate(self.hyperparameters["learning_rate"], self.q_network_optimizer)

    def step(self):
        """Runs a step within a game including a learning step if required"""
        # while not self.done:
        self.action = self.pick_action()
        self.conduct_action(self.action)
        if self.time_for_q_network_to_learn():
            for _ in range(self.hyperparameters["learning_iterations"]):
                self.learn()
        self.save_experience()
        if self.cons_scheme == 'LQN':
            self.temperature_vals.append(self.exploration_strategy.max_temp_thresh)
        if self.cons_scheme == 'MVEDQL' or self.cons_scheme == 'LQN':
            self.share_exp = self.export_experience()
        else:
            self.share_exp = None

        self.state = self.next_state  # this is to set the state for the next iteration
        self.global_step_number += 1
        self.episode_number += 1

    def update_exploration_rep(self,rep_dict):
        self.exploration_strategy.update_neighbors_rep(rep_dict)

    def get_avg_temperature(self):

        if len(self.temperature_vals) <= 1:
            return self.exploration_strategy.max_temp_thresh

        return statistics.mean(self.temperature_vals)

    def pick_action(self, state=None):
        """Uses the local Q network and an epsilon greedy policy to pick an action"""
        # PyTorch only accepts mini-batches and not single observations so we have to use unsqueeze to add
        # a "fake" dimension to make it a mini-batch rather than a single observation
        if state is None: state = self.state
        if isinstance(state, np.int64) or isinstance(state, int): state = np.array([state])
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        if len(state.shape) < 2: state = state.unsqueeze(0)
        self.q_network_local.eval() # puts network in evaluation mode
        with torch.no_grad():
            action_values = self.q_network_local(state)
        self.q_network_local.train() #puts network back in training mode
        if self.cons_scheme == "LQN":
            self.exploration_strategy.temperature = self.get_avg_temperature()

        action = self.exploration_strategy.perturb_action_for_exploration_purposes({"action_values": action_values,
                                                                                    "turn_off_exploration": self.turn_off_exploration,
                                                                                    "episode_number": self.episode_number})

        self.share_arm = self.exploration_strategy.share_arm

        return action

    def learn(self, experiences=None):
        """Runs a learning iteration for the Q network"""
        if experiences is None: states, actions, rewards, next_states, dones = self.sample_experiences() #Sample experiences
        else: states, actions, rewards, next_states, dones = experiences
        loss = self.compute_loss(states, next_states, rewards, actions, dones)

        if self.cons_scheme == "LQN":
            idxs = self.memory.recent_idxs
            temperature = self.exploration_strategy.temperature
            lenient_thresh = 1 - math.exp(-self.hyperparameters['batch_size'] * temperature)

            for idx in idxs:
                check_temp = self.exploration_strategy.beta * self.exploration_strategy.temperature
                if check_temp < self.exploration_strategy.max_temp_thresh:
                    self.temperature_vals[idx] = self.exploration_strategy.beta * self.exploration_strategy.temperature
                else:
                    self.temperature_vals[idx] = self.exploration_strategy.max_temp_thresh

            if random.random() < lenient_thresh:
                self.q_network_optimizer.zero_grad()
                return



        # self.logger.info("Action counts {}".format(Counter(actions_list)))
        self.take_optimisation_step(self.q_network_optimizer, self.q_network_local, loss, self.hyperparameters["gradient_clipping_norm"])

    def compute_loss(self, states, next_states, rewards, actions, dones):
        """Computes the loss required to train the Q network"""
        with torch.no_grad():
            Q_targets = self.compute_q_targets(next_states, rewards, dones)
        Q_expected = self.compute_expected_q_values(states, actions)
        loss = F.mse_loss(Q_expected, Q_targets)
        return loss

    def compute_q_targets(self, next_states, rewards, dones):
        """Computes the q_targets we will compare to predicted q values to create the loss to train the Q network"""
        Q_targets_next = self.compute_q_values_for_next_states(next_states)
        Q_targets = self.compute_q_values_for_current_states(rewards, Q_targets_next, dones)
        return Q_targets

    def compute_q_values_for_next_states(self, next_states):
        """Computes the q_values for next state we will use to create the loss to train the Q network"""
        Q_targets_next = self.q_network_local(next_states).detach().max(1)[0].unsqueeze(1)
        return Q_targets_next

    def compute_q_values_for_current_states(self, rewards, Q_targets_next, dones):
        """Computes the q_values for current state we will use to create the loss to train the Q network"""
        # Note for reputation system, discount deviation here

        Q_targets_current = rewards + (self.hyperparameters["discount_rate"] * Q_targets_next * (1 - dones))

        if self.cons_intent != 'ATTACK':
            if self.cons_scheme != 'TS' and self.cons_scheme != 'CONTROL':
                Q_targets_current = self.exploration_strategy.custom_update_law(Q_targets_current,self.retqvalues)

        return Q_targets_current



    def compute_expected_q_values(self, states, actions):
        """Computes the expected q_values we will use to create the loss to train the Q network"""
        Q_expected = self.q_network_local(states).gather(1, actions.long())  # must convert actions to long so can be used as index

        with torch.no_grad():
            minqval = np.inf
            maxqval = -np.inf
            for i in range(len(Q_expected)):
                minqval = min(minqval, Q_expected[i].item())
                maxqval = max(maxqval, Q_expected[i].item())

            for i in range(len(Q_expected)):
                try:
                    self.retqvalues[i] = (Q_expected[i].item() - minqval)/(maxqval-minqval)
                except ZeroDivisionError:
                    self.retqvalues[i] = 0.0


            if self.cons_scheme == "DBTS" or self.cons_scheme == "TS":
                self.exploration_strategy.update_ts(self.retqvalues)

        return Q_expected

    def locally_save_policy(self):
        """Saves the policy"""
        torch.save(self.q_network_local.state_dict(), "Models/{}_local_network.pt".format(self.agent_name))

    def time_for_q_network_to_learn(self):
        """Returns boolean indicating whether enough steps have been taken for learning to begin and there are
        enough experiences in the replay buffer to learn from"""
        return self.right_amount_of_steps_taken() and self.enough_experiences_to_learn_from()

    def right_amount_of_steps_taken(self):
        """Returns boolean indicating whether enough steps have been taken for learning to begin"""
        return self.global_step_number % self.hyperparameters["update_every_n_steps"] == 0

    def sample_experiences(self):
        """Draws a random sample of experience from the memory buffer"""
        experiences = self.memory.sample()
        states, actions, rewards, next_states, dones = experiences
        return states, actions, rewards, next_states, dones
