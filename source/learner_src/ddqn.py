import copy

from Base_Agent import Base_Agent
from dqn import DQN

class DQN_With_Fixed_Q_Targets(DQN):
    """A DQN agent that uses an older version of the q_network as the target network"""
    agent_name = "DQN with Fixed Q Targets"
    def __init__(self, config):
        DQN.__init__(self, config)
        self.q_network_target = self.create_NN(input_dim=self.state_size, output_dim=self.action_size)
        Base_Agent.copy_model_over(from_model=self.q_network_local, to_model=self.q_network_target)

    def learn(self, experiences=None):
        """Runs a learning iteration for the Q network"""
        super(DQN_With_Fixed_Q_Targets, self).learn(experiences=experiences)
        self.soft_update_of_target_network(self.q_network_local, self.q_network_target,
                                           self.hyperparameters["tau"])  # Update the target network

    def compute_q_values_for_next_states(self, next_states):
        """Computes the q_values for next state we will use to create the loss to train the Q network"""
        Q_targets_next = self.q_network_target(next_states).detach().max(1)[0].unsqueeze(1)
        return Q_targets_next


class DDQN(DQN_With_Fixed_Q_Targets):
    """A double DQN agent"""
    agent_name = "DDQN"

    def __init__(self, config):
        DQN_With_Fixed_Q_Targets.__init__(self, config)

    def compute_q_values_for_next_states(self, next_states):
        """Computes the q_values for next state we will use to create the loss to train the Q network. Double DQN
        uses the local index to pick the maximum q_value action and then the target network to calculate the q_value.
        The reasoning behind this is that it will help stop the network from overestimating q values"""
        max_action_indexes = self.q_network_local(next_states).detach().argmax(1)
        Q_targets_next = self.q_network_target(next_states).gather(1, max_action_indexes.unsqueeze(1))
        return Q_targets_next
