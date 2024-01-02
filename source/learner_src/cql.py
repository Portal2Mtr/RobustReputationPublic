"""
CQL Algorithm

Conservative Q-learning algorithm.
Code originally from https://github.com/BY571/CQL/blob/main/CQL-DQN/agent.py

"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import numpy as np
import random

class DDQN(nn.Module):
    def __init__(self, state_size, action_size, layer_size):
        super(DDQN, self).__init__()
        self.input_shape = state_size
        self.action_size = action_size
        self.head_1 = nn.Linear(self.input_shape[0], layer_size)
        self.ff_1 = nn.Linear(layer_size, layer_size)
        self.ff_2 = nn.Linear(layer_size, action_size)


    def forward(self, input):

        x = torch.relu(self.head_1(input))
        x = torch.relu(self.ff_1(x))
        out = self.ff_2(x)

        return out


class CQLAgent():
    def __init__(self, config, state_size, action_size, hidden_size=256, device="cpu"):
        self.config = config
        self.h_param = config.hyperparameters['CQL_Agent']
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.tau = self.h_param['tau']
        self.gamma = self.h_param['discount_rate']
        self.decay = self.h_param['decay_rate']

        self.network = DDQN(state_size=self.state_size,
                            action_size=self.action_size,
                            layer_size=hidden_size
                            ).to(self.device)

        self.target_net = DDQN(state_size=self.state_size,
                               action_size=self.action_size,
                               layer_size=hidden_size
                               ).to(self.device)

        self.optimizer = optim.Adam(params=self.network.parameters(), lr=1e-3)

        # Reputation variables
        self.minqvalues = [np.inf for _ in range(self.action_size)]
        self.maxqvalues = [0.0 for _ in range(self.action_size)]
        self.retqvalues = [0.0 for _ in range(self.action_size)]
        self.maxdecays = [0.0 for _ in range(self.action_size)]
        self.self_idx = self.h_param['self_idx']
        self.inner_learn = self.h_param['inner_learn']
        self.var_est = 0.0
        self.self_q_ref = 0.0

    def get_action(self, state, epsilon):
        if random.random() > epsilon:
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            self.network.eval()
            with torch.no_grad():
                action_values = self.network(state)
            self.network.train()
            action = np.argmax(action_values.cpu().data.numpy(), axis=1)
        else:
            action = random.choices(np.arange(self.action_size), k=1)

        return action

    def learn(self, experiences):
        self.optimizer.zero_grad()
        states = experiences[0]
        actions = experiences[1]
        rewards = experiences[2]
        next_states = experiences[3]
        dones = experiences[4]
        # states, actions, rewards, next_states, dones = experiences
        with torch.no_grad():
            Q_targets_next = self.target_net(next_states).detach().max(1)[0].unsqueeze(1)
            Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

            actcopy = actions.T
            actcopy = actcopy.tolist()[0]

            if self.self_idx in actcopy:
                for idx in range(len(actcopy)):
                    if actcopy[idx] == self.self_idx:
                        self.self_q_ref = Q_targets[idx]


        Q_a_s = self.network(states)
        Q_expected = Q_a_s.gather(1, actions)

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

        cql1_loss = torch.logsumexp(Q_a_s, dim=1).mean() - Q_a_s.mean()

        bellmann_error = F.mse_loss(Q_expected, Q_targets)

        q1_loss = cql1_loss + 0.5 * bellmann_error

        q1_loss.backward()
        clip_grad_norm_(self.network.parameters(), 1)
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.network, self.target_net)
        return q1_loss.detach().item(), cql1_loss.detach().item(), bellmann_error.detach().item()


    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)
