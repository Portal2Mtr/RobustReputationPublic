
import copy
import torch
import numpy as np
import math
from statistics import mean

from learner_src.cqi_tree import ConsDTLearner

from learner_src.dqnconfig import DQNConfig
from learner_src.dqn import DQN
from learner_src.cql import CQLAgent
from learner_src.cql_buffer import ReplayBuffer


class InnerLearner(torch.nn.Module):

    def __init__(self, config, n_inputs, num_classes, att_type, hon_noise, mixed_attack):
        """
        Base learner for arbitrary inner worker problem (Classification, Regression, etc.)
        :param config:
        :param n_inputs:
        :param att_type:
        """
        super(InnerLearner, self).__init__()

        self.conf = config
        self.att_type = att_type
        self.n_inputs = n_inputs
        self.n_outputs = num_classes
        self.loss = torch.nn.MSELoss()
        self.layers = torch.nn.ModuleList()
        self.make_layers()
        self.mixed_attack = mixed_attack
        self.param_shapes = []
        self.inner_act = torch.relu
        self.output_act = torch.sigmoid
        self.optim = torch.optim.SGD(self.parameters(), lr=self.conf['sgdlearn'])
        for param in self.parameters():
            self.param_shapes.append(list(param.size()))

        total_size = 0
        for shape in self.param_shapes:
            if len(shape) > 1:
                total_size += shape[0]*shape[1]
            else:
                total_size += shape[0]

        self.grads_size = total_size
        self.weight_list = []
        # Attack params (uniform, mad)
        # Uniform
        self.unif_mag = self.conf['unifattmag']
        self.step_size = self.conf['step_size']
        self.beta_val = self.conf['beta_val']
        self.init_state = None
        self.adv_loss = torch.nn.KLDivLoss()
        self.hon_noise = hon_noise
        self.var_store = [0.0]
        self.max_var_samples = 10
        self.var_est = 0.0

    def make_layers(self):
        """
        Create inner layers
        :return:
        """

        layer_weights = [self.n_inputs]
        layer_weights.extend(self.conf['layer_nodes'])
        layer_weights.append(self.n_outputs)

        # Declare layer-wise weights and biases
        for i in range(len(layer_weights) - 1):
            self.layers.append(
                torch.nn.Linear(layer_weights[i], layer_weights[i+1])
            )

    def forward(self, x):
        """
        Forward pass of the network
        Args:
            x (): Data input

        Returns:

        """

        out = x
        for i in range(len(self.layers)):
            out = self.layers[i](out)
            if i != len(self.layers)-1:
                out = self.inner_act(out)

        y = self.output_act(out)

        return y

    def store_grads(self, update_loss, rep_val=None):
        """
        Store grads for weight sharing
        :return:
        """

        self.weight_list = []
        temp_list = []
        for layer in self.parameters():
            layer_weight = layer.data.tolist()
            if type(layer_weight[0]) == list:
                layer_list = [x for xs in layer_weight for x in xs]
            else:
                layer_list = layer_weight
            temp_list.extend(layer_list)

        if 'UNIFORM' in self.att_type:

            conduct_attack = False
            # Add uniform pertubation to gradient values
            if self.mixed_attack:  # Chance to attack if average self reputation is high

                choice = np.random.uniform(0.0, 1.0)
                if choice < rep_val:
                    conduct_attack = True

            if conduct_attack or not self.mixed_attack:
                for idx, val in enumerate(temp_list):
                    temp_list[idx] = val + float(np.random.uniform(0, 1) * self.unif_mag)

        if 'MAD' in self.att_type:

            conduct_attack = False
            # Add uniform perturbation to gradient values
            if self.mixed_attack:  # Chance to attack if average self reputation is high

                choice = np.random.uniform(0.0, 1.0)
                if choice < rep_val:
                    conduct_attack = True

            if conduct_attack or not self.mixed_attack:

                eps = np.random.uniform(0.0, 1.0)

                if self.init_state is None:
                    self.init_state = copy.deepcopy(temp_list)
                    self.init_state = torch.tensor(self.init_state, requires_grad=True)

                new_state = torch.tensor(copy.deepcopy(temp_list), requires_grad=True)
                loss = -self.adv_loss(new_state, self.init_state)

                dist_grads = torch.autograd.grad(loss, new_state, retain_graph=False)[0]

                dist_grads = dist_grads.tolist()

                perturb = [i + math.sqrt(2.0/(self.beta_val*self.step_size)*eps) for i in dist_grads]

                # Use infinite norm
                perturb = [np.sign(i) for i in perturb]
                temp_list = [i - self.step_size*j for i, j in zip(temp_list, perturb)]
                temp_list = [max(i - 3*self.hon_noise, min(i, i+3*self.hon_noise)) for i in temp_list]

        self.weight_list = temp_list
        self.weight_list.append(update_loss.item())

    def update_weights(self, dist_weights):
        """
        Update weights with new biased weights
        :param dist_weights:
        :return:
        """
        with torch.no_grad():

            for idx, layer in enumerate(self.parameters()):
                layer.data = dist_weights[idx]

    def backward(self, y_true, y_pred, rep_val=None):
        """
        Custom backward to store gradients for sharing.
        :return:
        """

        loss = self.loss(y_true, y_pred)
        loss.backward(retain_graph=True)

        with torch.no_grad():
            data = []
            test = []
            for param in self.parameters():
                test.append(param)
                if len(param.grad.shape) == 1:
                    data.extend(param.grad.tolist())
                else:
                    for sublist in param.grad.tolist():
                        data.extend(sublist)

            var_est = float(np.var(np.array(data)))

        self.var_store.append(var_est)
        if len(self.var_store) > self.max_var_samples:
            self.var_store.pop(0)

        self.var_est = mean(self.var_store)
        self.store_grads(loss, rep_val)

# Conservative Q-Improvement

# cqiconfig = DQNConfig()
# cqiconfig.seed = 0
# cqiconfig.environment = None
# cqiconfig.num_episodes_to_run = 450
# cqiconfig.show_solution_score = False
# cqiconfig.visualise_individual_results = False
# cqiconfig.visualise_overall_agent_results = True
# cqiconfig.standard_deviation_results = 1.0
# cqiconfig.runs_per_agent = 1
# cqiconfig.use_GPU = False
# cqiconfig.overwrite_existing_results_file = False
# cqiconfig.randomise_random_seed = True
# cqiconfig.save_model = False
# cqiconfig.hyperparameters = {
#     "CQI_Agents": {
#         'initThresh': 1,
#         'splitDecay': 0.999,
#         'visitDecay': 0.9,
#         'gamma': 0.99,
#         'learnAlpha': 0.7,
#         'learnDecay': 0.001,
#         'numEpisodes': 20,
#         'lambdaRange': [5, 50],
#         'numActions': 0,
#         'maxTreeDepth': 5,
#         'trimVisitPerc': 0.1,
#         'maxQVals': 5,
#         'banditbatch': 30,
#         'self_idx': None,
#         'eps': 1.0,
#         'eps_decay': 0.001,
#         'min_eps': 0.01,
#         'q_decay': 0.99,
#         'inner_learn': 0.7
#
#     },
# }
#
#
# class CQIWrapper:
#
#     def __init__(self, env, self_idx):
#         self.env = env
#         self.conf = cqiconfig.hyperparameters['CQI_Agents']
#         self.conf['self_idx'] = self_idx
#         self.conf['num_actions'] = self.env.action_space.n
#         self.env = env
#         self.agent = ConsDTLearner(self.env.state_space.sample(), self.conf)
#         self.recent_action = None
#
#     def update_learn(self, isready):
#         """
#         Update CQI Tree
#         :param isready: Check if ready to learn
#         :return:
#         """
#         if not isready:
#             return
#
#         # Take actions until sim is complete
#         self.agent.learn_alpha = self.conf['learnAlpha']
#         # Select and perform an action
#         current_state = np.array(self.env.state_array[-1])
#         leaf_state = self.agent.get_leaf(current_state)
#         action = self.agent.take_action(leaf_state)
#         self.recent_action = action
#         next_state, reward, done = self.env.step(action)
#
#         # Observe new state
#         last_state = current_state
#         current_state = next_state
#         # Update q values and history
#         q_change = self.agent.update_q_values(current_state, reward, action)
#         self.agent.update_crits(last_state)
#         self.agent.update_visit_freq(leaf_state)
#         self.agent.update_poss_splits(leaf_state, last_state, action, q_change)
#         best_split, best_val = self.agent.best_split(leaf_state, action)
#         self.agent.check_perform_split(leaf_state, best_split, best_val)
#
#     def make_decision(self, var_est):
#         """
#         Make a decision based on available data.
#         :param var_est: Inner grad variance estimation
#         :return:
#         """
#
#         act_idx = self.recent_action
#         self.agent.var_est = var_est
#
#         return act_idx, self.agent.retqvalues[act_idx]


# DQN Config for DQN and DDQN
dqnconfig = DQNConfig()
dqnconfig.seed = 0
dqnconfig.environment = None
dqnconfig.num_episodes_to_run = 450
dqnconfig.show_solution_score = False
dqnconfig.visualise_individual_results = False
dqnconfig.visualise_overall_agent_results = True
dqnconfig.standard_deviation_results = 1.0
dqnconfig.runs_per_agent = 1
dqnconfig.use_GPU = False
dqnconfig.overwrite_existing_results_file = False
dqnconfig.randomise_random_seed = True
dqnconfig.save_model = False
dqnconfig.consensus_scheme = None
dqnconfig.consensus_attack = None
dqnconfig.hyperparameters = {
    "DQN_Agents": {
        "learning_rate": 0.001,
        "batch_size": 30,
        "buffer_size": 10000,
        "discount_rate": 0.99,
        "decay_rate": 0.99,
        "tau": 0.01,
        "alpha_prioritised_replay": 0.6,
        "beta_prioritised_replay": 0.1,
        "incremental_td_error": 1e-8,
        "update_every_n_steps": 1,
        "linear_hidden_units": [150, 50],
        "final_layer_activation": "None",
        "batch_norm": False,
        "gradient_clipping_norm": 0.7,
        "learning_iterations": 1,
        "clip_rewards": False,
        "self_idx": None,
        "inner_learn": 0.7
    }
}
dqnconfig.exploreparams = {
    "CONTROL":{
        "epsilon": 1.0,
        "epsilon_decay_rate_denominator": 1
    },
    "EPSGREEDY":{
        "epsilon": 1.0,
        "epsilon_decay_rate_denominator": 1
    },
    "MVEDQL":{
        "epsilon": 1.0,
        "epsilon_decay_rate_denominator": 1
    },
    "DBTS":{
        "scale": 50,
        "batch": 50,
        "self_idx": None,
        "update_every": 10,
        "nei_select": 2
    },
    "TS":{
        "scale": 50,
        "batch": 50,
        "self_idx": None,
        "update_every": 5,
        "nei_select": 2
    },
    "LQN":{
        "epsilon": 1.0,
        "epsilon_decay_rate_denominator": 1,
        "max_temp": 1.0,
        "mod_coeff": 2.0,
        "beta_start": 0.9,
        "beta_decay": 0.99,
        "beta_min": 0.4,
        "max_temp_decay": 0.999
    }
}

class DQNWrapper:
    def __init__(self, env,cons_scheme, self_idx,intent):
        """
        Wrapper class for DQN network api
        """

        agent_config = copy.deepcopy(dqnconfig)
        agent_config.environment = env
        agent_config.self_idx = self_idx
        agent_config.consensus_scheme = cons_scheme
        agent_config.consensus_intent = intent
        agent_config.hyperparameters = agent_config.hyperparameters['DQN_Agents']
        agent_config.exploreparams["DBTS"]["self_idx"] = self_idx
        agent_config.hyperparameters['batch_size'] = env.action_space.n
        self.agent = DQN(agent_config)

    def update_learn(self, is_ready):
        """
        Update outer learning variables.
        :param is_ready: Check if inner learner is ready.
        :return:
        """
        if not is_ready:
            return

        self.agent.state = np.array(self.agent.environment.state_array[-1])
        self.agent.step()

    def make_decision(self, var_est):
        """
        Make an intelligent decision based on available data.
        :param var_est: Inner gradient variance estimation.
        :return:
        """

        act_idx = self.agent.environment.recent_action
        self.agent.var_est = var_est

        return act_idx, self.agent.retqvalues[act_idx]


# class DDQNWrapper(DQNWrapper):
#
#     def __init__(self, env, self_idx):
#         super(DDQNWrapper, self).__init__(env,self_idx)
#
#
# cqlconfig = DQNConfig()
# cqlconfig.seed = 0
# cqlconfig.environment = None
# cqlconfig.num_episodes_to_run = 300
# cqlconfig.show_solution_score = False
# cqlconfig.visualise_individual_results = False
# cqlconfig.visualise_overall_agent_results = True
# cqlconfig.standard_deviation_results = 1.0
# cqlconfig.runs_per_agent = 1
# cqlconfig.use_GPU = False
# cqlconfig.overwrite_existing_results_file = False
# cqlconfig.randomise_random_seed = True
# cqlconfig.save_model = False
# cqlconfig.hyperparameters = {
#     "CQL_Agent": {
#         "learning_rate": 0.001,
#         "batch_size": 30,
#         "buffer_size": 10000,
#         "epsilon": 1.0,
#         "eps_frames": 1e3,
#         "min_eps": 0.01,
#         "discount_rate": 0.99,
#         "decay_rate": 0.99,
#         "tau": 0.001,
#         "alpha_prioritised_replay": 0.6,
#         "beta_prioritised_replay": 0.1,
#         "incremental_td_error": 1e-8,
#         "update_every_n_steps": 1,
#         "linear_hidden_units": [150, 50],
#         "final_layer_activation": "None",
#         "batch_norm": False,
#         "gradient_clipping_norm": 0.7,
#         "learning_iterations": 1,
#         "clip_rewards": False,
#         "self_idx": None,
#         "inner_learn": 0.7
#     },
# }


# class CQLWrapper:
#
#     def __init__(self, env, self_idx):
#
#         cqlconfig.hyperparameters['CQL_Agent']['self_idx'] = self_idx
#         cqlconfig.hyperparameters['CQL_Agent']['batch_size'] = env.action_space.n
#
#         self.agent = CQLAgent(cqlconfig,
#                               state_size=env.state_space.shape,
#                               action_size=env.action_space.n)
#
#
#         self.buff_size = cqlconfig.hyperparameters['CQL_Agent']['buffer_size']
#         self.batch_size = cqlconfig.hyperparameters['CQL_Agent']['batch_size']
#         self.eps = cqlconfig.hyperparameters['CQL_Agent']['epsilon']
#         self.eps_frames = cqlconfig.hyperparameters['CQL_Agent']['eps_frames']
#         self.min_eps = cqlconfig.hyperparameters['CQL_Agent']['min_eps']
#         self.steps = 0
#         self.buffer = ReplayBuffer(buffer_size=self.buff_size, batch_size=self.batch_size)
#         self.rewards = 0
#         self.d_eps = 1 - self.min_eps
#         self.episode_steps = 0
#         self.loss = []
#         self.cql_loss = []
#         self.env = env
#         self.state = None
#         self.recent_action = None
#
#     def update_learn(self, is_ready):
#         """
#         Update outer learning parameters
#         :param is_ready: Check if inner learner is ready to learn.
#         :return:
#         """
#
#         if not is_ready:
#             return
#
#         self.state = np.array(self.env.state_array[-1])
#         action = self.agent.get_action(self.state, epsilon=self.eps)
#         self.recent_action = action[0]
#         next_state, reward, done = self.env.step(action[0])
#         self.buffer.add(self.state, action, reward, next_state, done)
#
#         if len(self.buffer.memory) >= self.batch_size:
#             loss, cql_loss, bellman_error = self.agent.learn(self.buffer.sample())
#             self.loss.append(loss)
#             self.cql_loss.append(cql_loss)
#             self.state = next_state
#             self.rewards += reward
#             self.episode_steps += 1
#             self.eps = max(1 - ((self.steps*self.d_eps)/self.eps_frames), self.min_eps)
#
#     def make_decision(self, var_est):
#         """
#         Make an intelligent decision based on available data.
#         :param var_est: Inner learner variance estimation
#         :return:
#         """
#
#         act_idx = self.recent_action
#         self.agent.var_est = var_est
#
#         return act_idx, self.agent.retqvalues[act_idx]
