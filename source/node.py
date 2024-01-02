"""Node Class

Base node used for each communicating 'IoT' node in the network with secure reinforcement learning.

"""

import numpy
import random

import numpy as np
import torch.utils.data
import pandas as pd
from statistics import variance

from random import choice
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from _ethfraud import EthFraudDataset

from learners import *
from reputation import RepEnv

learnerObjs = {'CONTROL': None,  # No attack, no reputation
               # 'CQI': CQIWrapper,
               'DQN': DQNWrapper,
               # 'DDQN': DDQNWrapper,
               # 'CQL': CQLWrapper,
               'MULTIKRUM': None,
               'BRIDGE': None}


train_dict = {"ETHFRAUD": EthFraudDataset}



class Node:

    def __init__(self, conf, n_type, num_n, att_cut, n_idx, base_seed):

        random.seed(base_seed + n_idx)
        numpy.random.seed(base_seed + n_idx)
        torch.random.manual_seed(base_seed + n_idx)

        self.conf = conf
        self.n_type = n_type
        self.att_type = 'HONEST'
        self.mix_attack = False
        if self.n_type != "HONEST":

            att_type = self.conf['advAttack']
            if 'MIXED' in att_type:
                self.mix_attack = True
            self.att_type = att_type

        # Base and Control parameters
        self.mem_reg = []
        self.mem_data = []
        self.nei_reps = [0 for i in range(num_n)]
        self.nei_hist_mat = [[] for i in range(num_n)]
        self.nei_last_seen = [0 for i in range(num_n)]
        self.nei_times_comm = [0 for i in range(num_n)]
        self.nei_malc_hist = [0 for i in range(num_n)]
        self.nei_hon_hist = [0 for i in range(num_n)]
        self.share_data = None
        self.err_log = []
        self.num_ping = self.conf['B']
        self.ping_eps = self.conf['K']
        self.up_cnt = 0
        self.max_nei = num_n
        self.learn_type = self.conf['agentSolver']
        self.cons_type = self.conf['consensus_scheme']
        self.cons_intent = self.conf['consensus_attack']
        self.out_type = learnerObjs[self.learn_type]

        self.curr_pings = []
        self.n_nodes = num_n
        self.att_cut = att_cut
        self.att_data = []
        self.ignore_list = [n_idx]
        self.n_idx = n_idx

        # Dataset setup
        # Ethereum dataset
        working_data = train_dict[self.conf['train_file']]() # Load and parse dataset file
        self.training_data = working_data
        train_x, train_y = np.array([working_data.all_data[i] for i in working_data.train_idxs]), \
                           np.array([working_data.all_labels[i] for i in working_data.train_idxs])

        test_x, test_y = np.array([working_data.all_data[i] for i in working_data.train_idxs]), \
                         np.array([working_data.all_labels[i] for i in working_data.train_idxs])

        # train_perc = conf['innerParams']['trainperc']
        # iris_x, iris_y = datasets.load_iris(return_X_y=True)
        # scaler = MinMaxScaler()
        # scaler.fit(iris_x)
        # iris_x = scaler.transform(iris_x)
        # x_train, x_test, y_train, y_test = train_test_split(iris_x, iris_y, train_size=train_perc, random_state=0)
        x_train = train_x
        x_test = test_x
        y_train = train_y
        y_test = test_y
        dataset = [(i, j) for i, j in zip(x_train, y_train)]
        work_dataset = dataset
        x_train = [i[0] for i in work_dataset]
        y_train = [i[1] for i in work_dataset]
        num_classes = len(set(train_y))
        input_dim = len(x_train[0])
        self.input_dim = input_dim
        temp_train = x_train
        temp_test = x_test
        temp_label = y_train
        test_label = y_test
        train_data = []
        test_data = []
        for i in range(len(temp_train)):
            train_data.append([torch.tensor(temp_train[i]), torch.tensor(temp_label[i])])

        for i in range(len(temp_test)):
            test_data.append([torch.tensor(temp_test[i]), torch.tensor(test_label[i])])

        self.train_loader = torch.utils.data.DataLoader(train_data,
                                                        shuffle=True,
                                                        batch_size=len(train_data)//4)
        self.test_loader = torch.utils.data.DataLoader(test_data,
                                                       shuffle=True,
                                                       batch_size=len(test_data))

        # Define inner learner
        self.inn_learn = InnerLearner(conf['innerParams'],
                                      input_dim, num_classes,
                                      self.att_type,
                                      self.conf['gradnoisedev'],
                                      self.mix_attack)

        # Outer learner variables
        self.is_done = False
        self.local_vec = []
        if self.learn_type != 'CONTROL':
            self.reputation = RepEnv(self.n_nodes,
                                     self.inn_learn.grads_size,
                                     self.conf['numrepstore'],
                                     self.n_idx,
                                     self.conf['gradnoisedev'],
                                     conf['innerParams']['sgdlearn'])

        self.state_list = []
        self.grad_list = [[] for i in range(self.n_nodes)]
        self.nei_state_list = []
        if self.learn_type != 'CONTROL' and self.learn_type != 'MULTIKRUM' and self.learn_type != 'BRIDGE':
            self.out_learn = self.out_type(self.reputation,self.cons_type, self.n_idx,self.n_type)
        else:
            self.out_learn = None

        self.new_states = []

        # Other methods to compare
        self.krum_select = self.conf['krumselect']
        self.bridge_select = self.conf['bridgeselect']
        self.sum_cnt = [0 for _ in range(self.n_nodes)]
        self.max_sim = [0.0 for _ in range(self.n_nodes)]
        self.max_samples = self.conf['foolswait']
        self.acc = 0.0
        self.share_arm = None

    def create_new_data(self, rep_val=None,attack_start=None):
        """
        Creates a new data sample for the current node with gradient update.
        :param rep_val: Reputation value for if manipulating weights with adversary.
        :return:
        """

        # Add small noise to introduce additional synthetic datapoints
        stddev = 0.001
        mean = 0.0
        # Train on entire dataset then return gradients
        data, labels = next(iter(self.train_loader))
        data = data + torch.randn(data.size()) * stddev + mean
        y_pred = self.inn_learn.forward(data.float())
        # Calculate gradients and format for sharing
        self.inn_learn.backward(labels.float().view(-1, 1), y_pred.float(), rep_val)
        self.share_data = self.inn_learn.weight_list

        if self.cons_intent == 'GREEDY_UNIFORM' and self.n_type == "ATTACK":
            retvals = np.random.uniform(0.0,1.0,size=len(self.out_learn.agent.retqvalues))
            self.share_rep = retvals.tolist()
            self.share_arm = int(np.random.randint(0,9))
        elif self.cons_intent == 'GREEDY_MM' and self.n_type == "ATTACK":
            # Match variance of innocent nodes
            if self.out_learn.agent.exploration_strategy.attack_history:
                att_hist = [value for key,value in self.out_learn.agent.exploration_strategy.attack_history.items()]
                have_enough = True
                for att in att_hist:
                    if len(att) < 10:
                        have_enough = False
                if not have_enough:
                    retvals = np.random.uniform(0.0,1.0,size=len(self.out_learn.agent.retqvalues))
                    self.share_rep = retvals.tolist()
                    self.share_arm = int(np.random.randint(0,9))
                else:

                    recent_hist = [self.out_learn.agent.exploration_strategy.attack_history[i][-10:] for i in range(0, attack_start)]
                    # Gather arm info
                    node_info = []
                    num_arms = len(self.out_learn.agent.exploration_strategy.attack_history[0][0])
                    all_nodes_info = []
                    all_nodes_var = []
                    for i in range(len(recent_hist)):
                        node_info = []
                        var_arm_info = []
                        for k in range(num_arms):
                            node_arm_hist = [recent_hist[i][j][k] for j in range(len(recent_hist[i]))]
                            node_info.append(node_arm_hist)
                            var_arm_info.append(variance(node_arm_hist))

                        all_nodes_info.append(node_info)
                        all_nodes_var.append(var_arm_info)

                    # Calculate global measures for each arm
                    global_var_list = []
                    for k in range(num_arms):
                        temp = []
                        for i in range(len(recent_hist)):
                            temp.append(all_nodes_info[i][k])
                        temp_mean = np.mean(temp, axis=0).tolist()
                        global_var_list.append(variance(temp_mean))

                    # Find the best node to mimic
                    ideal_var = []
                    for i in range(len(all_nodes_var)):
                        ideal_var.append(sum([(all_nodes_var[i][j] - global_var_list[i]) for j in range(len(all_nodes_var[i]))]))

                    best_node = np.argmin(ideal_var)
                    working_est = self.out_learn.agent.exploration_strategy.attack_history[best_node][-1]
                    curr_est = [(1-working_est[i]) for i in range(len(working_est))]
                    self.share_rep = curr_est
                    self.mm_recc = int(np.argmax(curr_est))
                    self.share_arm = self.mm_recc
            else:
                retvals = np.random.uniform(0.0,1.0,size=len(self.out_learn.agent.retqvalues))
                self.share_rep = retvals.tolist()
                self.share_arm = int(np.random.randint(0,9))
        else:
            self.share_rep = self.out_learn.agent.retqvalues

        share_exp = self.out_learn.agent.share_exp

        if self.cons_intent == 'GREEDY_UNIFORM':

            if share_exp is not None:
                # Poison reward
                share_exp['reward'] = np.random.uniform(0.0,1.0)

        if self.cons_intent == 'GREEDY_MM':

            if share_exp is not None:
                # Poison reward
                share_exp['reward'] = self.share_rep[self.share_arm]

        self.share_vector = [self.share_rep, self.share_arm, share_exp]

        # Make ideal target based on local learning
        state_vec = []
        min_time = 999999
        min_com = 999999
        for i in range(self.n_nodes):
            min_time = min(min_time, self.nei_last_seen[i])
            min_com = min(min_com, self.nei_times_comm[i])
        state_vec.append(self.n_idx)
        state_vec.append(self.num_ping)
        state_vec.append(min_time)
        state_vec.append(min_com)
        state_vec.extend(self.inn_learn.weight_list)
        self.local_vec = np.array([state_vec], dtype=np.float32)

    def ping_data(self, time):
        """
        Select neighbors to ping for data.
        :param time: Used to form state space with node
        :return:
        """

        # Don't get data from self

        nei_idxs = [choice([i for i in range(0, self.max_nei) if i not in self.ignore_list])
                    for _ in range(self.num_ping)]
        self.curr_pings = nei_idxs
        for nei in nei_idxs:
            self.nei_last_seen[nei] = time - self.nei_last_seen[nei]
            self.nei_times_comm[nei] += 1
        return nei_idxs

    def clean_grad_list(self, new_list):
        """
        Restructure a gradient list into tensors for inner learner
        :param new_list:
        :return:
        """

        param_shapes = self.inn_learn.param_shapes
        update_list = []

        temp_cnt = 0
        for param in param_shapes:
            if len(param) > 1:

                param_list = []

                for i in range(param[0]):
                    temp_list = new_list[temp_cnt:temp_cnt + param[1]]
                    param_list.append(temp_list)
                    temp_cnt += param[1]

            else:
                param_list = []
                for j in range(temp_cnt, temp_cnt + param[0]):
                    param_list.append(new_list[j])

                temp_cnt += param[0]

            update_list.append(torch.tensor(param_list, dtype=torch.float32))

        return update_list

    def restruct_grads(self, weight_list):
        """
        Reconstruct gradients for updating weights
        :param weight_list: Raw weight value list
        :return: Cleaned weight list.
        """

        form_grads = []

        with torch.no_grad():
            for entry in weight_list:

                if len(entry) > 1:
                    sing_list = []
                    for sub_entry in entry:
                        sing_list.append(self.clean_grad_list(sub_entry))

                    mean_list = []
                    for i in range(len(sing_list[0])):
                        elements = [sing_list[j][i] for j in range(len(sing_list))]
                        mean_list.append(torch.mean(torch.stack(elements), 0))

                else:
                    mean_list = self.clean_grad_list(entry[0])

                form_grads.append(mean_list)

        return form_grads

    def dist_update(self, cont_nodes, decs):
        """
        Update local parameters with weighted average of distributed data
        :param cont_nodes: Contributing nodes
        :param decs: Decisions in [0,1]
        :return:
        """

        if self.learn_type == "CONTROL" or self.learn_type == 'MULTIKRUM' or self.learn_type == 'BRIDGE':
            # Update multikrum with gradient estimation
            new_grads = []
            for val in cont_nodes:
                new_grads.append(self.grad_list[val][:])

            # Restructure and average incoming grads from each node
            new_grads = self.restruct_grads(new_grads)
            new_grads.append(self.clean_grad_list(self.share_data))

            with torch.no_grad():

                sum_lit = []
                for idx in range(len(new_grads) - 1):
                    entry = new_grads[idx]
                    # Weigh weight updates equally (Q-learning has self reputation biased)
                    for jdx, element in enumerate(entry):
                        new_grads[idx][jdx] = torch.mul(new_grads[idx][jdx], decs[idx] / sum(decs))

                for jdx, element in enumerate(entry):
                    temp_list = []
                    for idx, entry in enumerate(new_grads):
                        temp_list.append(new_grads[idx][jdx])

                    sum_lit.append(torch.sum(torch.stack(temp_list), 0))

                self.inn_learn.update_weights(sum_lit)
                self.inn_learn.optim.step()
                for param in self.inn_learn.parameters():
                    param.grad = None

        else:
            # Get mean history from reputation memory
            recent_vec = self.reputation.state_record[cont_nodes]

            with torch.no_grad():

                recent_grad = recent_vec[4:-1]
                recent_grad = self.clean_grad_list(recent_grad)
                self_data = self.clean_grad_list(self.share_data)
                bias_grad = []

                for local_grad, new_grad in zip(self_data, recent_grad):
                    bias_grad.append(torch.add(torch.mul((2.0-decs)/2.0, local_grad), torch.mul(decs/2.0, new_grad)))

                self.inn_learn.update_weights(bias_grad)
                self.inn_learn.optim.step()
                for param in self.inn_learn.parameters():
                    param.grad = None

    def format_data(self, weight_samples):
        """
        Formats raw weight samples into weight space
        :param weight_samples: Raw weight samples from nodes
        :return:
        """

        self.nei_state_list = []
        # Format state vector (Num observ. nodes, Time since last seen node, times communicated, )
        for i in range(len(weight_samples)):
            state_vec = [self.curr_pings[i], self.num_ping, self.nei_last_seen[self.curr_pings[i]],
                         self.nei_times_comm[self.curr_pings[i]]]
            state_vec.extend(weight_samples[i])
            self.new_states.append(state_vec)

        self.nei_state_list.extend(self.curr_pings)

        return

    def update_data(self, samples, neis):
        """
        Update the local data log with data from neighbors.
        :param weight_samples: New samples from neighbors
        :return:
        """

        weight_samples = [i[0] for i in samples]
        rep_arm_samples = [i[1] for i in samples]
        rep_samples = [i[0] for i in rep_arm_samples]
        arm_samples = [i[1] for i in rep_arm_samples]
        new_exp_samples = [i[2] for i in rep_arm_samples]

        if self.cons_intent != 'GREEDY_UNIFORM' or self.cons_intent != 'GREEDY_MM':
            if self.out_learn.agent.cons_scheme != 'DBTS':
                for new_exp in new_exp_samples:
                    self.out_learn.agent.import_experience(new_exp)
                    if self.out_learn.agent.cons_scheme != 'MVEDQL':
                        if new_exp is not None:
                            self.out_learn.agent.temperature_vals.append(self.out_learn.agent.exploration_strategy.max_temp_thresh)

        rep_dict = {}
        for nei_idx, rep in zip(neis, rep_samples):
            rep_dict[nei_idx] = rep

        arm_dict = {}
        for nei_idx, rep in zip(neis, arm_samples):
            arm_dict[nei_idx] = rep

        self.new_states = []
        self.format_data(weight_samples)

        if self.learn_type != 'CONTROL':
            self.reputation.update_state(self.new_states, self.local_vec, self.is_done)

        for idx, val in enumerate(self.curr_pings):
            self.grad_list[val].append(weight_samples[idx])

        if self.learn_type == 'CONTROL':
            # Update gradients after getting samples from other nodes
            if self.up_cnt < self.ping_eps:
                self.up_cnt += 1
                self.inn_learn.optim.step()
                for param in self.inn_learn.parameters():
                    param.grad = None
                return

            self.up_cnt = 0
            dec_weights = [self.make_decision() for _ in range(self.n_nodes)]
            self.dist_update(set(self.nei_state_list), dec_weights)

        else:
            if not self.reputation.check_ready(self.learn_type =='BRIDGE' or self.learn_type =='MULTIKRUM') \
                    or self.up_cnt < self.ping_eps:
                self.up_cnt += 1
                self.inn_learn.optim.step()
                for param in self.inn_learn.parameters():
                    param.grad = None
                return

            self.share_arm = self.out_learn.agent.share_arm
            self.out_learn.agent.exploration_strategy.update_neighbors_rep(rep_dict)
            self.out_learn.agent.exploration_strategy.update_neighbors_arm(arm_dict)
            self_rep_dict = {self.n_idx: self.out_learn.agent.retqvalues}
            self.out_learn.agent.exploration_strategy.update_neighbors_rep(self_rep_dict)
            self_arm_dict = {self.n_idx: self.share_arm}
            self.out_learn.agent.exploration_strategy.update_neighbors_arm(self_arm_dict)
            self.out_learn.update_learn(self.reputation.check_ready())
            node_idx, rep_dec = self.make_decision()
            self.dist_update(node_idx, rep_dec)
            self.reputation.rep_record = self.out_learn.agent.retqvalues

    def weigh_samples(self, samples=None):
        """
        Weigh individual samples before updating local weights
        :param samples: Samples from nodes
        :return:
        """

        if self.learn_type == 'MULTIKRUM':

            decisions = [0.0 for _ in range(len(samples))]

            scores = []

            grads = [i[:-1] for i in samples]

            for idx in range(len(samples)):
                score = 0.0
                for jdx in range(len(samples)):
                    if jdx == idx:
                        continue
                    score += (np.linalg.norm([i-j for i, j in zip(grads[idx], grads[jdx])]) ** 2)
                scores.append(score)

            ind = np.argpartition(scores, -self.krum_select)[-self.krum_select:]
            for idx in range(len(decisions)):
                if idx in ind:
                    decisions[idx] = 1/(self.krum_select + 1)
                else:
                    decisions[idx] = 0.0

                decisions.append(1 / (self.krum_select + 1))

        else:  # Bridge

            # Find weight outliers from all sampled nodes and update weights
            b = self.bridge_select
            byzidx = []

            for i in range(b):
                byzidx.append(int(np.argsort(np.max(samples, axis=1))[-i]))
                byzidx.append(int(np.argsort(np.max(samples, axis=1))[i]))

            decs = []
            for i in range(len(samples)):
                if i in byzidx:
                    decs.append(0.0)
                else:
                    decs.append(1.0)

            decisions = decs
            decisions.append(1.0)

        return decisions

    def make_decision(self):
        """
        Wrapper for making a decision with various robustness techniques.
        :return:
        """

        if self.learn_type == "CONTROL":
            return 1.0  # Always accept data regardless
        else:

            self.reputation.var_est = self.inn_learn.var_est

            node_idx, rep_dec = self.out_learn.make_decision(self.inn_learn.var_est)

            return node_idx, rep_dec

    def calc_data_err(self):
        """
        Calculates learning error based on testing data for reporting.
        :return: Error value
        """

        with torch.no_grad():
            data, labels = next(iter(self.test_loader))
            _, y_pred = torch.max(self.inn_learn(data.float()), 1)
            error = self.inn_learn.loss(labels.float().view(-1, 1), y_pred.float())
            acc = accuracy_score(labels.float().view(-1, 1), y_pred.float())
            self.acc = acc

        return error.item()

    def get_adv_rep(self, adv_cut):
        """
        Get adversary reputation for what node believes are adversaries
        :param adv_cut: Adversary cutoff point from simulation.
        :return:
        """

        rep_list = []

        if self.learn_type != 'CONTROL' and self.learn_type != 'MULTIKRUM' and self.learn_type != 'BRIDGE':
            for j in range(adv_cut, self.n_nodes):
                rep_list.append(self.out_learn.agent.retqvalues[j])
        else:
            rep_list = [0.0 for _ in range(self.n_nodes)]

        return mean(rep_list)

    def get_hon_rep(self, adv_cut):
        """
        Get adversary reputation for what node believes are adversaries
        :param adv_cut: Adversary cutoff point from simulation.
        :return:
        """

        rep_list = []

        if self.learn_type != 'CONTROL' and self.learn_type != 'MULTIKRUM' and self.learn_type != 'BRIDGE':
            for j in range(0,adv_cut):
                rep_list.append(self.out_learn.agent.retqvalues[j])
        else:
            rep_list = [0.0 for _ in range(self.n_nodes)]

        return mean(rep_list)

