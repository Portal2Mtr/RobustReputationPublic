"""Sim.py

Main simulation functions for the secure reputation scheme.

"""

import torch
import random
import optuna
import pickle
import logging
import time
import numpy.random

from statistics import mean
from node import Node

logger = logging.getLogger(__name__)


class ReputationSim:

    def __init__(self, conf=None,trial=None):

        self.seed = conf['simparams']['seed']
        self.log_block = conf['simparams']['blockreplogging']
        random.seed(self.seed)
        numpy.random.seed(self.seed)
        torch.random.manual_seed(self.seed)

        # Organize configs
        self.s_conf = conf['simparams']
        self.n_conf = conf['nodeparams']
        # self.l_type = self.n_conf['agentSolver'] # Always DQN
        self.cons_type = conf['nodeparams']['consensus_scheme']

        # Sim parameters
        self.nodes = []
        self.T = self.s_conf['time']
        self.n = self.s_conf['n']

        self.att_perc = self.s_conf['attack']
        self.att_type = conf['nodeparams']['consensus_attack']
        if self.att_type == 'SYBIL':

            # Double the number of adversary nodes for Sybil attack
            self.att_perc = self.att_perc * 2
        self.att_cut = int((1 - self.att_perc) * self.n)

        # Initialize metrics to record for plots
        self.mean_inn_err = [0]
        self.mean_adv_rep = [0]
        self.mean_acc = [0]
        self.mean_hon_rep = [0]
        self.trial = trial

    def init_nodes(self, sim_round):
        """
        Initializes nodes for the simulation.
        :param sim_round: Time
        :return:
        """

        if sim_round == 1:
            for i in range(self.n):
                if i < self.att_cut:
                    node_type = "HONEST"
                else:
                    node_type = "ATTACK"
                self.nodes.append(Node(self.n_conf, node_type, self.n, self.att_cut, i, self.seed))

        # Get attack nodes reputation to decide if performing an attack
        if self.n_conf['agentSolver'] != 'CONTROL':
            att_rep_list = []
            for i in range(self.att_cut, self.n):
                att_rep_list.append(mean([self.nodes[j].reputation.rep_record[i] for j in range(self.att_cut)]))
        else:
            att_rep_list = []
            for i in range(self.att_cut, self.n):
                att_rep_list.append(0.0)

        rep_cnt = 0
        for i in range(self.n):
            if i < self.att_cut:
                self.nodes[i].create_new_data()
            else:
                self.nodes[i].create_new_data(att_rep_list[rep_cnt],self.att_cut)
                rep_cnt += 1

    def run_sim(self):
        """
        Main function to call for running the simulation.
        :return:
        """

        start = time.time()
        for i in range(1, self.T):
            self.init_nodes(i)

            sample_list = []

            for j in range(self.n):

                # Get data from neighbors
                neis = self.nodes[j].ping_data(i)
                new_samples = [[self.nodes[k].share_data, self.nodes[k].share_vector] for k in neis]
                sample_list.append(new_samples)
                self.nodes[j].update_data(new_samples,neis)

            self.log_metrics()
            if self.trial is not None:
                self.report_results(i)
                if self.trial.should_prune():
                    raise optuna.TrialPruned()

            if i % 10 == 0:
                logger.info("Finished epoch {} in {} sec ...".format(i, time.time() - start))
                start = time.time()

    def log_metrics(self):
        """
        Log the metrics from simulation for plotting.
        :return:
        """

        error_list = []
        reputation_list = []
        acc_list = []
        hon_list = []

        for i in range(self.att_cut):
            error_list.append(self.nodes[i].calc_data_err())
            reputation_list.append(self.nodes[i].get_adv_rep(self.att_cut))
            acc_list.append(self.nodes[i].acc)
            hon_list.append(self.nodes[i].get_hon_rep(self.att_cut))

        self.mean_inn_err.append(mean(error_list))
        self.mean_adv_rep.append(mean(reputation_list))
        self.mean_acc.append(mean(acc_list))
        self.mean_hon_rep.append(mean(hon_list))

    def save_results(self):
        """
        Output data from simulation for plotting.
        :return:
        """
        save_file = "../output/blockrepresultsCONSENSUS_{}_{}_{}.p".format(self.cons_type,self.att_type, self.seed)

        pickle.dump([self.mean_inn_err, self.mean_adv_rep, self.mean_acc,self.mean_hon_rep], open(save_file, 'wb'))

    def report_results(self, step, final=False):
        """
        Report results for optuna
        :return:
        """

        if not final:
            self.trial.report(self.mean_acc[-1], step)
        else:
            return self.mean_acc[-1]
