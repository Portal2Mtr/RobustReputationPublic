"""
Simulation of Proof of Reputation (PoR) from https://link.springer.com/chapter/10.1007/978-3-319-91458-9_41

Ouputs a pickle file with scaled reputation values in ../output for plotting in ./plotting.py
"""

import random
import pickle
from statistics import mean
import numpy as np
import yaml


class PoRNode:

    def __init__(self, config, node_id, att_cut, num_nodes, cond_attack):

        self.config = config['nodeparams']
        self.max_nodes = num_nodes
        self.node_idx = node_id
        self.act_log = []
        self.malc_perc = self.config['malcperc']
        self.responses = {'good': 1.0, 'general': 0.0, 'bad': -1.0}

        if self.node_idx > att_cut and cond_attack:
            self.type = "HONEST"
            self.good_betas = self.config['honestbetas']['good']
            self.gen_betas = self.config['honestbetas']['general']
            self.bad_betas = self.config['honestbetas']['bad']
        else:
            self.type = "ATTACK"
            self.good_betas = self.config['malcbetas']['good']
            self.gen_betas = self.config['malcbetas']['general']
            self.bad_betas = self.config['malcbetas']['bad']
            self.att_betas = self.config['malcbetas']['badattack']

    def make_request(self, service_node):
        """
        This node makes a request to other service node following PoR paper diagram.
        :param service_node: Node providing service and service quality
        :return:
        """

        # This node is a rater
        response = service_node.give_service()
        service_node.act_log.append(self.responses[response])

        return {'rater': self.node_idx, 'provider': service_node.node_idx, 'response': response}

    def give_service(self):
        """
        Generates service quality for forming reputation to request node. ONOFF attack performed here
        :return: 'good', 'general' or 'bad'
        """

        # This node responds to requests, may choose to attack based on beta distribution
        responsevals  = [
            np.random.beta(self.good_betas[0], self.good_betas[1]),
            np.random.beta(self.gen_betas[0], self.gen_betas[1]),
            np.random.beta(self.bad_betas[0], self.bad_betas[1])]

        if self.type == "ATTACK":
            att_chance = np.random.random()
            if att_chance < self.malc_perc:  # Chance to attack to obscure behavior
                responsevals[2] = np.random.beta(self.att_betas[0], self.att_betas[1])

        max_idx = np.argmax(responsevals)
        if max_idx == 0:
            return 'good'
        elif max_idx == 1:
            return 'general'
        else:  # maxidx  == 2
            return 'bad'


class PoRSim:

    def __init__(self, config):

        self.config = config['simparams']
        random.seed(self.config['seed'])
        np.random.seed(self.config['seed'])
        self.n = self.config['n']
        self.T = self.config['T']
        self.seed = self.config['seed']
        self.nodes = []
        self.block_hist = []
        self.rep_record = [0.0 for _ in range(self.n)]
        self.rep_metric = []
        self.phi = self.config['phi']
        self.num_sum = self.config['numsum']
        if self.config['netattacktype'] == 'SYBIL':
            self.config['attackperc'] = 2 * self.config['attackperc']  # Double malc nodes
        self.att_cut = int((1 - self.config['attackperc']) * self.n)
        self.lambda_val = self.config['lambdaval']
        # Generate nodes
        self.nodes.extend([PoRNode(config,
                                   i,
                                   self.att_cut, self.n,
                                   self.config['nodeattacktype'] == 'ONOFF')
                           for i in range(self.n)])

    def run_sim(self):
        """
        Main simulation function.
        :return:
        """

        idx_vector = [i for i in range(self.n)]

        for t in range(self.T):

            new_trxns = []

            for i in range(1, self.lambda_val):
                # Randomly select request and provider
                node_idxs = np.random.choice(idx_vector, 2, replace=False)
                request_idx = node_idxs[0]
                provide_idx = node_idxs[1]
                trxn = self.nodes[request_idx].make_request(self.nodes[provide_idx])
                new_trxns.append(trxn)

            if t % 100 == 0:
                print("Passed Epoch {}/{}".format(t, self.T))

            self.block_hist.append(new_trxns)  # Add transactions for reference (not used again)
            self.record_metric()  # Calculate reputation and update log

    def record_metric(self):
        """
        Record mean reputation for adversarial nodes for logging.
        :return:
        """

        # Update trustworthiness of each node
        for i in range(self.n):
            if len(self.nodes[i].act_log) == 0:
                self.rep_record[i] = 0.5
            else:
                act_sum = sum(self.nodes[i].act_log[-self.num_sum:])
                self.rep_record[i] = 1 / (1 + self.phi * np.exp(-act_sum))

        rep_vals = [self.rep_record[i] for i in range(self.n)]
        min_val = min(rep_vals)
        max_val = max(rep_vals)
        rep_vals = [(rep_vals[i] - min_val)/(max_val-min_val) for i in range(self.n)]
        self.rep_metric.append(mean([rep_vals[i] for i in range(self.att_cut, self.n)]))

    def save_results(self, seed):
        """
        Export reputation.
        :param seed:
        :return:
        """
        save_file = "../output/POR_results_{}_{}_{}.p".format(
            self.config['netattacktype'],
            self.config['nodeattacktype'],
            seed)
        pickle.dump(self.rep_metric, open(save_file, 'wb'))


if __name__ == "__main__":

    # Run PoR consensus for 10 trials
    max_seed = 10
    seeds = [i for i in range(max_seed)]

    net_attacks = ['DATA', 'SYBIL']
    node_attacks = ['CONTROL', 'ONOFF']

    run_all_setups = False  # Change to run all simulation setups

    if run_all_setups:
        for net in net_attacks:
            for node in node_attacks:

                for seed in seeds:
                    fileConfig = "../config/porconfig.yml"
                    with open(fileConfig, "r") as read_file:
                        config = yaml.load(read_file, Loader=yaml.FullLoader)
                    config['simparams']['netattacktype'] = net
                    config['simparams']['nodeattacktype'] = node
                    config['simparams']['seed'] = seed
                    simObj = PoRSim(config)
                    simObj.run_sim()
                    simObj.save_results(seed)
                    print("Finished Trial {} for {}_{}...".format(seed+1,net,node))

    else:
        for seed in seeds:
            fileConfig = "../config/porconfig.yml"
            with open(fileConfig, "r") as read_file:
                config = yaml.load(read_file, Loader=yaml.FullLoader)
            config['simparams']['seed'] = seed
            simObj = PoRSim(config)
            simObj.run_sim()
            simObj.save_results(seed)
            print("Finished Trial {} for single study...".format(seed+1))
