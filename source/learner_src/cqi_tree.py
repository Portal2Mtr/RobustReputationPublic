"""
Based on Conservative Q-improvement decision tree

"""

import copy
import random
from copy import deepcopy
import networkx as nx
import numpy as np
from statistics import mean
import logging


class ConsDTLearner(object):
    """
    Decision tree reinforcement learner based on: https://arxiv.org/abs/1907.01180
    """

    def __init__(self, env_state, sim_conf):
        """
        Create DT learner and setup learning parameters for bandit algorithm
        :param env_state: Initial state of the environment
        :param sim_conf: Simulation config file
        """

        # Binary tree parameters
        self.init_split_thresh = sim_conf['initThresh']
        self.hs = self.init_split_thresh
        self.split_decay = sim_conf['splitDecay']
        self.visit_decay = sim_conf['visitDecay']  # Decay for split
        self.tree = nx.Graph()
        self.steps_done = 0

        # State vectors for criterion generation
        self.min_state_vector = [0] * len(env_state)
        self.max_state_vector = [0] * len(env_state)
        self.num_env_feat = len(env_state)
        self.num_actions = sim_conf['num_actions']

        # Random dimension to start decision making
        self.max_q_vals = sim_conf['maxQVals']
        self.tree.add_node(0,
                           name="root",
                           qvalues=[[0.0] for _ in range(self.num_actions)],
                           rewvalues=[[] for _ in range(self.num_actions)],
                           prevState=0,
                           qMax=0,
                           children=[None, None],
                           parent=None,
                           type='Leaf',
                           history=[],
                           critVec=[0.5] * self.num_env_feat,
                           criterion=[0, 0],
                           visitFreq=0,
                           splits=NodeSplits(self.visit_decay, self.num_env_feat, self.num_actions,self.max_q_vals),
                           depth=0,
                           numVisits=1,
                           bandAlpha=[1] * self.num_actions,
                           bandBeta=[1] * self.num_actions,
                           bandCnt=0,
                           freqArms=[0 for _ in range(self.num_actions)])

        self.leaves = self.tree.subgraph(0)
        self.branches = self.tree.subgraph(0)

        # Q Learning parameters
        self.learn_alpha = sim_conf['learnAlpha']
        self.gamma = sim_conf['gamma']  # Bias towards next state values

        self.eps = sim_conf['eps']
        self.eps_decay = sim_conf['eps_decay']
        self.eps_min = sim_conf['min_eps']

        # Bandit algo parameters
        self.K = sim_conf['numActions']
        self.max_tree_depth = sim_conf['maxTreeDepth']
        self.trim_visit_perc = sim_conf['trimVisitPerc']
        self.bandit_batch = sim_conf['banditbatch']
        self.bandit_cnt = 0

        #Temporary ledger storage
        self.ledger = None
        self.intelAnalysis = False
        self.malc_trxns = []
        self.flagAttacks = False

        # Reputation system parameters
        # Reputation variables
        self.minqvalues = [np.inf for _ in range(self.num_actions)]
        self.maxqvalues = [0.0 for _ in range(self.num_actions)]
        self.self_idx = sim_conf['self_idx']
        self.q_decay = sim_conf['q_decay']
        self.retqvalues = [0.0 for _ in range(self.num_actions)]
        self.maxdecays = [0.0 for _ in range(self.num_actions)]
        self.inner_learn = sim_conf['inner_learn']
        self.var_est = 0.0

    def take_action(self, state):
        """
        Main bandit algorithm for selecting actions
        :param state: Current environment state
        :return: Action selection
        """
        # Using modified Bernoulli TS for bandit algorithm with filtered
        curr_state = self.tree.nodes[state]

        eps_dec = np.random.uniform(0, 1)

        q_values = curr_state['qvalues']
        if eps_dec < self.eps:
            # Random exploration
            action = random.choice([i for i in range(self.num_actions)])
            self.eps = max(self.eps - self.eps_decay, self.eps_min)
        else:
            # Exploitation
            lastvals = [i[-1] for i in q_values]
            try:
                action = np.argmax(lastvals)
            except:
                print("Encountered cqi error choosing random, lastvals: {}".format(lastvals))
                action = random.choice([i for i in range(self.num_actions)])
                self.eps = max(self.eps - self.eps_decay, self.eps_min)

        # Update number of visits
        curr_state['numVisits'] += 1
        curr_state['numVisits'] += 1
        return action

    def update_q_values(self, current_state, reward, action):
        """
        Updates the q values based on maximum reward for the decision tree node
        :param current_state:
        :param reward:
        :param action:
        :return:
        """

        curr_state = self.get_leaf(current_state)
        # Filter reward values with FilteredMean() approach
        temp_list = copy.deepcopy(self.tree.nodes[curr_state]['rewvalues'][action])
        temp_list.append(reward)
        self.tree.nodes[curr_state]['rewvalues'][action].append(reward)
        qvalues = self.tree.nodes[curr_state]['qvalues']
        if len(qvalues[action]) == 0:
            last_q_value = 0
            max_curr_reward = 0
        else:
            last_q_value = qvalues[action][-1]
            max_curr_reward = max(qvalues[action])

        # Add decay for reputation system
        new_q = (1 - self.learn_alpha) * last_q_value + self.learn_alpha * (reward + self.gamma * max_curr_reward)

        minqval = np.inf
        maxqval = -np.inf
        for i in range(len(self.tree.nodes[curr_state]['qvalues'])):
            minqval = min(minqval, self.tree.nodes[curr_state]['qvalues'][i][-1])
            maxqval = max(maxqval, self.tree.nodes[curr_state]['qvalues'][i][-1])

        try:
            self.retqvalues[i] = (self.tree.nodes[curr_state]['qvalues'][action][-1] - minqval)/(maxqval-minqval)
        except ZeroDivisionError:
            self.retqvalues[i] = 0.0

        if len(self.tree.nodes[curr_state]['qvalues'][action]) > self.max_q_vals:
            self.tree.nodes[curr_state]['qvalues'][action].pop(0)
            self.tree.nodes[curr_state]['qvalues'][action].append(new_q)
        else:
            self.tree.nodes[curr_state]['qvalues'][action].append(new_q)

        return new_q

    def update_crits(self, input_vec):
        """
        Update the decision criterion for each branch in the tree based on new info.
        :param input_vec: Most recent leaf state
        :return:
        """
        for idx, feature in enumerate(input_vec):
            self.min_state_vector[idx] = min(self.min_state_vector[idx], feature)
            self.max_state_vector[idx] = max(self.max_state_vector[idx], feature)

        # Update the criterion for each branch node
        for node, data in self.branches.nodes(data=True):
            criterion = data['criterion']
            self.branches.nodes[node]['criterion'][1] = \
                (self.max_state_vector[criterion[0]] - self.min_state_vector[criterion[0]]) * \
                data['critVec'][criterion[0]]

    def update_visit_freq(self, leaf_state):
        """
        Updates decision tree node visit frequency for node splitting
        :param leaf_state: Node to update
        :return: None
        """

        while self.tree.nodes[leaf_state]['name'] != 'root':
            self.tree.nodes[leaf_state]['visitFreq'] = self.tree.nodes[leaf_state]['visitFreq'] * \
                                                       self.visit_decay + (1 - self.visit_decay)
            parent = self.tree.nodes[leaf_state]['parent']
            if self.tree.nodes[parent]['children'][0] is not None:
                for child in self.tree.nodes[parent]['children']:
                    if child != leaf_state:
                        # Update sibling in binary tree
                        self.tree.nodes[child]['visitFreq'] = self.tree.nodes[child]['visitFreq'] * self.visit_decay
                        break
            leaf_state = parent

        # Update root
        self.tree.nodes[leaf_state]['visitFreq'] = self.tree.nodes[leaf_state]['visitFreq'] * \
                                                   self.visit_decay + (1 - self.visit_decay)

    def get_leaf(self, state):
        """
        Traverses binary decision tree to find state representing current ledger.
        :param state: Environment state
        :return: Leaf index
        """

        curr_leaf = 0
        while self.tree.nodes[curr_leaf]['type'] != 'Leaf':
            curr_crit = self.tree.nodes[curr_leaf]['criterion']
            if state[curr_crit[0]] < curr_crit[1]:
                # Traverse left node to get appropriate state space
                curr_leaf = self.tree.nodes[curr_leaf]['children'][0]
            else:
                # Traverse right node to get appropriate state space
                curr_leaf = self.tree.nodes[curr_leaf]['children'][1]
        return curr_leaf

    def get_max_depth(self):
        """
        Get max depth for current tree
        :return:
        """
        max_depth = 0
        for idx, data in self.leaves.nodes(data=True):
            max_depth = max(max_depth, data['depth'])
        return max_depth


    def update_poss_splits(self, leaf_state, env_state, action, q_change):
        """
        Updates the possible node splits for a given leaf node (conservative q improvement
        :param self: Decision tree object
        :param leaf_state: Current node in decision tree
        :param env_state: Last environment state
        :param action: Last action chosen
        :param q_change: Q value change
        :return: None
        """

        curr_node = self.tree.nodes[leaf_state]
        crit_vals = []
        for min_val, max_val, multi in zip(self.min_state_vector, self.max_state_vector, curr_node['critVec']):
            crit_vals.append((max_val-min_val) * multi)

        # Update Q Values and visits
        self.tree.nodes[leaf_state]['splits'].update_q_val(env_state, q_change, action, crit_vals)


    def best_split(self, leaf_state, action):
        """
        Finds the best split for a given tree leaf node.
        :param self: Decision tree object
        :param leaf_state: Current leaf state in tree
        :param action: Most recent action
        :return: [The best split feature, best split value]
        """

        # Get product of all visit vals from leaf to root
        curr_node = leaf_state
        visit_prod = self.tree.nodes[curr_node]['visitFreq']
        while self.tree.nodes[curr_node]['name'] != 'root':
            curr_node = self.tree.nodes[curr_node]['parent']
            visit_prod *= self.tree.nodes[curr_node]['visitFreq']

        curr_node = self.tree.nodes[leaf_state]
        sq_vals = []  # Mapping of split to split values

        leftsplits = np.array([i['left'] for i in curr_node['splits'].split_q_tables])
        rightsplits = np.array([i['right'] for i in curr_node['splits'].split_q_tables])
        leftmaxs = np.amax(leftsplits,axis=(0,-1))
        rightmaxs = np.amax(rightsplits,axis=(0,-1))

        for idx, (left, right) in enumerate(zip(leftmaxs,rightmaxs)):

            curr_left = left - abs(curr_node['qvalues'][action][-1])
            curr_right = right - abs(curr_node['qvalues'][action][-1])
            sq_vals.append(visit_prod * (curr_left * curr_node['splits'].split_q_tables[idx]['leftVisit'] +
                                         curr_right * curr_node['splits'].split_q_tables[idx]['rightVisit']))

        best_split_idx = np.argmax(sq_vals)
        best_val = sq_vals[best_split_idx]
        return best_split_idx, best_val


    def check_perform_split(self, leaf_state, best_split_idx, best_value):
        """
        Decide tree split point based on input data distribution.
        :param self: Decision tree object
        :param leaf_state: Current leaf state
        :param best_split_idx: Best feature for splitting
        :param best_value: Best split value
        :return: None
        """

        if best_value > self.hs:
            leaf_node = self.tree.nodes[leaf_state]
            if leaf_node['depth'] >= self.max_tree_depth:
                return  # Skip leaf generation if reached max depth
            self.perform_split(leaf_state, best_split_idx)
            # Reset split threshold to slow growth
            self.hs = self.init_split_thresh

        else:
            self.hs = self.hs * self.visit_decay


    def perform_split(self, leaf_state, best_split_idx):
        """
        Performs split in decision tree and creates new children nodes
        :param self: Decision tree object
        :param leaf_state: Current leaf state idx
        :param best_split_idx: Best split feature idx
        :return: None
        """
        # Create children and new branch node
        crit_val = (self.max_state_vector[best_split_idx] - self.min_state_vector[best_split_idx]) * \
                   self.tree.nodes[leaf_state]['critVec'][best_split_idx]

        self.tree.nodes[leaf_state]['criterion'] = [best_split_idx, crit_val]
        self.tree.nodes[leaf_state]['type'] = 'Branch'
        left_num = max(self.tree.nodes) + 1
        right_num = max(self.tree.nodes) + 2
        self.tree.nodes[leaf_state]['children'] = [left_num, right_num]
        parent_depth = self.tree.nodes[leaf_state]['depth']
        parent_visits = self.tree.nodes[leaf_state]['numVisits']
        parent_alpha = self.tree.nodes[leaf_state]['bandAlpha']
        parent_beta = self.tree.nodes[leaf_state]['bandBeta']
        parent_rewards = self.tree.nodes[leaf_state]['rewvalues']
        split_dict = self.tree.nodes[leaf_state]['splits'].split_q_tables[best_split_idx]

        base_vec = self.tree.nodes[leaf_state]['critVec']
        left_crit_vec = deepcopy(base_vec)
        left_crit_vec[best_split_idx] += -left_crit_vec[best_split_idx] / 2
        right_crit_vec = deepcopy(base_vec)
        right_crit_vec[best_split_idx] += right_crit_vec[best_split_idx] / 2

        self.tree.add_node(left_num,
                           name=str(left_num),
                           qvalues=[[0.0] for _ in range(self.num_actions)],
                           rewvalues=copy.deepcopy(parent_rewards),
                           prevState=0,
                           qMax=0,
                           children=[None, None],
                           parent=leaf_state,
                           type='Leaf',
                           critVec=left_crit_vec,
                           criterion=[0, 0],
                           visitFreq=split_dict['leftVisit'],
                           splits=NodeSplits(self.visit_decay, self.num_env_feat, self.num_actions,self.max_q_vals),
                           depth=parent_depth+1,
                           numVisits=parent_visits,
                           bandAlpha=parent_alpha,  # Used parent's learned experience
                           bandBeta=parent_beta,
                           bandCnt=0,
                           freqArms=[0 for _ in range(self.num_actions)])

        self.tree.add_node(right_num,
                           name=str(right_num),
                           qvalues=[[0.0] for _ in range(self.num_actions)],
                           rewvalues=copy.deepcopy(parent_rewards),
                           prevState=0,
                           qMax=0,
                           children=[None, None],
                           parent=leaf_state,
                           type='Leaf',
                           critVec=right_crit_vec,
                           criterion=[0, 0],
                           visitFreq=split_dict['rightVisit'],
                           splits=NodeSplits(self.visit_decay, self.num_env_feat, self.num_actions,self.max_q_vals),
                           depth=parent_depth+1,
                           numVisits=parent_visits,
                           bandAlpha=parent_alpha,  # Used parent's learned experience
                           bandBeta=parent_beta,
                           bandCnt=0,
                           freqArms=[0 for _ in range(self.num_actions)])

        self.tree.add_edge(leaf_state, left_num)
        self.tree.add_edge(leaf_state, right_num)
        self.update_leaf_branch()


    def update_leaf_branch(self):
        """
        Updates the leaf and branch graphs when nodes values are changed
        :param self:
        :return:
        """
        self.leaves = self.tree.subgraph([node for node, data in self.tree.nodes(data=True) if data['type'] == 'Leaf'])
        self.branches = self.tree.subgraph([node for node, data in self.tree.nodes(data=True) if data['type'] == 'Branch'])


    def both_child(self, parent_node):
        """
        Checks if both children in DT are leaves
        :param self: Decision tree object
        :param parent_node: Data for current parent node of both children
        :return:
        """
        left_child, right_child = parent_node['children']
        left_true = self.tree.nodes[left_child]['type'] == 'Leaf'
        right_true = self.tree.nodes[right_child]['type'] == 'Leaf'
        return left_true and right_true


    def prune_tree(self):
        """
        Prunes the decision tree to reduce tree bloat
        :param self: Decision tree object
        :return:
        """

        did_prune = True
        while did_prune:
            did_prune = False
            # Find branch node from leaves that is not visited
            for name, leaf in self.leaves.nodes(data=True):
                leaf_parent = leaf['parent']
                if leaf_parent is None:
                    did_prune = False
                    break
                parent_node = self.tree.nodes[leaf_parent]
                if self.both_child(parent_node):  # Check that children are leaves
                    if parent_node['visitFreq'] <= self.trim_visit_perc:
                        # Remove leaf and parent nodes
                        left_child, right_child = parent_node['children']
                        self.tree.remove_node(left_child)
                        self.tree.remove_node(right_child)
                        parent_node['type'] = 'Leaf'
                        parent_node['children'] = [None, None]
                        did_prune = True
                        break
                    else:
                        # Nodes have enough visits that they do not need to be trimmed
                        continue

            if did_prune:
                self.update_leaf_branch()

        return


"""
Decision tree node splitting class

Management class for handling node splits in the decision tree
"""

class NodeSplits(object):
    """
    Node split handling management class for conservative Q learning
    """

    def __init__(self, visit_decay, num_feat, num_actions,maxnum):
        """
        Initialize split object
        :param visit_decay: Decay value for reducing node split decay
        :param num_feat: Number of environment features
        :param num_actions: Number of actions available
        """

        self.splits = []
        act_dict = np.array([[0 for _ in range(maxnum)] for _ in range(num_actions)])
        left_dict = copy.copy(act_dict)
        right_dict = copy.copy(act_dict)

        self.split_q_tables = []
        for _ in range(num_feat):
            self.split_q_tables.append(copy.copy({
                'left': left_dict,
                'right': right_dict,
                'leftVisit': 0,
                'rightVisit': 0}
            ))
        self.best_split = 0
        self.visit_decay = visit_decay

    def get_best_splits(self):
        """
        Returns the best splits to grow the decision tree
        :return: Two lists with [Q,visitFreq],[Q,visitFreq]
        """

        lmax = 0
        left_visit = 0
        reward_max = 0
        right_visit = 0
        for split in self.split_q_tables:

            if split['left'] > lmax:
                lmax = split['left']
                left_visit = split['leftVisit']

            if split['right'] > reward_max:
                reward_max = split['right']
                right_visit = split['rightVisit']

        left_split = [lmax, left_visit]
        right_split = [reward_max, right_visit]

        return left_split, right_split

    def update_q_val(self, env_state, q_change, action, crit_vec):
        """
        Updates the q values for each potential side of the split table
        :param env_state: Last environment state
        :param q_change: Change in q value from bellman equation
        :param action: Last action
        :param crit_vec: Criterion vector for current decision tree node
        :return: None
        """

        for idx, dimension in enumerate(env_state):
            if dimension < crit_vec[idx]:
                # Update left side of split
                self.split_q_tables[idx]['left'][action][:-1] = self.split_q_tables[idx]['left'][action][1:]
                self.split_q_tables[idx]['left'][action][-1] = q_change
                self.split_q_tables[idx]['leftVisit'] = self.split_q_tables[idx]['leftVisit'] * self.visit_decay \
                                                        + (1 - self.visit_decay)
            else:
                # Update right side of split
                self.split_q_tables[idx]['right'][action][:-1] = self.split_q_tables[idx]['right'][action][1:]
                self.split_q_tables[idx]['right'][action][-1] = q_change
                self.split_q_tables[idx]['rightVisit'] = self.split_q_tables[idx]['rightVisit'] * self.visit_decay \
                                                         + (1 - self.visit_decay)

