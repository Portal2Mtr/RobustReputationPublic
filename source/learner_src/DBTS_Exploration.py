from .Base_Exploration_Strategy import Base_Exploration_Strategy
import numpy as np
import random
import torch
import scipy

"""
Contains the novel Thompson Sampling exploration strategy to form machine-learning consensus

"""

class DBTS_Exploration(Base_Exploration_Strategy):
    """Implements an epsilon greedy exploration strategy"""
    def __init__(self, config,params):
        super().__init__(config,params)
        self.notified_that_exploration_turned_off = False
        self.exploration_cycle_episodes_length = None
        self.random_episodes_to_run = 0
        num_neighbors = 10
        self.max_select = num_neighbors - 1
        self.neighbors_rep_dict = {}
        self.neighbors_arm_dict = {}
        for i in range(num_neighbors):
            self.neighbors_rep_dict[i] = [0.0 for _ in range(num_neighbors)]

        self.alphas = [1 for i in range(num_neighbors)]
        self.betas = [1 for i in range(num_neighbors)]
        dbtsconfig = config.exploreparams['DBTS']
        self.scale = dbtsconfig['scale']
        self.batch = dbtsconfig['batch']
        self.batch_cnt = 0
        self.num_neis = num_neighbors
        self.nei_select = dbtsconfig['nei_select']
        self.nei_history = {}
        self.select = None
        self.n_idx = dbtsconfig['self_idx']

        self.trusted_updates = {}
        self.trusted_neis = [self.n_idx]

        self.attack_history = {}
        self.first_batch = False

        self.update_every = dbtsconfig['update_every']
        self.is_vanilla_ts = False

    def update_neighbors_rep(self, rep_dict):

        for key in rep_dict.keys():
            self.neighbors_rep_dict[key] = rep_dict[key]

            if key in self.nei_history.keys():
                self.nei_history[key].append(rep_dict[key])
            else:
                self.nei_history[key] = [rep_dict[key]]

            if key not in self.trusted_updates.keys():
                self.trusted_updates[key] = None

        if len(rep_dict.keys()) > 1:
            self.curr_neis = [i for i in rep_dict.keys()]

        for key in rep_dict.keys():

            if key not in self.attack_history.keys():
                self.attack_history[key] = []

            self.attack_history[key].append(rep_dict[key])





    def update_neighbors_arm(self,arm_dict):

        for key in arm_dict.keys():
            self.neighbors_arm_dict[key] = arm_dict[key]


    def perturb_action_for_exploration_purposes(self, action_info):
        """Perturbs the action of the agent to encourage exploration"""

        arm_samples = []
        for alpha,beta in zip(self.alphas,self.betas):
            arm_samples.append(np.random.beta(alpha, beta))

        act_select = np.argmax(arm_samples)
        self.select = act_select
        self.share_arm = int(act_select)

        return np.array(act_select)


    def update_ts(self,reward):

        max_rew = reward[self.select]

        if not self.is_vanilla_ts:
            self.alphas[self.select] += max_rew
            self.betas[self.select] += (1-max_rew)
        else:

            select = self.select

            nei_nodes = [i for i in self.nei_history.keys()]
            selects = []
            for nei in nei_nodes:
                selects.append(nei)
            reward_arm = []
            for nei in selects:
                if nei in self.nei_history.keys():
                    if len(self.nei_history[nei]):
                        temp = self.nei_history[nei][:]
                        rew_samples = [temp[i][select] for i in range(len(temp))]
                        reward_arm.extend(rew_samples)

            reward_arm.append(reward[self.select])
            mean_reward = np.mean(reward_arm)

            self.alphas[select] += mean_reward
            self.betas[select] += (1-mean_reward)

            self.batch_cnt += 1

            if self.batch_cnt > self.batch:
                self.reset_nei_hist()
                self.batch_cnt = 0


    def reset_nei_hist(self):

        self.nei_history = {}

    def custom_update_law(self, Q_target, Q_current):

        # Update thompson sampling

        self.batch_cnt += 1

        if len(self.curr_neis) == 0:
            self.reset_nei_hist()  # Vanilla TS
            return Q_target

        if self.batch_cnt < self.batch and not self.first_batch:
            return Q_target

        if self.batch_cnt < self.update_every and self.first_batch:
            return Q_target

        self.first_batch = True
        self.batch_cnt = 0
        if self.num_neis == 0:
            self.reset_nei_hist()  # Vanilla TS
            return Q_target

        if self.first_batch:
            for idx,update in enumerate(self.trusted_updates.values()):
                if update is not None:
                    arm = update[0]
                    work_hist = self.nei_history[self.n_idx]
                    self_mean = np.mean([work_hist[i][arm] for i in range(len(work_hist)-self.batch,len(work_hist))])
                    update_val = update[1]
                    self.alphas[arm] += update_val * self_mean
                    self.betas[arm] += (1-update_val) * self_mean

        nei_nodes = [i for i in range(self.num_neis) if i != self.n_idx]

        selects = []
        for nei in nei_nodes:
            selects.append(int(self.neighbors_arm_dict[nei]))

        for idx,arm_select in enumerate(selects):
            # Perform levenes test for each potential arm update
            # Gather known arm info
            nei_id = nei_nodes[idx]
            samples = []
            nei_compare = [i for i in self.trusted_neis if i != nei_id]
            nei_select = self.nei_select
            nei_nodes_select = list(np.random.randint(0, len(nei_compare), nei_select))
            actual_neis = [nei_compare[i] for i in nei_nodes_select]
            actual_neis.append(nei_id)
            for nei in actual_neis:
                temp = self.nei_history[nei]
                nei_arm_hist = [temp[i][arm_select] for i in range(len(temp)-self.batch,len(temp))]
                samples.append(nei_arm_hist)

            # Compare minimum of 2 neighbors, maximum of all neighbors
            result = scipy.stats.levene(*samples)

            result = result.statistic
            signif = 0.1

            batch_crit = scipy.stats.f.ppf(1-signif,nei_select-1,self.batch*nei_select - nei_select)

            passed_crit = False
            if result > batch_crit:
                passed_crit = True
            # else:

            if nei_id not in self.trusted_neis:
                self.trusted_neis.append(nei_id)
            # self.bandit.nei_select[idx] += 1
            # self.bandit.nei_select[idx] = min(self.bandit.nei_select[idx],self.bandit.max_select)
            temp = self.nei_history[self.n_idx]
            self_mean = np.mean([temp[i][arm_select] for i in range(len(temp)-self.batch,len(temp))])
            temp2 = self.nei_history[nei_nodes[idx]]
            node_mean = np.mean([temp2[i][arm_select] for i in range(len(temp2)-self.batch,len(temp2))])
            meandiff = abs(self_mean - node_mean)
            J = max(1,self.batch * (1-meandiff))
            K = max(1,self.batch * meandiff)

            update_val = np.random.beta(J, K)

            if self.trusted_updates[nei_nodes[idx]] is not None:
                prev_val = self.trusted_updates[nei_nodes[idx]][1]
                if passed_crit:
                    self.trusted_updates[nei_nodes[idx]] = [arm_select, max(update_val,prev_val)]
                else:
                    self.trusted_updates[nei_nodes[idx]] = [arm_select, min(update_val,prev_val)]

            self.trusted_updates[nei_nodes[idx]] = [arm_select, update_val]

            if len(self.trusted_neis) == self.num_neis:

                min_nei = None
                min_val = 1.0
                for key,value in self.trusted_updates.items():
                    if value is None:
                        continue
                    if value[1] < min_val:
                        min_nei = key
                        min_val = value[1]

                self.trusted_updates[min_nei] = None
                # self.bandit.nei_select[idx] = 1
                # Remove node from trusted neighbors
                self.trusted_neis.remove(min_nei)

        return Q_target


    def add_exploration_rewards(self, reward_info):
        """Actions intrinsic rewards to encourage exploration"""
        return reward_info["reward"]

    def reset(self):
        """Resets the noise process"""
        pass
