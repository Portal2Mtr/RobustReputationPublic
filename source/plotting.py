import matplotlib.pyplot as plt
import pickle
import numpy as np
from os.path import exists
import pandas as pd
import matplotlib

def repplotting():
    font = {'size'   : 10}
    #
    matplotlib.rc('font', **font)
    pe = 1 # Plot every
    max_step=300 # Maximum plotting

    max_seed = 2
    seeds = [i for i in range(max_seed)]
    cons_schemes = ["EPSGREEDY","ROBUSTQD","DBTS"]
    mean_err_lists = {}
    mean_rep_lists = {}
    mean_acc_lists = {}

    attackTypes = ["DATA"]

    schemes = ["CONTROL"]

    for scheme in schemes:

        for attack in attackTypes:

            mean_err_list = []
            mean_rep_list = []
            mean_acc_list = []

            for learner in cons_schemes:

                err_list = []
                rep_list = []
                acc_list = []
                for seed in seeds:
                    filename = "../output/blockrepresultsCONSENSUS_{}_{}_{}_{}.p".format(learner, attack, scheme, seed)
                    if not exists(filename):
                        continue
                    error, adv_rep, acc, hon_rep = pickle.load(open(filename, 'rb'))
                    err_list.append(error)
                    rep_list.append(hon_rep)
                    acc_list.append(acc)

                err_mean = np.mean(err_list, axis=0)
                rep_mean = np.mean(rep_list, axis=0)
                acc_mean = np.mean(acc_list, axis=0)

                mean_err_lists['{}_{}_{}'.format(learner, attack, scheme)] = err_mean
                mean_rep_lists['{}_{}_{}'.format(learner, attack, scheme)] = rep_mean
                mean_acc_lists['{}_{}_{}'.format(learner, attack, scheme)] = acc_mean

    ticklist = ['x', '2', 'D']
    colorlist = ['tab:blue', 'tab:orange', 'tab:purple']

    for key,val in mean_err_lists.items():
        if not np.any(np.isnan(val)):
            mean_err_lists[key] = val[:max_step]

    for key,val in mean_rep_lists.items():
        if not np.any(np.isnan(val)):
            mean_rep_lists[key] = val[:max_step]

    for key,val in mean_acc_lists.items():
        if not np.any(np.isnan(val)):
            mean_acc_lists[key] = val[:max_step]

    # ############# Control case error
    sellist = ['EPSGREEDY_DATA_CONTROL', 'ROBUSTQD_DATA_CONTROL', 'DBTS_DATA_CONTROL']
    plotlist = ['DQN (Control)', 'Robust-QD', 'DBTS']

    worklist = []
    for select in sellist:
        worklist.append(mean_err_lists[select])

    fig, axs = plt.subplots()

    xvec = np.array([i for i in range(len(worklist[0]))])
    for label, data, color, tick in zip(plotlist,worklist,colorlist,ticklist):
        axs.plot(xvec[::pe], data[::pe],color, label=label,marker=tick,markevery=5,linestyle='dashed')

    axs.set_ylabel("Average Inner Loss")
    axs.set_xlim([0, len(xvec)])
    axs.set_xlabel("Time [Epoch]")
    axs.set_ylim([0.0, 1.0])

    handles, labels = axs.get_legend_handles_labels()
    plt.legend()


    plt.savefig("../output/figures/errorcompconsensuscontrol.png", dpi=400, bbox_inches='tight')


def blockchainplotting():

    # font = {'size'   : 15}

    # matplotlib.rc('font', **font)

    max_seed = 4  # Ten trials
    pe = 20 # Plot every
    seeds = [i for i in range(max_seed)]
    ticklist = ['.','x', '2', 'D', '1']
    colorlist = ['tab:blue', 'tab:orange','tab:gray', 'tab:green','tab:purple']

    # GEt data for WBR
    constypes = ['MVEDQL','LQN','TS', 'DBTS']
    netatttypes = ['CONTROL','GREEDY_UNIFORM']

    rep_dict = {}
    for cons in constypes:
        for netatt in netatttypes:
            mean_rep = []
            for seed in seeds:
                filename = "../output/blockrepresultsCONSENSUS_{}_{}_{}.p".format(cons,netatt, seed)
                # if not exists(filename):
                #     continue
                error_vals, _, _, _ = pickle.load(open(filename, 'rb'))
                mean_rep.append(error_vals)
            avg_rep = list(np.mean(mean_rep,axis=0))
            rep_dict['{}_{}'.format(cons, netatt)] = avg_rep


    constypes = ['TS', 'DBTS']
    netatttypes = ['GREEDY_MM']

    for cons in constypes:
        for netatt in netatttypes:
            mean_rep = []
            for seed in seeds:
                filename = "../output/blockrepresultsCONSENSUS_{}_{}_{}.p".format(cons,netatt, seed)
                # if not exists(filename):
                #     continue
                error_vals, _, _, _ = pickle.load(open(filename, 'rb'))
                mean_rep.append(error_vals)
            avg_rep = list(np.mean(mean_rep,axis=0))
            rep_dict['{}_{}'.format(cons, netatt)] = avg_rep


    constypes = ['CONTROL']
    netatttypes = ['CONTROL']

    for cons in constypes:
        for netatt in netatttypes:
            mean_rep = []
            for seed in seeds:
                filename = "../output/blockrepresultsCONSENSUS_{}_{}_{}.p".format(cons,netatt, seed)
                # if not exists(filename):
                #     continue
                error_vals, _, _, _ = pickle.load(open(filename, 'rb'))
                mean_rep.append(error_vals)
            avg_rep = list(np.mean(mean_rep,axis=0))
            rep_dict['{}_{}'.format(cons, netatt)] = avg_rep

    # Plot control reputation
    sellist = ['CONTROL_CONTROL','MVEDQL_CONTROL', 'LQN_CONTROL','DBTS_CONTROL']
    plotlist = ['Control','MVEDQL','LQN','DBTS (This work)']

    worklist = []
    for select in sellist:
        worklist.append(rep_dict[select])

    fig, axs = plt.subplots()

    fig.set_figheight(2.4)
    xvec = np.array([i for i in range(len(worklist[0]))])
    for label, data,color,tick in zip(plotlist, worklist,colorlist,ticklist):
        axs.plot(xvec, data, label=label,c=color,marker=tick,markevery=10)

    axs.set_ylabel("Inner Learning Error")
    axs.set_xlim([0, len(xvec)])
    axs.set_xlabel("Time")
    axs.set_ylim([0.0, 1.2])
    axs.legend(loc='upper left',ncol=2,columnspacing=0.5)

    plt.savefig("../output/figures/errorcompconsensuscontrol.png", dpi=400, bbox_inches='tight')

    sellist = ['MVEDQL_GREEDY_UNIFORM', 'LQN_GREEDY_UNIFORM','DBTS_GREEDY_UNIFORM']
    plotlist =  ['MVEDQL','LQN','DBTS (This work)']

    worklist = []
    for select in sellist:
        worklist.append(rep_dict[select])

    fig, axs = plt.subplots()

    fig.set_figheight(2.4)
    xvec = np.array([i for i in range(len(worklist[0]))])
    for label, data,color,tick in zip(plotlist, worklist,colorlist[1:],ticklist[1:]):
        axs.plot(xvec, data, label=label,c=color,marker=tick,markevery=10)

    axs.set_ylabel("Inner Learning Error")
    axs.set_xlim([0, len(xvec)])
    axs.set_xlabel("Time")
    axs.set_ylim([0.0, 1.2])
    axs.legend(loc='upper left',ncol=2,columnspacing=0.5)

    plt.savefig("../output/figures/errorcompconsensusgreedyuniform.png", dpi=400, bbox_inches='tight')

    # sellist = ['TS_GREEDY_MM', 'DBTS_GREEDY_MM']
    # plotlist =  ['TS (FC)', 'DBTS (FC)']
    #
    # worklist = []
    # for select in sellist:
    #     worklist.append(rep_dict[select])
    #
    # fig, axs = plt.subplots()
    #
    # fig.set_figheight(2.4)
    # xvec = np.array([i for i in range(len(worklist[0]))])
    # for label, data,color,tick in zip(plotlist, worklist,colorlist[-2:],ticklist[-2:]):
    #     axs.plot(xvec, data, label=label,c=color,marker=tick,markevery=10)
    #
    # axs.set_ylabel("Inner Learning Error")
    # axs.set_xlim([0, len(xvec)])
    # axs.set_xlabel("Time")
    # axs.set_ylim([0.0, 1.2])
    #
    # axs.legend(loc='upper left',ncol=2,columnspacing=0.5)
    #
    # plt.savefig("../output/figures/errorcompconsensusgreedymm.png", dpi=400, bbox_inches='tight')




if __name__ == "__main__":

    # repplotting()
    blockchainplotting()
