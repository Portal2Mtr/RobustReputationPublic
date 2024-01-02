"""Secure Reputation System Simulator

Main simulation file for the secure IoT Blockchain reputation system. Contains several agents using homogeneous solvers
to aggregate SVD vectors for calculations. Adversaries try to manipulate the distributed data to divert matrices used
for learning in distributed systems.

"""
import yaml
import torch
import logging
import argparse
import warnings

from colorlog import ColoredFormatter
from sim import ReputationSim

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


def configure_logging(verbosity, enable_colors):
    """
    Configures the code logger for reporting various information.
    :param verbosity: Sets logger verbosity level
    :type verbosity: str
    :param enable_colors: Enables logger colors
    :type enable_colors: bool
    :return:
    :rtype:
    """
    root_logger = logging.getLogger()
    console = logging.StreamHandler()

    if enable_colors:
        # create a colorized formatter
        formatter = ColoredFormatter(
            "%(log_color)s[%(filename)s] %(asctime)s %(levelname)-8s%(reset)s %(white)s%(message)s",
            datefmt=None,
            reset=True,
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "cyan,bg_red",
            },
            secondary_log_colors={},
            style="%"
        )
    else:
        # create a plain old formatter
        formatter = logging.Formatter(
            "[%(filename)s] %(asctime)s %(levelname)-8s %(message)s"
        )

    # Add the formatter to the console handler, and the console handler to the root logger
    console.setFormatter(formatter)
    for hdlr in root_logger.handlers:
        root_logger.removeHandler(hdlr)
    root_logger.addHandler(console)

    # Set logging level for root logger
    root_logger.setLevel(verbosity)


def run_sim(run_learn=True, run_attack=False, run_consensus=False, config=[]):
    """
    Manages configuration options to run the reputation simulation.
    :param run_learn: Learners to run for simulation
    :param run_attack: Attack configs to run for simulation
    :param config: Config file
    :return:
    """


    learners = ['DQN']

    consensuses = ['TS']
    # consensuses = ['DBTS']
    cons_attacks = ['CONTROL', "GREEDY_UNIFORM","GREEDY_MM"]

    if run_learn: # Run all learners for a single attack from config file
        num_tests = 3  # Repeated tests for accuracy
        seeds = [i for i in range(num_tests)]

        for learn in learners:

            # Set agent solver
            config['nodeparams']['agentSolver'] = learn

            # Handle for control case
            if learn == "BASELINE":
                config['nodeparams']['advAttack'] = 'UNIFORM'
            logger.info("Starting tests for {}".format(learn))

            # Execute agents
            for seed in seeds:
                config['simparams']['seed'] = seed
                block_sim = ReputationSim(config)
                block_sim.run_sim()
                block_sim.save_results()
                logger.info("###########Finished trial #{}!".format(seed))

    # elif run_attack: # Run all attacks for all learner from config file
    #
    #     num_tests = 5  # Repeated tests for accuracy
    #     seeds = [i for i in range(num_tests)]
    #
    #     for attack in attacks:
    #
    #         condsybil, atttype = attack.split('_')
    #
    #         config['nodeparams']['advAttack'] = atttype
    #         config['simparams']['attacktype'] = condsybil
    #         logger.info("Running test for the {} attack...".format(attack))
    #         for learn in learners:
    #
    #             config['nodeparams']['agentSolver'] = learn
    #
    #             logger.info("Starting tests for {} with {}...".format(learn, attack))
    #
    #             for seed in seeds:
    #                 config['simparams']['seed'] = seed
    #                 block_sim = ReputationSim(config)
    #                 block_sim.run_sim()
    #                 block_sim.save_results()
    #                 logger.info("###########Finished trial #{} for {} with {}!".format(seed,learn,attack))


    elif run_consensus:

        num_tests = 5  # Repeated tests for accuracy
        seeds = [i for i in range(num_tests)]

        # Assume one learner
        learn = learners[0]

        # Set agent solver
        config['nodeparams']['agentSolver'] = learn

        for consensus in consensuses:

            for attack in cons_attacks:

                config['nodeparams']['consensus_scheme'] = consensus
                config['nodeparams']['consensus_attack'] = attack

                # Handle for control case
                logger.info("Starting tests for {} with {} using {} attack...".format(learn, consensus,attack))

                # Execute agents
                for seed in seeds:
                    config['simparams']['seed'] = seed
                    block_sim = ReputationSim(config)
                    block_sim.run_sim()
                    block_sim.save_results()
                    logger.info("###########Finished trial #{}!".format(seed))

    else:  # Single attack and single learner from config file
        num_tests = 3  # Repeated tests for accuracy
        seeds = [i for i in range(num_tests)]
        for seed in seeds:
            config['simparams']['seed'] = seed
            block_sim = ReputationSim(config)
            block_sim.run_sim()
            block_sim.save_results()
            logger.info("###########Finished trial #{}!".format(seed))

    return


def updateparams(args, config):
    """
    Updates config parameters from program args
    :param args:
    :param config:
    :return:
    """


    config['simparams']['attacktype'] = args['attacktype']
    config['simparams']['seed'] = args['custseed']
    config['nodeparams']['advAttack'] = args['advattack']
    config['nodeparams']['agentSolver'] = args['agentsolver']
    config['nodeparams']['consensus_scheme'] = args['consensus_scheme']

    logger.info("Starting study for seed {} for {} nodes with {} attack using {}".format(args['custseed'],
                                                                                         args['attacktype'],
                                                                                         args['advattack'],
                                                                                         args['agentsolver']))

    return config


if __name__ == "__main__":

    configure_logging("INFO", False)

    parser = argparse.ArgumentParser(description='Config file input')
    parser.add_argument('--runalllearners', type=int, default=0, help='Run all learners')
    parser.add_argument('--runallattacks', type=int, default=0, help='Run all attack Types')
    parser.add_argument('--runallconsensus', type=int, default=0, help='Run all attack Types')
    parser.add_argument('--customparam', type=bool, default=False, help='update with custom params')
    parser.add_argument('--runblockchain', type=bool, default=False, help='Save under different name for plotting')
    parser.add_argument('--custseed', type=int,default=0)
    parser.add_argument('--attacktype', type=str,default='DATA',help='DATA or SYBIL')
    parser.add_argument('--advattack', type=str,default='CONTROL',help='CONTROL or GREEDY')
    parser.add_argument('--agentsolver', type=str,default='DQN',help='Agent solver for sim')
    parser.add_argument('--consensus_type', type=str,default='EPSGREEDY',help='Options: EPSGREEDY, MVEDQL, DBTS')

    args = parser.parse_args()
    torch.set_default_dtype(torch.float32)
    torch.autograd.set_detect_anomaly = False
    torch.autograd.profiler.profile = False

    parseargs = {'attacktype': args.attacktype,
                 'custseed': args.custseed,
                 'advattack': args.advattack,
                 'agentsolver': args.agentsolver,
                 'consensus_type': args.consensus_type}

    fileConfig = "../config/simconfig.yml"
    with open(fileConfig, "r") as read_file:
        config = yaml.load(read_file, Loader=yaml.FullLoader)

    if args.customparam:
        config = updateparams(parseargs, config)

    if args.runblockchain:
        config['simparams']['blockreplogging'] = True

    run_learners = args.runalllearners
    run_attacks = args.runallattacks
    run_consensus = args.runallconsensus

    run_sim(run_learners, run_attacks, run_consensus,config)





