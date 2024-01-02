"""Control Case Optimization

Control case optimization simplified simulator

"""

import yaml
import torch
import optuna
import warnings
import logging

from sim import ReputationSim
from colorlog import ColoredFormatter

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


def study_func(trial):

    file_config = "../config/simconfig.yml"
    with open(file_config, "r") as read_file:
        config = yaml.load(read_file, Loader=yaml.FullLoader)

    # Set trail parameters
    # Num nodes ping
    paramname = "B"
    param = trial.suggest_int(paramname, 5, 10)
    config['nodeparams'][paramname] = param

    # Num trials wait
    paramname = "K"
    param = trial.suggest_int(paramname, 2, 50)
    config['nodeparams'][paramname] = param

    paramname = "trainperc"
    param = trial.suggest_float(paramname, 0.5, 0.8, step=0.1)
    config['nodeparams']['innerParams'][paramname] = param

    paramname = "layernode2"
    layer2 = trial.suggest_int(paramname, 100, 150)
    config['nodeparams']['innerParams']['layer_nodes'][0] = layer2

    paramname = "layernode1"
    layer1 = trial.suggest_int(paramname, 20, layer2)
    config['nodeparams']['innerParams']['layer_nodes'][0] = layer1

    paramname = "sgdlearn"
    param = trial.suggest_float(paramname,0.0001, 1.0, step=0.0001)
    config['nodeparams']['innerParams']['sgdlearn'] = param

    block_sim = ReputationSim(config, trial)
    block_sim.run_sim()

    return block_sim.report_results(step=0, final=True)


def run_sim_optuna(config=[]):
    """
    Manages configuration options to run the reputation simulation.
    :param run_learn: Learners to run for simulation
    :param run_attack: Attack configs to run for simulation
    :param config: Config file
    :return:
    """

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=100, interval_steps=10)
    )

    study.optimize(study_func, n_trials=config['optunaparams']['numtrials'])

    logger.info("Study complete!")
    logger.info("Best params: {}".format(study.best_trial.params))
    logger.info("Best Accuracy: {}".format(study.best_trial.value))

    return


if __name__ == "__main__":

    configure_logging("INFO", False)
    attacks = ["DATA"]
    torch.set_default_dtype(torch.float32)
    torch.autograd.set_detect_anomaly = False
    torch.autograd.profiler.profile = False

    fileConfig = "../config/optunaconfig.yml"
    with open(fileConfig, "r") as read_file:
        config = yaml.load(read_file, Loader=yaml.FullLoader)

    run_sim_optuna(config)





