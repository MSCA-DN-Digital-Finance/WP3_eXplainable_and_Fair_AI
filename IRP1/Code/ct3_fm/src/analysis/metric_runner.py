# This file contains the functions required for the metric to compute the parameter statistics for a given run.

import json

from src.prediction.pred_runner import get_exp_id_from_meta
from src.utils import load_config




def calculate_param_stats(run_path, param_registry):
    """
    Calculates the parameter statistics for a given run and saves the results to a JSON file.

    Args:
        run_path (str): Path to the run directory.
        param_registry (dict): A dictionary mapping parameter statistic names to their corresponding functions.

    Returns:
        dict: A dictionary containing the computed parameter statistics.
    """
    # get experiment_id from artifacts/trajectories/hash/meta.json for a given run path
    exp_id = get_exp_id_from_meta(run_path)

    # get parameter statistic from experiment_id for the current experiment_id
    exp_config = load_config("../../experiment_config.yaml")
    param_stat_name = exp_config['experiments'][exp_id]['analysis']['param_stat']

    # select parameter statistic function from param_registry
    param_stat_func = param_registry.get(param_stat_name)

    # create a results dictionary to store the computed parameter statistics

    # for each model folder

        # load the model's name from meta.json

        # load the model's predictions from artifacts/trajectories/hash/predictions.json

        # run parameter statistic function

        # add param stat to dictionnary

    # save the results dictionary to run_path/param_stats.json


def run_metric(dir, param_registry):
    """
    Iterate over each run directory in the given path and call calculate_param_stats for each run.

    Args:
        dir (str): Path to the directory containing run directories.
        param_registry (dict): A dictionary mapping parameter statistic names to their corresponding functions.

    """
    

    

