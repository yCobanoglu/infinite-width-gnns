import argparse
import itertools
import multiprocessing
import os
import shutil
from pathlib import Path

import torch
import torch.multiprocessing as mp
import yaml
from yaml import Loader
from gnn import device
from gnn.config.data_config import DataConfig
from gnn.metric import aggregate_list_of_metric_lists

mp.set_start_method("spawn", force=True)


MULTIPROCESSING = os.environ.get("MY_DEBUG") == "0"


def run(config, num_runs, run):
    configs = []
    for i in range(num_runs):
        config_ = config.clone_reset_name()
        config_.run = i + 1
        configs.append(config_)

    if not MULTIPROCESSING:
        results = list(itertools.starmap(run, [(config, True) for config in configs]))
    else:
        with mp.Pool(num_runs) as executor:
            results = executor.starmap(run, [(config, False) for config in configs])
    metrics = aggregate_list_of_metric_lists([result.metrics for result in results])
    config.set_metrics(metrics)
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", help="Directory with config file for experiments")
    parser.add_argument("-t", "--test_run", action="store_true", help="Run in test mode")
    args = parser.parse_args()
    if args.test_run:
        print("Test run!")
        MULTIPROCESSING = False

    experiments_dir = Path(args.directory)
    config_path = experiments_dir / "config.yaml"
    experiments_log_dir = experiments_dir / "logs"

    with open(config_path, "r") as f:
        meta_config = yaml.load(f, Loader=Loader)

    if os.path.exists(experiments_log_dir):
        shutil.rmtree(experiments_log_dir)
    os.makedirs(experiments_log_dir)

    models = meta_config["models"]
    if _device := meta_config.get("device"):
        if _device == "cuda" and torch.cuda.is_available():
            device.DEVICE = "cuda"
        else:
            device.DEVICE = "cpu"

    # after modifying device variable import everything
    from gnn.config.config import Config

    from gnn.runners.custom.unsupervised_classification import (
        run as inductive_clustering,
    )
    from gnn.runners.default.run_default import run as default
    from gnn.runners.link_prediction.link_prediction import run as link_prediction
    from gnn.summarize_logs import summarize

    def select_run(type):
        match type:
            case "link prediction":
                return link_prediction
            case "inductive_clustering":
                return inductive_clustering
            case _:
                return default

    datasets = meta_config["datasets"]
    learning_task = meta_config["learning_task"]
    accuracy = meta_config.get("accuracy")
    loss = meta_config.get("loss")
    runs = meta_config["runs"] if not args.test_run else 1

    parallel = meta_config["parallel"]
    if parallel == -1:
        parallel = multiprocessing.cpu_count()

    processes_num = max(parallel // runs, 1)  # processes dividied by runs because every process spawns {runs} processes

    configs = []
    for model, dataset in list(itertools.product(models, datasets)):
        if args.test_run:
            model["model"]["epochs"] = 1

        config_ = Config(
            model | dataset | {"learning_task": learning_task, "loss": loss, "accuracy": accuracy},
            experiments_log_dir,
        )
        configs.append(config_)

    tasks_to_process = [(config, runs, select_run(config.learning_task)) for config in configs]

    datasets_configs = list(map(lambda data: DataConfig(**data["data"]), datasets))

    if processes_num == 1:
        MULTIPROCESSING = False
    if not MULTIPROCESSING:
        print("No multiprocessing!")
        results = list(itertools.starmap(run, tasks_to_process))
    else:
        processes = min(processes_num * runs, len(datasets_configs) * runs)
        print(f"Multiprocessing! ({processes} processes)")
        with mp.Pool(processes) as executor:
            results = executor.starmap(run, tasks_to_process)

    datasets_configs = list(map(lambda data: DataConfig(**data["data"]), datasets))
    summarize(datasets_configs, results, experiments_log_dir, num_runs=runs)
