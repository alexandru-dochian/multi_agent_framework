import argparse
import json
import multiprocessing as mp
import os
import time
import threading
import signal

from core import Worker, ExperimentConfig
from communicator import Communicator, get_communicator
from agent import spawn_agent
from logger import spawn_logger
from environment import spawn_environment

CONFIG_DIRECTORY = "config"


def parse_arguments():
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Multi agent framework for collective intelligence research",
    )
    parser.add_argument("experiment_config_file", nargs="?", default=f"{CONFIG_DIRECTORY}/default.json")
    arguments = parser.parse_args()
    print("arguments", arguments)
    print(f"Using config_file = [{arguments.experiment_config_file}]")

    if not arguments.experiment_config_file.endswith(".json"):
        raise Exception("experiment_config_file should be a json file")

    return arguments.experiment_config_file


def load_config(file_path: str) -> dict:
    abs_file_path = os.path.abspath(file_path)
    expected_directory_path = os.path.abspath(CONFIG_DIRECTORY)
    assert os.path.commonprefix([abs_file_path, expected_directory_path]) == expected_directory_path, \
        f"Config file should be placed under `{CONFIG_DIRECTORY}` directory!"
    with open(os.path.join(CONFIG_DIRECTORY, abs_file_path)) as fp:
        return json.load(fp)


def get_worker_type(worker: Worker) -> type:
    if worker == Worker.Thread:
        return threading.Thread
    elif worker == Worker.Process:
        return mp.Process
    else:
        raise Exception("Valid worker must be specified in this case!")


if __name__ == "__main__":
    # Load experiment config
    experiment_config: ExperimentConfig = ExperimentConfig(
        **load_config(parse_arguments())
    )

    # Setup communicator and bind stop callback to OS signal
    communicator: Communicator = get_communicator(
        dict(experiment_config.communicator)
    )
    signal.signal(signal.SIGTERM, communicator.stop_communication)
    signal.signal(signal.SIGINT, communicator.stop_communication)
    communicator.start_communication()

    # Setup processes
    processes = []
    for environment_config in experiment_config.environments:
        worker_type: type = get_worker_type(environment_config.worker)
        processes.append(
            worker_type(
                target=spawn_environment,
                args=(environment_config.class_name, environment_config.params),
            )
        )

    for agent_config in experiment_config.agents:
        worker_type: type = get_worker_type(agent_config.worker)
        processes.append(
            worker_type(
                target=spawn_agent, args=(agent_config.class_name, agent_config.params)
            )
        )

    for logger_config in experiment_config.loggers:
        worker_type: type = get_worker_type(logger_config.worker)
        processes.append(
            worker_type(
                target=spawn_logger,
                args=(logger_config.class_name, logger_config.params),
            )
        )

    # Starting processes
    for p in processes:
        p.start()

    while communicator.is_active():
        time.sleep(0.1)

    # Wait for processes to finish
    for p in processes:
        p.join()

    print("Finished root process")
