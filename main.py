import argparse
import json
import multiprocessing as mp
import os
import signal
import time
import threading
from abc import ABC
from typing import Union

from core import Worker, Config, ObjectInitConfig, ProcessInitConfig
from communicator import Communicator, get_communicator
from agent import spawn_agent
from log_handler import spawn_log_handler
from environment import spawn_environment

CONFIG_DIR = "config"

AppProcess = Union[threading.Thread, mp.Process]


class AppConfig(Config):
    communicators: list[ObjectInitConfig]
    environments: list[ProcessInitConfig]
    agents: list[ProcessInitConfig]
    log_handlers: list[ProcessInitConfig]


class App(ABC):
    config: AppConfig

    communicators: list[Communicator] = []
    processes: list[AppProcess] = []

    def __init__(self, config: dict):
        self.config: AppConfig = AppConfig(**config)
        self.init_communicators()
        self.init_processes()

    def init_communicators(self):
        for communicator_config in self.config.communicators:
            communicator: Communicator = get_communicator(communicator_config)
            signal.signal(signal.SIGTERM, communicator.stop)
            signal.signal(signal.SIGINT, communicator.stop)
            self.communicators.append(communicator)

    def init_processes(self):
        for environment_config in self.config.environments:
            worker_type: type[AppProcess] = get_worker_type(environment_config.worker)
            self.processes.append(
                worker_type(
                    target=spawn_environment,
                    args=(environment_config,),
                )
            )

        for agent_config in self.config.agents:
            worker_type: type[AppProcess] = get_worker_type(agent_config.worker)
            self.processes.append(worker_type(target=spawn_agent, args=(agent_config,)))

        for logger_config in self.config.log_handlers:
            worker_type: type[AppProcess] = get_worker_type(logger_config.worker)
            self.processes.append(
                worker_type(
                    target=spawn_log_handler,
                    args=(logger_config,),
                )
            )

    def start(self):
        print("Starting application")

        for communicator in self.communicators:
            communicator.activate()

        # Starting app processes
        for p in self.processes:
            p.start()

        while all([communicator.is_active() for communicator in self.communicators]):
            time.sleep(0.1)

        # Wait for app processes to finish
        for p in self.processes:
            p.join()

        print("Finished application")


def parse_arguments():
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Multi agent framework for collective intelligence research",
    )
    parser.add_argument(
        "experiment_config_file", nargs="?", default=f"{CONFIG_DIR}/default.json"
    )
    arguments = parser.parse_args()
    print("arguments", arguments)
    print(f"Using config_file = [{arguments.experiment_config_file}]")

    if not arguments.experiment_config_file.endswith(".json"):
        raise Exception("experiment_config_file should be a json file")

    return arguments.experiment_config_file


def load_config(file_path: str) -> dict:
    abs_file_path = os.path.abspath(file_path)
    expected_directory_path = os.path.abspath(CONFIG_DIR)
    assert (
            os.path.commonprefix([abs_file_path, expected_directory_path])
            == expected_directory_path
    ), f"Config file should be placed under `{CONFIG_DIR}` directory!"
    with open(os.path.join(CONFIG_DIR, abs_file_path)) as fp:
        return json.load(fp)


def get_worker_type(worker: Worker) -> type[AppProcess]:
    if worker == Worker.Thread:
        return threading.Thread
    elif worker == Worker.Process:
        return mp.Process
    else:
        raise Exception("Valid worker must be specified in this case!")


if __name__ == "__main__":
    App(load_config(parse_arguments())).start()
