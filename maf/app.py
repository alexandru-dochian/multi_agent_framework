import logging
import signal
import threading
import multiprocessing as mp
import time
from typing import Union

from maf.agent import spawn_agent
from maf.communicator import get_communicator
from maf.core import Communicator, Config, ObjectInitConfig, ProcessInitConfig, Worker
from maf.environment import spawn_environment
from maf.log_handler import spawn_log_handler

AppProcess = Union[threading.Thread, mp.Process]


class AppConfig(Config):
    communicators: list[ObjectInitConfig]
    environments: list[ProcessInitConfig]
    agents: list[ProcessInitConfig]
    log_handlers: list[ProcessInitConfig]


class App:
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

    def start_processes(self):
        try:
            for p in self.processes:
                logging.info(f"Starting process {p.name}")
                p.start()
                logging.info(f"Process {p.name} started successfully")
        except Exception as e:
            logging.error(
                f"Error occurred while starting processes: {e}", exc_info=True
            )

            # Optionally, you might want to clean up or terminate processes here

    def start(self):
        print("Starting application")
        for communicator in self.communicators:
            communicator.activate()

        # Starting app processes
        for p in self.processes:
            try:
                p.start()
                logging.info(f"Process {p.name} started successfully")
            except:
                logging.exception(f"Process {p.name} failed to start")

        while all([communicator.is_active() for communicator in self.communicators]):
            time.sleep(0.1)

        # Wait for app processes to finish
        for p in self.processes:
            try:
                p.join()
                logging.info(f"Process {p.name} finished successfully")
            except:
                logging.exception(f"Process {p.name} failed to finish")

        print("Finished application")


def get_worker_type(worker: Worker) -> type[AppProcess]:
    if worker == Worker.Thread:
        return threading.Thread
    elif worker == Worker.Process:
        return mp.Process
    else:
        raise Exception("Valid worker must be specified in this case")
