import logging
import time
from abc import ABC, abstractmethod

import torch

from maf.core import Config, Position, Communicator, ProcessInitConfig, State
from maf.communicator import get_communicator


class Environment(ABC):
    config: Config
    state: State
    communicator: Communicator

    def __init__(self, config: Config, state: State, communicator: Communicator):
        self.config = config
        self.state = state
        self.communicator = communicator

    @abstractmethod
    def run(self):
        ...


def apply_distribution(
        tensor: torch.Tensor, position: Position, sigma=5, amplitude=1, add=True
):
    """
    Apply a Gaussian distribution around a given position on a 2D tensor.

    Args:
    - tensor (torch.Tensor): Input 2D tensor representing the map.
    - position (Position): Position (x, y) where the distribution will be centered.
    - sigma (float): Standard deviation of the Gaussian distribution.
    - amplitude (float): Amplitude of the Gaussian distribution.

    Returns:
    - torch.Tensor: Modified tensor with the Gaussian distribution applied.
    """
    x = position.x
    y = position.y
    grid_x, grid_y = torch.meshgrid(
        torch.arange(tensor.size(0)), torch.arange(tensor.size(1))
    )
    gaussian = amplitude * torch.exp(
        -((grid_x - x) ** 2 + (grid_y - y) ** 2) / (2 * sigma ** 2)
    )
    if add:
        return tensor + gaussian
    else:
        return tensor - gaussian


class FieldModulationEnvironmentConfig(Config):
    clock_freq: int = 10


class FieldModulationEnvironment(Environment):
    config: FieldModulationEnvironmentConfig
    communicator: Communicator

    def __init__(self, config: dict, communicator: dict):
        super().__init__(
            FieldModulationEnvironmentConfig(**config),
            None,
            get_communicator(communicator),
        )

    def run(self):
        logging.info("Starting environment")

        while self.communicator.is_active():
            logging.info("Loop environment")

            for agent in self.communicator.registered_agents():
                logging.info(f"Environment: agent = [{agent}]")

            time.sleep(1 / self.config.clock_freq)

        logging.info("Finished environment")


def spawn_environment(
        init_config: ProcessInitConfig,
):
    """
    This method is the entrypoint for the environment process
    """
    logging.info(f"Spawn {init_config.class_name} environment {init_config.worker}!")
    environment_class: type[Environment] = globals()[init_config.class_name]
    assert issubclass(
        environment_class, Environment
    ), f"Environment [{init_config.class_name}] was not found"
    environment: Environment = environment_class(**init_config.params)
    environment.run()
