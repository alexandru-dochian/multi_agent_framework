"""
Config
"""
import time

import torch

from communicator import get_communicator
from core import Config, Environment, Position, Communicator


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
        super().__init__(FieldModulationEnvironmentConfig(**config), None, get_communicator(communicator))

    def run(self):
        print("Starting environment")

        while self.communicator.is_active():
            print("Loop environment")

            for agent in self.communicator.registered_agents():
                print(f"Environment: agent = [{agent}]")

            time.sleep(1 / self.config.clock_freq)

        print("Finished environment")


def spawn_environment(
        class_name: str,
        params: dict,
):
    # """
    # This method is the entrypoint for the environment process
    # """
    print(f"Spawn {class_name} environment!")
    environment_class: type[Environment] = globals()[class_name]
    environment: Environment = environment_class(**params)

    environment.run()
