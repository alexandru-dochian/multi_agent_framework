import multiprocessing as mp
import os
import time
import threading
import signal
from abc import ABC

import torch

from analytics import input_space_viz, central_viz
from core import get_communicator, Communicator

from agent import spawn_agent, Position, FieldState, Field


def signal_handler(signal_received, frame):
    communicator: Communicator = get_communicator("RedisCommunicator")
    print(f"\nReceived CTRL-C for [{os.getpid()}] of parent [{os.getppid()}]...")
    communicator.send(Communicator.CommKey.STOP_EVENT, True)


signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


def apply_distribution(tensor: torch.Tensor, position: Position, sigma=5, amplitude=1, add=True):
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
    grid_x, grid_y = torch.meshgrid(torch.arange(tensor.size(0)), torch.arange(tensor.size(1)))
    gaussian = amplitude * torch.exp(-((grid_x - x) ** 2 + (grid_y - y) ** 2) / (2 * sigma ** 2))
    if add:
        return tensor + gaussian
    else:
        return tensor - gaussian


class EnvironmentConfig(ABC):
    def __init__(self):
        ...


class Environment(ABC):
    def __init__(self, config: EnvironmentConfig):
        ...

    def run(self):
        ...


class FieldModulationEnvironment(ABC):
    def __init__(self, config: EnvironmentConfig):
        ...

    def run(self):
        ...


def environment_loop(agent_ids: list[str]):
    ...
    # env_comm: Communicator = get_communicator("RedisCommunicator")
    # env_comm.send(Communicator.CommKey.STOP_EVENT, False)
    #
    # positions: list[Position] = []
    # big_field: Field = Field(data=torch.zeros(1024, 1024))
    # for agent_id in agent_ids:
    #     state_key = Communicator.get_state_key(agent_id)
    #     state: FieldState = env_comm.recv(state_key)
    #     positions.append(state.position)
    #     apply_distribution(big_field, state.position, add=False)


if __name__ == "__main__":
    threads = [
        threading.Thread(target=spawn_agent, args=(
            "CFAgent2D",
            "CFAgent2DConfig",
            {
                "agent_id": "radio://0/100/2M/E7E7E7E707",
                "clock_freq": 10,
                "default_height": 0.4,
                "max_vel": 0.2,
                "communicator": "RedisCommunicator",
                "log_variables": [
                    "stateEstimate.x",
                    "stateEstimate.y",
                    "stateEstimate.z",
                ],
                "log_interval_ms": 500,
                "total_time": 60,
            },
            "GoToPointController",
            "CFControllerConfig",
            {
                "target_position": {
                    "x": 1.5,
                    "y": 2,
                },
            },
        )),
        threading.Thread(target=spawn_agent, args=(
            "CFAgent2D",
            "CFAgent2DConfig",
            {
                "agent_id": "radio://0/100/2M/E7E7E7E704",
                "clock_freq": 10,
                "default_height": 0.4,
                "max_vel": 0.2,
                "communicator": "RedisCommunicator",
                "log_variables": [
                    "stateEstimate.x",
                    "stateEstimate.y",
                    "stateEstimate.z",
                ],
                "log_interval_ms": 500,
                "total_time": 60,
            },
            "GoToPointController",
            "CFControllerConfig",
            {
                "target_position": {
                    "x": -2.5,
                    "y": -1,
                },
            },
        )),
        threading.Thread(target=spawn_agent, args=(
            "CFAgent2D",
            "CFAgent2DConfig",
            {
                "agent_id": "radio://0/100/2M/E7E7E7E70A",
                "clock_freq": 10,
                "default_height": 0.4,
                "max_vel": 0.2,
                "communicator": "RedisCommunicator",
                "log_variables": [
                    "stateEstimate.x",
                    "stateEstimate.y",
                    "stateEstimate.z",
                ],
                "log_interval_ms": 500,
                "total_time": 60,
            },
            "GoToPointController",
            "CFControllerConfig",
            {
                "target_position": {
                    "x": -2.5,
                    "y": 2,
                },
            },
        )),
        threading.Thread(target=spawn_agent, args=(
            "CFAgent2D",
            "CFAgent2DConfig",
            {
                "agent_id": "radio://0/100/2M/E7E7E7E708",
                "clock_freq": 10,
                "default_height": 0.4,
                "max_vel": 0.2,
                "communicator": "RedisCommunicator",
                "log_variables": [
                    "stateEstimate.x",
                    "stateEstimate.y",
                    "stateEstimate.z",
                ],
                "log_interval_ms": 500,
                "total_time": 60,
            },
            "GoToPointController",
            "CFControllerConfig",
            {
                "target_position": {
                    "x": 1.5,
                    "y": -1,
                },
            },
        ))
    ]

    agent_ids = [
        "radio://0/100/2M/E7E7E7E707",
        "radio://0/100/2M/E7E7E7E704",
        "radio://0/100/2M/E7E7E7E70A",
        "radio://0/100/2M/E7E7E7E708",
    ]
    processes = [
        mp.Process(target=input_space_viz, args=("radio://0/100/2M/E7E7E7E707", "RedisCommunicator")),
        mp.Process(target=input_space_viz, args=("radio://0/100/2M/E7E7E7E704", "RedisCommunicator")),
        mp.Process(target=input_space_viz, args=("radio://0/100/2M/E7E7E7E70A", "RedisCommunicator")),
        mp.Process(target=input_space_viz, args=("radio://0/100/2M/E7E7E7E708", "RedisCommunicator")),
        mp.Process(target=central_viz, args=(agent_ids, "RedisCommunicator")),
    ]

    # Starting processes
    for p in processes:
        p.start()

    # Starting threads
    for t in threads:
        t.start()

    communicator: Communicator = get_communicator("RedisCommunicator")
    communicator.send(Communicator.CommKey.STOP_EVENT, False)

    while not (communicator.recv(Communicator.CommKey.STOP_EVENT) is True):
        environment_loop(agent_ids)
        time.sleep(0.5)

    # Wait for threads to finish
    for t in threads:
        t.join()

    # Wait for processes to finish
    for p in processes:
        p.join()

    print("Finished root process")
