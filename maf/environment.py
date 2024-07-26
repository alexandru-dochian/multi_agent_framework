import logging
import time
from abc import ABC, abstractmethod

import numpy as np

from maf import field_modulation, logger_config
from maf.core import (
    Config,
    Communicator,
    ProcessInitConfig,
    State,
    Position,
    SpaceLimit,
    FieldModulationEnvironmentState,
)
from maf.communicator import get_communicator

logger: logging.Logger = logger_config.get_logger(__name__)


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


class FieldModulationEnvironmentConfig(Config):
    delay: int = 100  # ms
    space_limit: SpaceLimit
    modulations: list[Position] = []
    rotation_center: Position
    theta: float = 5  # degrees


class FieldModulationEnvironment(Environment):
    config: FieldModulationEnvironmentConfig
    state: FieldModulationEnvironmentState
    communicator: Communicator

    def __init__(self, config: dict, communicator: dict):
        config: FieldModulationEnvironmentConfig = FieldModulationEnvironmentConfig(
            **config
        )
        state: FieldModulationEnvironmentState = FieldModulationEnvironmentState(
            modulations=np.array(
                list(map(lambda position: position.to_numpy_2d(), config.modulations))
            ),
            space_limit=config.space_limit,
        )
        super().__init__(
            config,
            state,
            get_communicator(communicator),
        )

    def run(self):
        while self.communicator.is_active():
            self.update_state()
            self.communicator.broadcast_environment_state(self.state)

            delay_seconds = self.config.delay / 1000
            time.sleep(delay_seconds)

    def update_state(self):
        self.state.modulations = field_modulation.rotate_points(
            self.state.modulations,
            self.config.rotation_center.to_numpy_2d(),
            self.config.theta,
        )


def spawn_environment(
    init_config: ProcessInitConfig,
):
    """
    This method is the entrypoint for the environment process
    """
    logger.info(f"Spawn {init_config.class_name} environment {init_config.worker}!")
    environment_class: type[Environment] = globals()[init_config.class_name]
    assert issubclass(
        environment_class, Environment
    ), f"Environment [{init_config.class_name}] was not found"
    environment: Environment = environment_class(**init_config.params)
    environment.run()
    logger.info(f"Finished {init_config.class_name} environment!")
