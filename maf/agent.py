import logging
import time
from abc import ABC, abstractmethod

from threading import Event

import re

import cflib.crtp
import numpy as np
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander

from maf.communicator import Communicator, get_communicator
from maf.controller import get_controller, Controller
from maf.core import (
    Config,
    FieldState,
    Position,
    Action,
    VelocityCommand2D,
    SimpleAction2D,
    ProcessInitConfig,
    State,
    Command,
    PositionState,
    Field,
)


class Agent(ABC):
    config: Config
    state: State
    controller: Controller
    communicator: Communicator

    def __init__(
            self,
            config: Config,
            state: State,
            controller: Controller,
            communicator: Communicator,
    ):
        self.config = config
        self.state = state
        self.controller: Controller = controller
        self.communicator: Communicator = communicator

    @abstractmethod
    def run(self):
        ...

    @staticmethod
    @abstractmethod
    def action_to_command(action: Action) -> Command:
        ...


class VirtualDrone2DConfig(Config):
    agent_id: str
    clock_freq: int = 5  # hz
    max_vel: float = 0.1  # m/s
    total_time: int = 60  # seconds
    start_position: Position = Position(x=0, y=0, z=0)


class VirtualDrone2D(Agent):
    config: VirtualDrone2DConfig
    state: FieldState
    controller: Controller
    communicator: Communicator

    # TODO config
    field_size: list[int] = [84, 84]

    def __init__(self, config: dict, controller: dict, communicator: dict):
        super().__init__(
            VirtualDrone2DConfig(**config),
            FieldState(),
            get_controller(controller),
            get_communicator(communicator),
        )

    def run(self):
        logging.info(f"Starting [{self.config.agent_id}]!")

        # register on communicator
        self.communicator.register_agent(self.config.agent_id)

        # register initial state
        self.communicator.broadcast_state(
            self.config.agent_id,
            FieldState(
                position=self.config.start_position,
                field=Field(data=np.zeros(self.field_size)),
            ),
        )

        spent_time = 0
        while (spent_time < self.config.total_time) and self.communicator.is_active():
            state: FieldState = self.communicator.get_state(self.config.agent_id)

            self.controller.set_state(state)
            action: SimpleAction2D = self.controller.predict()
            command: VelocityCommand2D = self.action_to_command(action)
            command = command.scale(self.config.max_vel)

            new_state = self.compute_new_state(state, command)
            self.communicator.broadcast_state(self.config.agent_id, new_state)

            duration: float = 1 / self.config.clock_freq
            time.sleep(duration)
            spent_time += duration

        logging.info(f"Finished [{self.config.agent_id}]!")

    @staticmethod
    def apply_distribution(field, position, sigma=10, amplitude=0.2, operation='add'):
        """
        Apply a Gaussian distribution around a given position on a 2D tensor.

        Args:
        - field (np.ndarray): Input 2D array representing the map.
        - position (tuple): Position (x, y) where the distribution will be centered.
        - sigma (float): Standard deviation of the Gaussian distribution.
        - amplitude (float): Amplitude of the Gaussian distribution.
        - operation (str): Operation type ('add', 'subtract', 'replace').

        Returns:
        - np.ndarray: Modified tensor with the Gaussian distribution applied.
        """
        x, y = position
        grid_x, grid_y = np.meshgrid(
            np.arange(field.shape[0]), np.arange(field.shape[1]), indexing="ij"
        )
        gaussian = amplitude * np.exp(
            -((grid_x - x) ** 2 + (grid_y - y) ** 2) / (2 * sigma ** 2)
        )

        if operation == 'add':
            return field + gaussian
        elif operation == 'subtract':
            return field - gaussian
        elif operation == 'replace':
            mask = gaussian > 0
            field[mask] = gaussian[mask]
            return field
        else:
            raise ValueError("Invalid operation type. Use 'add', 'subtract', or 'replace'.")

    @staticmethod
    def map_to_tensor_space(
            points: np.array, center: tuple, tensor_size: tuple, limit_range
    ):
        x_center, y_center = center
        tensor_x, tensor_y = tensor_size
        scale_x = tensor_x / (2 * (limit_range[1] - limit_range[0]))
        scale_y = tensor_y / (2 * (limit_range[1] - limit_range[0]))

        tensor_points = np.ones_like(points)
        tensor_points[:, 0] = (points[:, 0] - x_center) * scale_x + tensor_x // 2
        tensor_points[:, 1] = (points[:, 1] - y_center) * scale_y + tensor_y // 2

        return tensor_points

    def compute_new_state(
            self, old_state: FieldState, command: VelocityCommand2D
    ) -> FieldState:
        duration: float = 1 / self.config.clock_freq
        old_position: Position = old_state.position
        new_position: Position = Position(
            x=old_position.x + command.vel_x * duration,
            y=old_position.y + command.vel_y * duration,
            z=old_position.z,
        )

        current: list = [old_state.position.x, old_state.position.x]
        others: list = []
        real_limit: list = [-1, 1]
        for other_agent_id in self.communicator.registered_agents():
            if other_agent_id == self.config.agent_id:
                # we are interested only in other agents
                continue

            other_agent_state: PositionState = self.communicator.get_state(
                other_agent_id
            )
            others.append([other_agent_state.position.x, other_agent_state.position.y])

        neighbouring_points: np.array = self.map_to_tensor_space(
            np.array(others), current, self.field_size, real_limit
        )

        reward_points: np.array = self.map_to_tensor_space(
            np.array([[0, 0]]), current, self.field_size, real_limit
        )

        new_field: np.array = np.zeros(self.field_size)
        if new_field is not None:
            for field_point in neighbouring_points:
                new_field = self.apply_distribution(new_field, field_point, operation='subtract')

            for reward_point in reward_points:
                new_field = self.apply_distribution(new_field, reward_point, sigma=20, amplitude=0.5, operation='add')

        return FieldState(position=new_position, field=Field(data=new_field))

    @staticmethod
    def action_to_command(action: SimpleAction2D) -> VelocityCommand2D:
        if action == SimpleAction2D.FRONT_LEFT:
            return VelocityCommand2D(1, 1)

        if action == SimpleAction2D.FRONT:
            return VelocityCommand2D(1, 0)

        if action == SimpleAction2D.FRONT_RIGHT:
            return VelocityCommand2D(1, -1)

        if action == SimpleAction2D.RIGHT:
            return VelocityCommand2D(0, -1)

        if action == SimpleAction2D.BACK_RIGHT:
            return VelocityCommand2D(-1, -1)

        if action == SimpleAction2D.BACK:
            return VelocityCommand2D(-1, 0)

        if action == SimpleAction2D.BACK_LEFT:
            return VelocityCommand2D(-1, 1)

        if action == SimpleAction2D.LEFT:
            return VelocityCommand2D(0, 1)

        if action == SimpleAction2D.STOP:
            return VelocityCommand2D(0, 0)


class CFDrone2DConfig(Config):
    agent_id: str
    clock_freq: int = 5  # hz
    default_height: float = 0.4
    max_vel: float = 0.1  # m/s
    log_variables: list[str]
    log_interval_ms: int = 20  # ms
    total_time: int = 60  # seconds


class CFDrone2D(Agent):
    config: CFDrone2DConfig
    state: FieldState
    controller: Controller
    communicator: Communicator

    hex_address: str
    loco_positioning_deck_attached_event: Event

    def __init__(self, config: dict, controller: dict, communicator: dict):
        super().__init__(
            CFDrone2DConfig(**config),
            FieldState(),
            get_controller(controller),
            get_communicator(communicator),
        )
        cflib.crtp.init_drivers()
        # e.g. `radio://0/100/2M/E7E7E7E704`
        assert re.match(
            r"radio://\d/\d{1,3}/\dM/[A-Z0-9]{10}", self.config.agent_id
        ), f"Invalid agent id [{self.config.agent_id}] for crazyflie"
        self.hex_address: str = self.config.agent_id[-10:]
        self.deck_attached_event: Event = Event()

    def run(self):
        def param_deck_flow_anon(_, value_str):
            value = int(value_str)
            logging.info(value)
            if value:
                self.deck_attached_event.set()
                logging.info(f"CFDrone2D [{self.hex_address}] | Deck flow is attached!")
            else:
                logging.info(f"CFDrone2D [{self.hex_address}] | Deck flow is NOT attached!")

        logging.info(f"Initializing [{self.hex_address}]...")
        with SyncCrazyflie(
                self.config.agent_id, Crazyflie(rw_cache=f"./cache/{self.hex_address}")
        ) as scf:
            scf.wait_for_params()
            ...
            scf.cf.param.add_update_callback(
                group="deck", name="bcFlow2", cb=param_deck_flow_anon
            )
            time.sleep(1)
            logconf = LogConfig(
                name="Position", period_in_ms=self.config.log_interval_ms
            )

            for log_variable in self.config.log_variables:
                logconf.add_variable(log_variable, "float")

            scf.cf.log.add_config(logconf)
            logconf.data_received_cb.add_callback(self.log_pos_callback)

            try:
                logconf.start()
                self.control_loop(scf)
            finally:
                logconf.stop()

    def log_pos_callback(self, timestamp, data, logconf):
        self.state.position = Position(
            x=data["stateEstimate.x"],
            y=data["stateEstimate.y"],
            z=data["stateEstimate.z"],
        )
        self.communicator.broadcast_state(self.config.agent_id, self.state)

    def control_loop(self, scf):
        logging.info(f"CFDrone2D {self.hex_address} starts!")
        with MotionCommander(scf, default_height=self.config.default_height) as mc:
            logging.info(f"CFDrone2D {self.hex_address} | Taking off!")
            time.sleep(2)  # necessary for taking-off
            logging.info(f"CFDrone2D {self.hex_address} | Spawned | in air!")

            # register on communicator
            self.communicator.register_agent(self.config.agent_id)

            spent_time = 0
            while (
                    spent_time < self.config.total_time
            ) and self.communicator.is_active():
                self.controller.set_state(self.state)
                action: Action = self.controller.predict()
                command: VelocityCommand2D = self.action_to_command(action)
                command = command.scale(self.config.max_vel)
                mc.start_linear_motion(command.vel_x, command.vel_y, 0)
                duration: float = 1 / self.config.clock_freq
                time.sleep(duration)
                spent_time += duration

        logging.info(f"CFAgent {self.hex_address} finished!")

    @staticmethod
    def action_to_command(action: SimpleAction2D) -> VelocityCommand2D:
        if action == SimpleAction2D.FRONT_LEFT:
            return VelocityCommand2D(1, 1)

        if action == SimpleAction2D.FRONT:
            return VelocityCommand2D(1, 0)

        if action == SimpleAction2D.FRONT_RIGHT:
            return VelocityCommand2D(1, -1)

        if action == SimpleAction2D.RIGHT:
            return VelocityCommand2D(0, -1)

        if action == SimpleAction2D.BACK_RIGHT:
            return VelocityCommand2D(-1, -1)

        if action == SimpleAction2D.BACK:
            return VelocityCommand2D(-1, 0)

        if action == SimpleAction2D.BACK_LEFT:
            return VelocityCommand2D(-1, 1)

        if action == SimpleAction2D.LEFT:
            return VelocityCommand2D(0, 1)

        if action == SimpleAction2D.STOP:
            return VelocityCommand2D(0, 0)


def spawn_agent(init_config: ProcessInitConfig):
    """
    This method is the entrypoint for the agent process
    """
    logging.info(f"Spawn {init_config.class_name} agent {init_config.worker}!")
    agent_class: type[Agent] = globals()[init_config.class_name]
    assert issubclass(
        agent_class, Agent
    ), f"Communicator [{init_config.class_name}] was not found"
    agent: Agent = agent_class(**init_config.params)
    agent.run()
