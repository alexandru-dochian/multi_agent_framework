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

from maf import field_modulation, logger_config
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
    DroneAngle,
    FieldModulationEnvironmentState,
)

logger: logging.Logger = logger_config.get_logger(__name__)

# TODO: refactor
VICINITY_LIMIT = [0.5, 0.5]
FIELD_SIZE: list[int] = [84, 84]


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


"""
###############################################################################
######## VirtualDrone2D #######################################################
###############################################################################
"""


class VirtualDrone2DConfig(Config):
    delay: int = 100  # ms
    agent_id: str
    max_vel: float = 0.1  # m/s
    default_height: float = 0.4  # m
    total_time: int = 60  # seconds
    initial_position: Position = Position(x=0, y=0, z=0.4)


class VirtualDrone2D(Agent):
    config: VirtualDrone2DConfig
    state: FieldState
    controller: Controller
    communicator: Communicator

    def __init__(self, config: dict, controller: dict, communicator: dict):
        config: VirtualDrone2DConfig = VirtualDrone2DConfig(**config)
        state: FieldState = FieldState(
            position=config.initial_position,
            field=Field(data=np.zeros(FIELD_SIZE)),
        )
        super().__init__(
            config,
            state,
            get_controller(controller),
            get_communicator(communicator),
        )

    def run(self):
        logger.info(f"Starting [{self.config.agent_id}]!")

        # register on communicator
        self.communicator.register_agent(self.config.agent_id)

        # register initial state
        self.communicator.broadcast_agent_state(
            self.config.agent_id,
            self.state,
        )

        spent_time = 0
        while (spent_time < self.config.total_time) and self.communicator.is_active():
            self.controller.set_state(self.state)
            action: SimpleAction2D = self.controller.predict()
            command: VelocityCommand2D = self.action_to_command(action)
            command = command.scale(self.config.max_vel)

            self.state = self.compute_new_state(self.state, command)
            self.communicator.broadcast_agent_state(self.config.agent_id, self.state)

            delay_seconds = self.config.delay / 1000
            time.sleep(delay_seconds)
            spent_time += delay_seconds

        logger.info(f"Finished [{self.config.agent_id}]!")

    def compute_new_state(
        self, old_state: FieldState, command: VelocityCommand2D
    ) -> FieldState:
        delay_seconds: float = self.config.delay / 1000

        new_position: Position = Position(
            x=old_state.position.x + command.vel_x * delay_seconds,
            y=old_state.position.y + command.vel_y * delay_seconds,
            z=self.config.default_height,
        )

        return FieldState(
            position=new_position, field=Field(data=self.generate_field(new_position))
        )

    def generate_field(self, position: Position) -> np.array:
        environment_state: FieldModulationEnvironmentState = (
            self.communicator.fetch_environment_state()
        )
        if environment_state is None:
            return np.zeros(FIELD_SIZE)

        return field_modulation.generate_field(
            field_size=FIELD_SIZE,
            center=[position.x, position.y],
            neighbours=self.get_neighbours(),
            modulations=environment_state.modulations,
            vicinity_limit=VICINITY_LIMIT,
            space_limit=environment_state.space_limit,
        )

    def get_neighbours(self):
        neighbours: list = []
        for neighbour_agent_id in self.communicator.fetch_registered_agents():
            if neighbour_agent_id == self.config.agent_id:
                # we are interested only in neighbouring agents
                continue

            neighbour_agent_state: PositionState = self.communicator.fetch_agent_state(
                neighbour_agent_id
            )
            neighbours.append(
                [neighbour_agent_state.position.x, neighbour_agent_state.position.y]
            )

        return np.array(neighbours)

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


"""
###############################################################################
######## CFDrone2D ############################################################
###############################################################################
"""


class CFDrone2DConfig(Config):
    delay: int = 100  # ms
    agent_id: str
    default_height: float = 0.4  # m
    max_vel: float = 0.1  # m/s
    log_variables: list[str]
    log_interval_ms: int = 50  # ms
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
        self.lighthouse_deck_attached_event: Event = Event()

    def run(self):
        def param_deck_flow_anon(_, value_str):
            if int(value_str):
                self.lighthouse_deck_attached_event.set()

        logger.info(f"Initializing [{self.hex_address}]...")
        with SyncCrazyflie(
            self.config.agent_id, Crazyflie(rw_cache=f"./cache/{self.hex_address}")
        ) as scf:
            scf.cf.param.add_update_callback(
                group="deck", name="bcLighthouse4", cb=param_deck_flow_anon
            )
            scf.wait_for_params()

            if not self.lighthouse_deck_attached_event.wait(timeout=2):
                raise Exception(f"Lighthouse deck missing on {self.hex_address}!")

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
        # https://www.bitcraze.io/documentation/repository/crazyflie-firmware/master/api/logs/
        new_position: Position = Position(
            x=data["stateEstimate.x"],
            y=data["stateEstimate.y"],
            z=data["stateEstimate.z"],
        )
        new_angle: DroneAngle = DroneAngle(yaw=data["stateEstimate.yaw"])
        new_field = self.generate_field(new_position)

        self.state = FieldState(position=new_position, field=new_field, angle=new_angle)
        self.communicator.broadcast_agent_state(self.config.agent_id, self.state)

    def generate_field(self, position: Position) -> Field:
        environment_state: FieldModulationEnvironmentState = (
            self.communicator.fetch_environment_state()
        )
        field: np.array = field_modulation.generate_field(
            field_size=[84, 84],
            center=[position.x, position.y],
            neighbours=self.get_neighbours(),
            modulations=environment_state.modulations,
            vicinity_limit=[1, 1],
            space_limit=environment_state.space_limit,
        )

        return Field(data=field)

    def get_neighbours(self):
        neighbours: list = []
        for neighbour_agent_id in self.communicator.fetch_registered_agents():
            if neighbour_agent_id == self.config.agent_id:
                # we are interested only in neighbouring agents
                continue

            neighbour_agent_state: PositionState = self.communicator.fetch_agent_state(
                neighbour_agent_id
            )
            neighbours.append(
                [neighbour_agent_state.position.x, neighbour_agent_state.position.y]
            )

        return np.array(neighbours)

    def control_loop(self, scf):
        logger.info(f"CFDrone2D {self.hex_address} starts!")
        with MotionCommander(scf, default_height=self.config.default_height) as mc:
            logger.info(f"CFDrone2D {self.hex_address} | Taking off!")
            time.sleep(2)  # necessary for taking-off
            logger.info(f"CFDrone2D {self.hex_address} | Spawned | in air!")

            # register on communicator
            self.communicator.register_agent(self.config.agent_id)
            spent_time = 0
            while (
                spent_time < self.config.total_time
            ) and self.communicator.is_active():
                # state gets updated asynchronously in `self.log_pos_callback`
                self.controller.set_state(self.state)
                action: Action = self.controller.predict()
                command: VelocityCommand2D = self.action_to_command(action)
                command = command.scale(self.config.max_vel)
                mc.start_linear_motion(
                    command.vel_x, command.vel_y, 0, self.command_yaw_change()
                )

                delay_seconds = self.config.delay / 1000
                time.sleep(delay_seconds)
                spent_time += delay_seconds

        logger.info(f"CFDrone2D {self.hex_address} finished!")

    def command_yaw_change(
        self, target_angles=None, rate_of_change: float = 5, epsilon: float = 3
    ) -> float:
        if target_angles is None:
            # Will keep the drone pointing on the ox axis
            target_angles = [-180, 180]

        def closest_target(yaw):
            return min(target_angles, key=lambda t: (yaw - t + 180) % 360 - 180)

        target_yaw = closest_target(self.state.angle.yaw)

        def normalize_angle(angle):
            while angle > 180:
                angle -= 360
            while angle < -180:
                angle += 360
            return angle

        angle_difference = normalize_angle(target_yaw - self.state.angle.yaw)
        if 180 - abs(angle_difference) < epsilon:
            return 0

        if angle_difference > 0:
            return min(angle_difference, rate_of_change)
        else:
            return max(angle_difference, -rate_of_change)

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


"""
###############################################################################
######## HelloWorldAgent ######################################################
###############################################################################
"""


class HelloWorldAgentConfig(Config):
    ...


class HelloWorldAgent(Agent):
    config: VirtualDrone2DConfig
    state: State
    controller: Controller
    communicator: Communicator

    def __init__(self, config: dict, controller: dict, communicator: dict):
        config: VirtualDrone2DConfig = VirtualDrone2DConfig(**config)
        state: FieldState = FieldState(
            position=config.initial_position,
            field=Field(data=np.zeros(FIELD_SIZE)),
        )
        super().__init__(
            config,
            state,
            get_controller(controller),
            get_communicator(communicator),
        )

    def run(self):
        logger.info(self.controller.predict())


def spawn_agent(init_config: ProcessInitConfig):
    """
    This method is the entrypoint for the agent process
    """
    logger.info(f"Spawn {init_config.class_name} agent {init_config.worker}!")
    agent_class: type[Agent] = globals()[init_config.class_name]
    assert issubclass(
        agent_class, Agent
    ), f"Communicator [{init_config.class_name}] was not found"
    agent: Agent = agent_class(**init_config.params)
    agent.run()
