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
from scipy.ndimage import zoom

from maf import utils
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
    default_height: float = 0.4  # m
    total_time: int = 60  # seconds
    start_position: Position = Position(x=0, y=0, z=0)


class VirtualDrone2D(Agent):
    config: VirtualDrone2DConfig
    state: FieldState
    controller: Controller
    communicator: Communicator

    # TODO config
    field_size: list[int] = [84, 84]

    # TODO Environment
    field_modulation = np.array([[0, 0], [1, 1], [-1, 1], [1, -1], [-1, -1]])

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

        factor_vicinity = 1
        core_vicinity_limit = np.array([2, 2])
        vicinity_limit = core_vicinity_limit * factor_vicinity

        sigma_scale = 15 / factor_vicinity
        amplitude = 1 / factor_vicinity

        center_point: list = np.array([new_position.x, new_position.y])
        # Find neighboring points within the box limit of vicinity_limit
        x_min, x_max = (
            center_point[0] - vicinity_limit[0],
            center_point[0] + vicinity_limit[0],
        )
        y_min, y_max = (
            center_point[1] - vicinity_limit[1],
            center_point[1] + vicinity_limit[1],
        )

        others: list = []
        for other_agent_id in self.communicator.registered_agents():
            if other_agent_id == self.config.agent_id:
                # we are interested only in other agents
                continue

            other_agent_state: PositionState = self.communicator.get_state(
                other_agent_id
            )
            others.append([other_agent_state.position.x, other_agent_state.position.y])
        others = np.array(others)
        if len(others) > 0:
            filtered_others = others[
                (others[:, 0] >= x_min)
                & (others[:, 0] <= x_max)
                & (others[:, 1] >= y_min)
                & (others[:, 1] <= y_max)
            ]
        else:
            filtered_others = others

        # Step 4: Project the points onto a centered torch.Tensor of shape (tensor_size[0], tensor_size[1])
        field: np.array = np.zeros(self.field_size)

        # Apply Gaussian to each neighboring point, excluding the center point
        for point in filtered_others:
            if np.all(point == center_point):
                continue

            x_tensor, y_tensor = self.to_tensor_space(
                point, center_point, self.field_size, vicinity_limit
            )

            # Ensure the indices are within the bounds of the tensor
            if (
                0 <= x_tensor < self.field_size[1]
                and 0 <= y_tensor < self.field_size[0]
            ):
                field = utils.apply_distribution(
                    field,
                    (x_tensor, y_tensor),
                    sigma=sigma_scale,
                    amplitude=amplitude,
                    operation="subtract",
                )
            else:
                print(
                    f"Point {point} projected to out-of-bounds tensor coordinates ({x_tensor}, {y_tensor})"
                )

        def rotate_points(points, theta, center):
            # Convert theta to radians
            theta_rad = np.radians(theta)

            # Define the rotation matrix
            rotation_matrix = np.array(
                [
                    [np.cos(theta_rad), -np.sin(theta_rad)],
                    [np.sin(theta_rad), np.cos(theta_rad)],
                ]
            )

            # Translate points to the origin (subtract the center)
            translated_points = points - center

            # Rotate each point
            rotated_points = np.dot(translated_points, rotation_matrix.T)

            # Translate points back to the original center
            rotated_points += center

            return rotated_points

        theta = 1
        rot_center = [0.5, 0.5]
        self.field_modulation = rotate_points(self.field_modulation, theta, rot_center)
        field_modulation = self.field_modulation

        field_modulation_filtered = field_modulation[
            (field_modulation[:, 0] >= x_min)
            & (field_modulation[:, 0] <= x_max)
            & (field_modulation[:, 1] >= y_min)
            & (field_modulation[:, 1] <= y_max)
        ]
        # Apply field_modulation
        for point in field_modulation_filtered:
            x_tensor, y_tensor = self.to_tensor_space(
                point, center_point, self.field_size, vicinity_limit
            )

            # Ensure the indices are within the bounds of the tensor
            if (
                0 <= x_tensor < self.field_size[1]
                and 0 <= y_tensor < self.field_size[0]
            ):
                field = utils.apply_distribution(
                    field,
                    (x_tensor, y_tensor),
                    sigma=sigma_scale / 1.5,
                    amplitude=amplitude,
                    operation="add",
                )
            else:
                print(
                    f"Point {point} projected to out-of-bounds tensor coordinates ({x_tensor}, {y_tensor})"
                )

        # Clip and resize the tensor
        clip_size = [self.field_size[0] // 2, self.field_size[1] // 2]
        start_x = self.field_size[1] // 2 - clip_size[1] // 2
        start_y = self.field_size[0] // 2 - clip_size[0] // 2

        clipped_tensor = field[
            start_y : start_y + clip_size[0], start_x : start_x + clip_size[1]
        ]
        rescaled_tensor = zoom(
            input=clipped_tensor,
            zoom=(self.field_size[0] / clip_size[0], self.field_size[1] / clip_size[1]),
            order=1,
        )

        return FieldState(position=new_position, field=Field(data=rescaled_tensor))

    @staticmethod
    def to_tensor_space(point, center_point, tensor_size, vicinity_limit):
        x, y = point
        cx, cy = center_point
        # Normalize to [0, 1] range first
        x_normalized = (x - (cx - vicinity_limit[0])) / (2 * vicinity_limit[0])
        y_normalized = (y - (cy - vicinity_limit[1])) / (2 * vicinity_limit[1])
        # Convert to tensor space indices
        x_tensor = int(round(x_normalized * (tensor_size[1] - 1)))
        y_tensor = int(round(y_normalized * (tensor_size[0] - 1)))
        # Clamp indices to ensure they are within bounds
        x_tensor = np.clip(x_tensor, 0, tensor_size[1] - 1)
        y_tensor = np.clip(y_tensor, 0, tensor_size[0] - 1)
        return x_tensor, y_tensor

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
    default_height: float = 0.4  # m
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
    yaw: float

    # TODO config
    field_size: list[int] = [84, 84]

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
                logging.info(
                    f"CFDrone2D [{self.hex_address}] | Deck flow is NOT attached!"
                )

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

    @staticmethod
    def to_tensor_space(point, center_point, tensor_size, vicinity_limit):
        x, y = point
        cx, cy = center_point
        # Normalize to [0, 1] range first
        x_normalized = (x - (cx - vicinity_limit[0])) / (2 * vicinity_limit[0])
        y_normalized = (y - (cy - vicinity_limit[1])) / (2 * vicinity_limit[1])
        # Convert to tensor space indices
        x_tensor = int(round(x_normalized * (tensor_size[1] - 1)))
        y_tensor = int(round(y_normalized * (tensor_size[0] - 1)))
        # Clamp indices to ensure they are within bounds
        x_tensor = np.clip(x_tensor, 0, tensor_size[1] - 1)
        y_tensor = np.clip(y_tensor, 0, tensor_size[0] - 1)
        return x_tensor, y_tensor

    def log_pos_callback(self, timestamp, data, logconf):
        # https://www.bitcraze.io/documentation/repository/crazyflie-firmware/master/api/logs/
        self.yaw: float = data["stateEstimate.yaw"]

        new_position: Position = Position(
            x=data["stateEstimate.x"],
            y=data["stateEstimate.y"],
            z=data["stateEstimate.z"],
        )

        factor_vicinity = 1
        core_vicinity_limit = np.array([1.5, 1.5])
        vicinity_limit = core_vicinity_limit * factor_vicinity

        sigma_scale = 15 / factor_vicinity
        amplitude = 1 / factor_vicinity

        current: list = [new_position.x, new_position.y]
        center_point: list = np.array(current)
        # Find neighboring points within the box limit of vicinity_limit
        x_min, x_max = (
            center_point[0] - vicinity_limit[0],
            center_point[0] + vicinity_limit[0],
        )
        y_min, y_max = (
            center_point[1] - vicinity_limit[1],
            center_point[1] + vicinity_limit[1],
        )

        others: list = []
        for other_agent_id in self.communicator.registered_agents():
            if other_agent_id == self.config.agent_id:
                # we are interested only in other agents
                continue

            other_agent_state: PositionState = self.communicator.get_state(
                other_agent_id
            )
            others.append([other_agent_state.position.x, other_agent_state.position.y])
        others = np.array(others)
        if len(others) > 0:
            filtered_others = others[
                (others[:, 0] >= x_min)
                & (others[:, 0] <= x_max)
                & (others[:, 1] >= y_min)
                & (others[:, 1] <= y_max)
            ]
        else:
            filtered_others = others

        # Step 4: Project the points onto a centered torch.Tensor of shape (tensor_size[0], tensor_size[1])
        field: np.array = np.zeros(self.field_size)

        # Apply Gaussian to each neighboring point, excluding the center point
        for point in filtered_others:
            if np.all(point == center_point):
                continue

            x_tensor, y_tensor = self.to_tensor_space(
                point, center_point, self.field_size, vicinity_limit
            )

            # Ensure the indices are within the bounds of the tensor
            if (
                0 <= x_tensor < self.field_size[1]
                and 0 <= y_tensor < self.field_size[0]
            ):
                field = utils.apply_distribution(
                    field,
                    (x_tensor, y_tensor),
                    sigma=sigma_scale,
                    amplitude=amplitude,
                    operation="subtract",
                )
            else:
                print(
                    f"Point {point} projected to out-of-bounds tensor coordinates ({x_tensor}, {y_tensor})"
                )

        # TODO GET FROM ENVIRONMENT
        field_modulation = np.array(
            [
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1],
            ]
        )
        field_modulation_filtered = field_modulation[
            (field_modulation[:, 0] >= x_min)
            & (field_modulation[:, 0] <= x_max)
            & (field_modulation[:, 1] >= y_min)
            & (field_modulation[:, 1] <= y_max)
        ]
        # Apply field_modulation
        for point in field_modulation_filtered:
            x_tensor, y_tensor = self.to_tensor_space(
                point, center_point, self.field_size, vicinity_limit
            )

            # Ensure the indices are within the bounds of the tensor
            if (
                0 <= x_tensor < self.field_size[1]
                and 0 <= y_tensor < self.field_size[0]
            ):
                field = utils.apply_distribution(
                    field,
                    (x_tensor, y_tensor),
                    sigma=sigma_scale / 2,
                    amplitude=amplitude,
                    operation="add",
                )
            else:
                print(
                    f"Point {point} projected to out-of-bounds tensor coordinates ({x_tensor}, {y_tensor})"
                )

        # Clip and resize the tensor
        clip_size = [self.field_size[0] // 2, self.field_size[1] // 2]
        start_x = self.field_size[1] // 2 - clip_size[1] // 2
        start_y = self.field_size[0] // 2 - clip_size[0] // 2

        clipped_tensor = field[
            start_y : start_y + clip_size[0], start_x : start_x + clip_size[1]
        ]
        rescaled_tensor = zoom(
            input=clipped_tensor,
            zoom=(self.field_size[0] / clip_size[0], self.field_size[1] / clip_size[1]),
            order=1,
        )

        new_state: FieldState = FieldState(
            position=new_position, field=Field(data=rescaled_tensor)
        )
        self.state = new_state
        self.communicator.broadcast_state(self.config.agent_id, self.state)

    def control_loop(self, scf):
        logging.info(f"CFDrone2D {self.hex_address} starts!")
        with MotionCommander(scf, default_height=self.config.default_height) as mc:
            logging.info(f"CFDrone2D {self.hex_address} | Taking off!")
            time.sleep(2)  # necessary for taking-off
            logging.info(f"CFDrone2D {self.hex_address} | Spawned | in air!")

            # register on communicator
            self.communicator.register_agent(self.config.agent_id)
            print("self.yaw", self.yaw)
            spent_time = 0
            while (
                spent_time < self.config.total_time
            ) and self.communicator.is_active():
                self.controller.set_state(self.state)
                action: Action = self.controller.predict()
                command: VelocityCommand2D = self.action_to_command(action)
                command = command.scale(self.config.max_vel)
                mc.start_linear_motion(command.vel_x, command.vel_y, 0, 20)
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
