import time
from enum import Enum

from threading import Event

import re
from abc import ABC, abstractmethod

import pydantic
import pygame
import torch
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
import pandas as pd

from core import get_communicator, Communicator


class State(ABC):
    ...


class ControllerConfig(pydantic.BaseModel, ABC):
    ...


class AgentConfig(pydantic.BaseModel, ABC):
    ...


class Position(pydantic.BaseModel):
    x: float | None = None
    y: float | None = None
    z: float | None = None

    def is_set(self):
        return not (self.x is None and self.y is None and self.z is None)

    def __str__(self):
        return f"Position(x={self.x}, y={self.y}, z={self.z})"


class Field:
    data: torch.Tensor

    def __init__(self, data):
        self.data = data


class FieldState(State):
    position: Position
    field: Field

    def __init__(self, position: Position = None, field: Field = None):
        self.position = position
        self.field = field


class CFControllerConfig(ControllerConfig):
    target_position: Position | None = None
    box_limit: float = 0.5  # meters


class CFAgent2DConfig(AgentConfig):
    agent_id: str
    clock_freq: int = 5  # hz
    default_height: float = 0.4
    max_vel: float = 0.1  # m/s
    communicator: str
    log_variables: list[str]
    log_interval_ms: int = 20  # ms
    total_time: int = 60  # seconds


class VirtualAgentConfig(AgentConfig):
    agent_id: str
    clock_freq: int = 5  # hz
    max_vel: float = 0.1  # m/s
    communicator: str
    total_time: int = 60  # seconds


class Action(str, Enum):
    ...


class Command(ABC):
    ...


class Controller(ABC):
    def __init__(self, *kwargs):
        ...

    @abstractmethod
    def predict(self) -> Action:
        ...

    @abstractmethod
    def set_state(self, state: State):
        ...


class Agent(ABC):
    def __init__(self, config: AgentConfig, controller: Controller, *kwargs):
        self.config = config
        self.controller: Controller = controller
        ...

    @abstractmethod
    def run(self):
        ...

    @staticmethod
    @abstractmethod
    def action_to_command(action: Action) -> Command:
        ...

    @abstractmethod
    def set_state(self, state: State):
        ...

    @abstractmethod
    def get_state(self) -> State:
        ...


class SimpleAction2D(Action):
    FRONT = "FRONT"
    RIGHT = "RIGHT"
    BACK = "BACK"
    LEFT = "LEFT"
    STOP = "STOP"


class VelocityCommand2D(Command):
    def __init__(self, vel_x: float, vel_y: float):
        self.vel_x = vel_x
        self.vel_y = vel_y

    def scale(self, scale_factor: float) -> Command:
        self.vel_y *= scale_factor
        self.vel_x *= scale_factor
        return self

    def __str__(self):
        return f"CFCommand(vel_x={self.vel_x}, vel_y={self.vel_y})"


class GoToPointController(Controller):
    def __init__(self, config: CFControllerConfig):
        super().__init__()
        self.config = config
        assert (
                self.config.target_position is not None
        ), f"Target position [target_position] must be specified for {self.__class__.__name__}"
        self.state: FieldState = FieldState()

    def set_state(self, state: FieldState):
        self.state = state

    def predict(self) -> SimpleAction2D:
        if not self.state.position.is_set():
            raise Exception("Could not predict, unknown [position]!")
        if not self.config.target_position.is_set():
            raise Exception("Could not predict, unknown [target_position]!")

        current_x = self.state.position.x
        current_y = self.state.position.y
        target_x = self.config.target_position.x
        target_y = self.config.target_position.y

        delta_x = target_x - current_x
        delta_y = target_y - current_y

        epsilon: float = 10e-6
        if abs(delta_x) < epsilon and abs(delta_y) < epsilon:
            return SimpleAction2D.STOP

        if abs(delta_x) > abs(delta_y):
            if delta_x > 0:
                return SimpleAction2D.FRONT
            elif delta_x < 0:
                return SimpleAction2D.BACK
        else:
            if delta_y > 0:
                return SimpleAction2D.LEFT
            elif delta_y < 0:
                return SimpleAction2D.RIGHT


class KeyboardController(Controller):
    def __init__(self, config: CFControllerConfig):
        super().__init__()
        self.config = config
        self.state: FieldState = FieldState()
        self.last_action: SimpleAction2D = SimpleAction2D.STOP
        pygame.init()
        pygame.display.set_mode((800, 600))

    def set_state(self, state: FieldState):
        ...

    # @override
    def predict(self) -> SimpleAction2D:
        events = pygame.event.get()
        for event in events:

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.last_action = SimpleAction2D.FRONT
                    break

                if event.key == pygame.K_DOWN:
                    self.last_action = SimpleAction2D.BACK
                    break

                if event.key == pygame.K_LEFT:
                    self.last_action = SimpleAction2D.LEFT
                    break

                if event.key == pygame.K_RIGHT:
                    self.last_action = SimpleAction2D.RIGHT
                    break

                if event.key == pygame.K_SPACE:
                    self.last_action = SimpleAction2D.STOP
                    break

        return self.last_action


class CFAgent2D(Agent):
    def __init__(self, config: CFAgent2DConfig, controller: Controller):
        super().__init__(config, controller)
        cflib.crtp.init_drivers()
        # e.g. `radio://0/100/2M/E7E7E7E704`
        assert re.match(
            r"radio://\d/\d{1,3}/\dM/[A-Z0-9]{10}", self.config.agent_id
        ), f"Invalid agent id [{self.config.agent_id}] for crazyflie"
        self.hex_address: str = self.config.agent_id[-10:]
        self.deck_attached_event: Event = Event()
        self.state: FieldState = FieldState()
        self.communicator: Communicator = get_communicator(self.config.communicator)
        self.df: pd.DataFrame = pd.DataFrame(columns=self.config.log_variables)

    def run(self):
        def param_deck_flow_anon(_, value_str):
            value = int(value_str)
            print(value)
            if value:
                self.deck_attached_event.set()
                print(f"CFAgent [{self.hex_address}] | Deck flow is attached!")
            else:
                print(f"CFAgent [{self.hex_address}] | Deck flow is NOT attached!")

        print(f"Initializing [{self.hex_address}]...")
        with SyncCrazyflie(self.config.agent_id, Crazyflie(rw_cache=f"./cache_{self.hex_address}")) as scf:
            scf.wait_for_params()
            ...
            scf.cf.param.add_update_callback(
                group="deck", name="bcFlow2", cb=param_deck_flow_anon
            )
            time.sleep(1)
            logconf = LogConfig(name="Position", period_in_ms=self.config.log_interval_ms)
            [
                logconf.add_variable(log_variable, "float")
                for log_variable in self.config.log_variables
            ]

            scf.cf.log.add_config(logconf)
            logconf.data_received_cb.add_callback(self.log_pos_callback)

            try:
                logconf.start()
                self.control_loop(scf)
            finally:
                logconf.stop()

    def log_pos_callback(self, timestamp, data, logconf):
        row_df = pd.DataFrame(data, index=[0])
        # self.df = pd.concat([self.df, row_df])
        self.state.position = Position(
            x=data["stateEstimate.x"],
            y=data["stateEstimate.y"],
            z=data["stateEstimate.z"],
        )

    def control_loop(self, scf):
        print(f"CFAgent {self.hex_address} starts!")
        with MotionCommander(scf, default_height=self.config.default_height) as mc:
            print(f"CFAgent {self.hex_address} | Taking off!")
            time.sleep(2)  # necessary for taking-off
            print(f"CFAgent {self.hex_address} | in air!")

            spent_time = 0
            while (spent_time < self.config.total_time) and not self.should_stop():
                self.controller.set_state(self.state)
                action: Action = self.controller.predict()
                command: VelocityCommand2D = self.action_to_command(action)
                command = command.scale(self.config.max_vel)
                mc.start_linear_motion(command.vel_x, command.vel_y, 0)
                duration: float = 1 / self.config.clock_freq
                time.sleep(duration)
                spent_time += duration
                print(f"current_position = {self.state.position} | action = [{action}] command = {command}")

        self.log_experiment()
        print(f"CFAgent {self.hex_address} finished!")

    @staticmethod
    def action_to_command(action: SimpleAction2D) -> VelocityCommand2D:
        if action == SimpleAction2D.FRONT:
            return VelocityCommand2D(1, 0)

        if action == SimpleAction2D.BACK:
            return VelocityCommand2D(-1, 0)

        if action == SimpleAction2D.LEFT:
            return VelocityCommand2D(0, 1)

        if action == SimpleAction2D.RIGHT:
            return VelocityCommand2D(0, -1)

        if action == SimpleAction2D.STOP:
            return VelocityCommand2D(vel_x=0, vel_y=0)

    def get_state(self) -> State:
        return self.state

    def set_state(self, state: State):
        self.state = state

    def should_stop(self) -> bool:
        return self.communicator.recv(Communicator.CommKey.STOP_EVENT) is True

    def log_experiment(self):
        self.df.to_csv(f"{self.hex_address}.csv", index=False)


class VirtualAgent(Agent):
    def __init__(self, config: CFAgent2DConfig, controller: Controller):
        super().__init__(config, controller)
        self.communicator: Communicator = get_communicator(self.config.communicator)
        self.df: pd.DataFrame = pd.DataFrame(columns=self.config.log_variables)

    def run(self):
        print(f"Starting [{self.config.agent_id}]!")
        spent_time = 0
        while (spent_time < self.config.total_time) and not self.should_stop():
            state: FieldState = self.get_state()
            self.controller.set_state(state)

            action: SimpleAction2D = self.controller.predict()
            command: VelocityCommand2D = self.action_to_command(action)
            command = command.scale(self.config.max_vel)

            new_state = self.compute_new_state(state, command)
            self.set_state(new_state)

            duration: float = 1 / self.config.clock_freq
            time.sleep(duration)
            spent_time += duration
            print(f"current_position = {state.position} | action = [{action}] command = {command}")

        print(f"Finished [{self.config.agent_id}]!")

    def compute_new_state(self, old_state: FieldState, command: VelocityCommand2D) -> FieldState:
        duration: float = 1 / self.config.clock_freq
        old_position: Position = old_state.position
        new_position: Position = Position(
            x=old_position.x + command.vel_x * duration,
            y=old_position.y + command.vel_y * duration
        )
        return FieldState(
            position=new_position,
            field=old_state.field
        )

    @staticmethod
    def action_to_command(action: SimpleAction2D) -> VelocityCommand2D:
        if action == SimpleAction2D.FRONT:
            return VelocityCommand2D(1, 0)

        if action == SimpleAction2D.BACK:
            return VelocityCommand2D(-1, 0)

        if action == SimpleAction2D.LEFT:
            return VelocityCommand2D(0, 1)

        if action == SimpleAction2D.RIGHT:
            return VelocityCommand2D(0, -1)

        if action == SimpleAction2D.STOP:
            return VelocityCommand2D(vel_x=0, vel_y=0)

    def should_stop(self) -> bool:
        return self.communicator.recv(Communicator.CommKey.STOP_EVENT) is True

    def set_state(self, state: FieldState) -> None:
        state_key: str = Communicator.get_state_key(self.config.agent_id)
        self.communicator.send(state_key, state)

    def get_state(self) -> FieldState:
        state_key: str = Communicator.get_state_key(self.config.agent_id)
        state: FieldState = self.communicator.recv(state_key)
        return state


def spawn_agent(
        agent: str,
        agent_config: str,
        agent_config_data: dict,
        controller: str,
        controller_config: str,
        controller_config_data: dict
):
    """
    This method is the entrypoint for the agent thread
    """
    controller_class: type[Controller] = globals()[controller]
    controller_config_class: type[ControllerConfig] = globals()[controller_config]
    controller: Controller = controller_class(controller_config_class(**controller_config_data))

    agent_class: type[Agent] = globals()[agent]
    agent_config_class: type[AgentConfig] = globals()[agent_config]
    agent = agent_class(
        config=agent_config_class(**agent_config_data),
        controller=controller
    )

    agent.run()
