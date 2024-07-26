from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
import pydantic


class Config(pydantic.BaseModel, ABC):
    ...


class Worker(str, Enum):
    Process = "Process"
    Thread = "Thread"


class ObjectInitConfig(Config):
    class_name: str
    params: dict = {}


class ProcessInitConfig(ObjectInitConfig):
    worker: Worker


class State(ABC):
    ...


class Action(str, Enum):
    ...


class Command(ABC):
    ...


class SimpleAction2D(Action):
    BACK_RIGHT = "BACK_RIGHT"
    BACK = "BACK"
    BACK_LEFT = "BACK_LEFT"

    RIGHT = "RIGHT"
    STOP = "STOP"
    LEFT = "LEFT"

    FRONT_RIGHT = "FRONT_RIGHT"
    FRONT = "FRONT"
    FRONT_LEFT = "FRONT_LEFT"


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


class SpaceLimit(pydantic.BaseModel):
    x_min: float | None = None
    x_max: float | None = None
    y_min: float | None = None
    y_max: float | None = None
    z_min: float | None = None
    z_max: float | None = None


class Position(pydantic.BaseModel):
    x: float | None = None
    y: float | None = None
    z: float | None = None

    def is_set(self):
        return not (self.x is None and self.y is None and self.z is None)

    def to_numpy(self) -> np.array:
        return [self.x, self.y, self.z]

    def to_numpy_2d(self) -> np.array:
        return [self.x, self.y]

    def __str__(self):
        return f"Position(x={self.x}, y={self.y}, z={self.z})"


class DroneAngle(pydantic.BaseModel):
    roll: float | None = None
    pitch: float | None = None
    yaw: float | None = None


class Field:
    data: np.array

    def __init__(self, data):
        self.data = data


class PositionState(State):
    position: Position

    def __init__(self, position: Position = None):
        self.position = position


class FieldState(PositionState):
    position: Position
    field: Field
    angle: DroneAngle

    def __init__(
            self, position: Position = None, field: Field = None, angle: DroneAngle = None
    ):
        super().__init__(position)
        self.field = field
        self.angle = angle


class FieldModulationEnvironmentState(State):
    modulations: np.array = np.array([])
    space_limit: SpaceLimit

    def __init__(self, modulations: np.array, space_limit: SpaceLimit):
        self.modulations = modulations
        self.space_limit = space_limit


class Communicator(ABC):
    config: Config

    def __init__(self, config: Config):
        self.config = config

    @abstractmethod
    def activate(self):
        ...

    @abstractmethod
    def stop(self, *kwargs):
        ...

    @abstractmethod
    def is_active(self):
        ...

    @abstractmethod
    def send(self, comm_key: str, info: object):
        ...

    @abstractmethod
    def recv(self, comm_key: str) -> object | None:
        ...

    @abstractmethod
    def register_agent(self, agent_id: str):
        ...

    @abstractmethod
    def deregister_agent(self, agent_id: str):
        ...

    @abstractmethod
    def broadcast_environment_state(self, state: State):
        ...

    @abstractmethod
    def broadcast_agent_state(self, agent_id: str, state: State):
        ...

    @abstractmethod
    def fetch_environment_state(self):
        ...

    @abstractmethod
    def fetch_agent_state(self, agent_id: str) -> State:
        ...

    @abstractmethod
    def fetch_registered_agents(self) -> list[str]:
        ...
