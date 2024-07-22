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


# class SimpleAction2D(Action):
#     """
#     Enum class representing various 2D actions with preserved ordering.
#     Do not change the order as it matched the 3x3 grid:
#
#     The grid representation is as follows:
#     BACK_LEFT  |   LEFT     |  FRONT_LEFT
#     -------------------------------------
#     BACK       |   STOP     |  FRONT
#     -------------------------------------
#     BACK_RIGHT |   RIGHT    |  FRONT_RIGHT
#     """
#
#     BACK_LEFT = "BACK_LEFT"
#     LEFT = "LEFT"
#     FRONT_LEFT = "FRONT_LEFT"
#
#     BACK = "BACK"
#     STOP = "STOP"
#     FRONT = "FRONT"
#
#     BACK_RIGHT = "BACK_RIGHT"
#     RIGHT = "RIGHT"
#     FRONT_RIGHT = "FRONT_RIGHT"

# class SimpleAction2D(Action):
#     """
#     Enum class representing various 2D actions with preserved ordering.
#     Do not change the order as it matched the 3x3 grid:
#
#     The grid representation is as follows:
#
#
#     FRONT_LEFT  |   FRONT     |  FRONT_RIGHT
#     -------------------------------------
#     LEFT       |   STOP     |  RIGHT
#     -------------------------------------
#     BACK_LEFT |   BACK    |  BACK_RIGHT
#     """
#     FRONT_LEFT = "FRONT_LEFT"
#     FRONT = "FRONT"
#     FRONT_RIGHT = "FRONT_RIGHT"
#
#     LEFT = "LEFT"
#     STOP = "STOP"
#     RIGHT = "RIGHT"
#
#     BACK_LEFT = "BACK_LEFT"
#     BACK = "BACK"
#     BACK_RIGHT = "BACK_RIGHT"


class SimpleAction2D(Action):
    """
    Enum class representing various 2D actions with preserved ordering.
    Do not change the order as it matched the 3x3 grid:

    The grid representation is as follows:


    BACK_RIGHT  |   BACK     | BACK_LEFT
    -------------------------------------
    RIGHT       |   STOP     |  LEFT
    -------------------------------------
    FRONT_RIGHT |   FRONT    |  FRONT_LEFT
    """

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


class Position(pydantic.BaseModel):
    x: float | None = None
    y: float | None = None
    z: float | None = None

    def is_set(self):
        return not (self.x is None and self.y is None and self.z is None)

    def to_numpy(self) -> np.array:
        return [self.x, self.y, self.z]

    def __str__(self):
        return f"Position(x={self.x}, y={self.y}, z={self.z})"


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

    def __init__(self, position: Position = None, field: Field = None):
        super().__init__(position)
        self.field = field


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
    def registered_agents(self) -> list[str]:
        ...

    @abstractmethod
    def broadcast_state(self, agent_id: str, state: State):
        ...

    @abstractmethod
    def broadcast_action(self, agent_id: str, action: Action):
        ...

    @abstractmethod
    def get_state(self, agent_id: str) -> State:
        ...

    @abstractmethod
    def get_action(self, agent_id: str) -> Action:
        ...
