from abc import ABC, abstractmethod

import pygame

from maf.core import (
    Config,
    Position,
    SimpleAction2D,
    State,
    ObjectInitConfig,
    Action,
    PositionState,
)


class Controller(ABC):
    config: Config
    state: State

    def __init__(self, config: Config, state: State):
        self.config = config
        self.state = state

    @abstractmethod
    def predict(self) -> Action:
        ...

    @abstractmethod
    def set_state(self, state: State):
        ...


class GoToPointControllerConfig(Config):
    target_position: Position | None = None
    box_limit: float = 0.5  # meters


class GoToPointController(Controller):
    config: GoToPointControllerConfig
    state: PositionState

    def __init__(self, config: dict):
        super().__init__(GoToPointControllerConfig(**config), PositionState())
        assert (
                self.config.target_position is not None
        ), f"Target position [target_position] must be specified for {self.__class__.__name__}"

    def set_state(self, state: PositionState):
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

        epsilon: float = 0.1
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
    def __init__(self):
        super().__init__(None, None)
        self.last_action: SimpleAction2D = SimpleAction2D.STOP
        pygame.init()
        pygame.display.set_mode((800, 600))

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

    def set_state(self, state: State):
        pass


def get_controller(init_config: dict) -> Controller:
    init_config: ObjectInitConfig = ObjectInitConfig(**init_config)

    controller_class: type[Controller] = globals()[init_config.class_name]
    assert issubclass(
        controller_class, Controller
    ), f"Controller [{init_config.class_name}] was not found"
    return controller_class(**init_config.params)
