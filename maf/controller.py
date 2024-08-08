import os

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

import datetime
from abc import ABC, abstractmethod

import numpy as np
import pygame

from maf import field_modulation
from maf.core import (
    Config,
    Position,
    SimpleAction2D,
    State,
    ObjectInitConfig,
    Action,
    PositionState,
    FieldState,
)


class Controller(ABC):
    config: Config
    state: State

    def __init__(self, config: Config, state: State):
        self.config = config
        self.state = state

    @abstractmethod
    def predict(self) -> Action: ...

    @abstractmethod
    def set_state(self, state: State): ...


"""
###############################################################################
######## GoToPointController ##################################################
###############################################################################
"""


class GoToPointControllerConfig(Config):
    target_position: Position | None = None


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


"""
###############################################################################
######## KeyboardController ###################################################
###############################################################################
"""


class KeyboardControllerConfig(Config):
    agent_id: str = "Unknown"


class KeyboardController(Controller):
    WIDTH = 480
    HEIGHT = 480

    def __init__(self, config: dict | None = None):
        super().__init__(
            (
                KeyboardControllerConfig(**config)
                if config
                else KeyboardControllerConfig()
            ),
            None,
        )
        self.last_action: SimpleAction2D = SimpleAction2D.STOP
        self.pressed_keys: set[int] = set()

        pygame.init()
        pygame.time.delay(500)  # ms
        pygame.display.set_caption(
            f"{self.__class__.__name__} for agent [{self.config.agent_id}]"
        )
        self.display_surface = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.font = pygame.font.Font("freesansbold.ttf", 32)

    def predict(self) -> SimpleAction2D:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                self.pressed_keys.add(event.key)
            elif event.type == pygame.KEYUP:
                self.pressed_keys.discard(event.key)

        if pygame.K_SPACE in self.pressed_keys:
            self.last_action = SimpleAction2D.STOP
        else:
            if {pygame.K_UP, pygame.K_LEFT}.issubset(self.pressed_keys):
                self.last_action = SimpleAction2D.FRONT_LEFT
            elif {pygame.K_UP, pygame.K_RIGHT}.issubset(self.pressed_keys):
                self.last_action = SimpleAction2D.FRONT_RIGHT
            elif {pygame.K_DOWN, pygame.K_RIGHT}.issubset(self.pressed_keys):
                self.last_action = SimpleAction2D.BACK_RIGHT
            elif {pygame.K_DOWN, pygame.K_LEFT}.issubset(self.pressed_keys):
                self.last_action = SimpleAction2D.BACK_LEFT
            elif pygame.K_UP in self.pressed_keys:
                self.last_action = SimpleAction2D.FRONT
            elif pygame.K_RIGHT in self.pressed_keys:
                self.last_action = SimpleAction2D.RIGHT
            elif pygame.K_DOWN in self.pressed_keys:
                self.last_action = SimpleAction2D.BACK
            elif pygame.K_LEFT in self.pressed_keys:
                self.last_action = SimpleAction2D.LEFT

        self.update_display()
        return self.last_action

    def update_display(self):
        self.display_surface.fill((0, 0, 0))
        self.display_text_at_pos(self.last_action, (self.WIDTH // 2, self.HEIGHT // 2))
        self.display_text_at_pos(
            f"[{str(datetime.datetime.now())}]", (self.WIDTH // 2, self.HEIGHT // 4)
        )
        pygame.display.update()

    def display_text_at_pos(self, text: str, pos: tuple):
        green = (0, 255, 0)
        blue = (0, 0, 128)

        text = self.font.render(text, True, green, blue)
        text_rect = text.get_rect()
        text_rect.center = pos
        self.display_surface.blit(text, text_rect)

    def set_state(self, state: State):
        pass


"""
###############################################################################
######## HillClimbingController ###############################################
###############################################################################
"""


class HillClimbingControllerConfig(Config):
    agent_id: str = "Unknown"
    pooling_function: str = "sum"


class HillClimbingController(Controller):
    config: HillClimbingControllerConfig
    state: FieldState

    """
    BACK_RIGHT  |   RIGHT     | FRONT_RIGHT
    -------------------------------------
    BACK        |   STOP      | FRONT
    -------------------------------------
    BACK_LEFT   |   LEFT      | FRONT_LEFT
    """
    GRID_INDEX_TO_ACTION: dict = {
        0: SimpleAction2D.BACK_RIGHT,
        1: SimpleAction2D.RIGHT,
        2: SimpleAction2D.FRONT_RIGHT,
        3: SimpleAction2D.BACK,
        4: SimpleAction2D.STOP,
        5: SimpleAction2D.FRONT,
        6: SimpleAction2D.BACK_LEFT,
        7: SimpleAction2D.LEFT,
        8: SimpleAction2D.FRONT_LEFT,
    }

    def __init__(self, config: dict | None = None):
        super().__init__(
            (
                HillClimbingControllerConfig(**config)
                if config
                else HillClimbingControllerConfig()
            ),
            FieldState(),
        )

    def predict(self) -> SimpleAction2D:
        def all_elements_close(arr, tol=1e-3):
            return np.all(np.isclose(arr, arr[0], atol=tol))

        if self.state.field is None:
            return SimpleAction2D.STOP

        field: np.array = self.state.field.data
        grid: np.array = field_modulation.pooling_to_3x3(
            field, self.config.pooling_function
        )
        if all_elements_close(grid):
            return SimpleAction2D.STOP

        return self.GRID_INDEX_TO_ACTION[np.argmax(grid)]

    def set_state(self, state: State):
        self.state = state


"""
###############################################################################
######## HelloWorldController #################################################
###############################################################################
"""


class HelloWorldControllerConfig(Config):
    initial_state: int = 0


class HelloWorldController(Controller):
    config: HelloWorldControllerConfig
    state: int

    def __init__(self, config: dict | None = None):
        if config is None:
            config = {}

        config: HelloWorldControllerConfig = HelloWorldControllerConfig(**config)
        super().__init__(
            config=config,
            state=config.initial_state,
        )

    def predict(self) -> str:
        info: str = f"[{self.__class__.__name__} says hello {self.state}]"
        self.state += 1
        return info

    def set_state(self, state: State):
        pass


def get_controller(init_config: dict) -> Controller:
    init_config: ObjectInitConfig = ObjectInitConfig(**init_config)

    controller_class: type[Controller] = globals()[init_config.class_name]
    assert issubclass(
        controller_class, Controller
    ), f"Controller [{init_config.class_name}] was not found"
    return controller_class(**init_config.params)
