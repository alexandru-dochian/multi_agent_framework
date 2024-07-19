import numpy as np

from abc import ABC, abstractmethod

import torch

import persistence
from communicator import Communicator, get_communicator
from core import ProcessInitConfig, Config, PositionState, FieldState, Field


class LogHandler(ABC):
    config: Config
    communicator: Communicator

    def __init__(self, config: Config, communicator: Communicator):
        self.config = config
        self.communicator = communicator

    @abstractmethod
    def run(self):
        ...


class PositionLoggerConfig(Config):
    experiment_dir: str
    gui_enabled: bool = True
    delay: int = 500  # ms


class PositionLogger(LogHandler):
    config: PositionLoggerConfig
    communicator: Communicator

    scene: object
    animator: object
    points: object

    def __init__(self, config: dict, communicator: dict) -> None:
        super().__init__(PositionLoggerConfig(**config), get_communicator(communicator))

    def run(self):
        # For some reason mayavi imports need to be at method level
        from mayavi import mlab
        from mayavi.core.scene import Scene
        from mayavi.tools.animator import Animator
        from mayavi.tools.helper_functions import Points3d

        @mlab.animate(delay=self.config.delay)
        def anim():
            try:
                while self.communicator.is_active():
                    x_new = []
                    y_new = []
                    z_new = []
                    for agent in self.communicator.registered_agents():
                        state: PositionState = self.communicator.get_state(agent)
                        x_new.append(state.position.x)
                        y_new.append(state.position.y)
                        z_new.append(state.position.z)
                        self.write_to_disk(agent, state)

                    x_new = np.array(x_new)
                    y_new = np.array(y_new)
                    z_new = np.array(z_new)

                    self.points.mlab_source.reset(x=x_new, y=y_new, z=z_new)
                    self.scene.scene.reset_zoom()
                    yield
            finally:
                mlab.clf()
                self.animator.close()
                mlab.close(all=True)

        self.scene: Scene = mlab.figure(
            "PositionLogger", bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(800, 600)
        )

        mlab.clf()
        mlab.view(0, 180, None, (0, 0, 0))

        x, y, z = np.random.random((3, 0))
        self.points: Points3d = mlab.points3d(
            x, y, z, scale_factor=0.1, color=(0, 1, 0)
        )

        self.animator: Animator = anim()
        mlab.orientation_axes(line_width=1, xlabel="X", ylabel="Y", zlabel="Z")
        mlab.show()

    def write_to_disk(self, agent: str, state: PositionState):
        persistence.store(
            self.config.experiment_dir,
            self.__class__.__name__,
            {
                'agent': agent,
                'state': state
            }
        )


class FieldStateLoggerConfig(Config):
    agent_id: str
    experiment_dir: str
    gui_enabled: bool = True
    delay: int = 500  # ms


class FieldStateLogger(LogHandler):
    config: FieldStateLoggerConfig
    communicator: Communicator

    scene: object
    animator: object
    surface: object

    def __init__(self, config: dict, communicator: dict) -> None:
        super().__init__(FieldStateLoggerConfig(**config), get_communicator(communicator))

    def run(self):
        # For some reason mayavi imports need to be at method level
        from mayavi import mlab
        from mayavi.core.scene import Scene
        from mayavi.tools.animator import Animator
        from mayavi.modules.surface import Surface

        @mlab.animate(delay=self.config.delay)
        def anim():
            try:
                while self.communicator.is_active():
                    field_state: FieldState = FieldState(
                        field=Field(data=torch.randn(84, 84))
                    )
                    self.surface.mlab_source.reset(scalars=field_state.field.data.numpy())
                    self.scene.scene.reset_zoom()
                    self.write_to_disk(self.config.agent_id, field_state)
                    yield
            finally:
                mlab.clf()
                self.animator.close()
                mlab.close(all=True)

        self.scene: Scene = mlab.figure(
            f"FieldState for [{self.config.agent_id}]", bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(400, 300)
        )
        mlab.clf()
        mlab.view(0, 180, None, (0, 0, 0))
        self.surface: Surface = mlab.surf(torch.zeros(84, 84).numpy(), warp_scale="auto")
        self.animator: Animator = anim()
        mlab.orientation_axes(line_width=1, xlabel="X", ylabel="Y", zlabel="Z")
        mlab.show()

    def write_to_disk(self, agent: str, state: FieldState):
        persistence.store(
            self.config.experiment_dir,
            self.__class__.__name__,
            {
                'agent': agent,
                'state': state
            }
        )


def spawn_log_handler(init_config: ProcessInitConfig):
    """
    This method is the entrypoint for the logger process
    """
    print(f"Spawn {init_config.class_name} log handler {init_config.worker}!")
    log_handler_class: type[LogHandler] = globals()[init_config.class_name]
    assert issubclass(
        log_handler_class, LogHandler
    ), f"LogHandler [{init_config.class_name}] was not found"
    log_handler: LogHandler = log_handler_class(**init_config.params)
    log_handler.run()
    print(f"Finalized {init_config.class_name} log handler {init_config.worker}!")
