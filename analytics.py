import torch

from core import Communicator, get_communicator


def input_space_viz(agent_id: str, communicator: str):
    from mayavi import mlab
    from mayavi.modules.surface import Surface
    from mayavi.tools.sources import MArray2DSource

    print(f"Starting visualization for [{agent_id}]")
    communicator: Communicator = get_communicator(communicator)

    mlab.figure(f"Input space of [{agent_id}]")
    fig = mlab.gcf()

    # tensor = communicator.recv(get_state_key(agent_id))
    # if tensor is None:
    tensor = torch.zeros(84, 84)
    m: Surface = mlab.surf(tensor, warp_scale="auto")

    @mlab.animate(delay=500)
    def anim():
        while not (communicator.recv(Communicator.CommKey.STOP_EVENT) is True):
            surface_source: MArray2DSource = m.mlab_source
            new_tensor = torch.randn(84, 84).numpy()
            surface_source.reset(scalars=new_tensor)
            fig.scene.reset_zoom()
            yield

        mlab.clf()
        mlab.close(all=True)

    anim()
    mlab.show()
    print(f"Finished Input space Viz for {agent_id}")


def central_viz(agents: list[str], communicator: str):
    from mayavi import mlab
    import numpy as np

    print(f"Starting master visualization for agents [{agents}]")

    communicator: Communicator = get_communicator(communicator)

    number_of_agents: int = len(agents)

    mlab.figure(f"Central visualization")
    x, y, z, value = np.random.random((4, number_of_agents))
    points = mlab.points3d(x, y, z, value)

    @mlab.animate
    def anim():
        while not (communicator.recv(Communicator.CommKey.STOP_EVENT) is True):
            x_new = x
            y_new = y
            z_new = z

            x_new += np.random.normal(0, 0.1, size=x_new.shape)
            y_new += np.random.normal(0, 0.1, size=y_new.shape)
            z_new += np.random.normal(0, 0.1, size=z_new.shape)

            points.mlab_source.reset(x=x_new, y=y_new, z=z_new)
            yield

        mlab.clf()
        mlab.close(all=True)

    anim()
    mlab.show()

    print("Finished Master Viz")
