from abc import ABC, abstractmethod
from enum import Enum

import redis
import pickle


class Communicator(ABC):
    class CommKey(str, Enum):
        STOP_EVENT = "STOP_EVENT"

    @abstractmethod
    def send(self, comm_key: CommKey, info: object):
        ...

    @abstractmethod
    def recv(self, comm_key: CommKey) -> object | None:
        ...

    @staticmethod
    def get_pos_key(agent_id: str) -> str:
        return f"{agent_id}_pos"

    @staticmethod
    def get_action_key(agent_id: str) -> str:
        return f"{agent_id}_action"

    @staticmethod
    def get_state_key(agent_id: str) -> str:
        return f"{agent_id}_state"


class RedisCommunicator(Communicator):
    def __init__(self):
        super()
        self.redis_instance = redis.StrictRedis(host="localhost", port=6400, db=0)

    def send(self, key: Communicator.CommKey, info: object):
        data: bytes = pickle.dumps(info)
        self.redis_instance.set(key, data)

    def recv(self, key: Communicator.CommKey) -> object | None:
        data: bytes | None = self.redis_instance.get(key)
        return None if data is None else pickle.loads(data)


class InMemoryCommunicator(Communicator):
    def __init__(self):
        super()
        self.data: dict[Communicator.CommKey, object] = dict()

    def send(self, key: Communicator.CommKey, info: object):
        self.data[key] = info

    def recv(self, key: Communicator.CommKey) -> object | None:
        return self.data[key] if key in self.data else None


def get_communicator(communicator: str) -> Communicator:
    communicator_class: type[Communicator] = globals()[communicator]
    assert issubclass(
        communicator_class, Communicator
    ), f"Communicator [{communicator}] was not found"
    return communicator_class()
