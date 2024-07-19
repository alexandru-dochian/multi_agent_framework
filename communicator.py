import os
import pickle

import redis

from core import Communicator, Config, ObjectInitConfig, Action, State


class RedisCommunicatorConfig(Config):
    host: str = "localhost"
    port: int = 6400
    db: int = 0


class RedisCommunicator(Communicator):
    config: RedisCommunicatorConfig

    STOP_EVENT = "stop-event"
    REGISTERED_AGENTS = "registered-agents"

    def __init__(self, config: dict | None = None):
        super().__init__(
            RedisCommunicatorConfig(**config) if config else RedisCommunicatorConfig()
        )
        self.redis_instance = redis.StrictRedis(
            host=self.config.host, port=self.config.port, db=self.config.db
        )

    def activate(self):
        # clear cached registered agents
        self.redis_instance.unlink(self.REGISTERED_AGENTS)
        self.send(self.STOP_EVENT, False)

    def stop(self, signal, *kwargs):
        print(
            f"Signal [{signal}] received on [{os.getpid()}] of parent [{os.getppid()}]. Stopping communication!"
        )
        self.send(self.STOP_EVENT, True)

    def is_active(self):
        return self.recv(self.STOP_EVENT) is False

    def send(self, key: str, info: object):
        data: bytes = pickle.dumps(info)
        self.redis_instance.set(key, data)

    def recv(self, key: str) -> object | None:
        data: bytes | None = self.redis_instance.get(key)
        return None if data is None else pickle.loads(data)

    #
    def register_agent(self, agent_id: str):
        self.redis_instance.sadd(self.REGISTERED_AGENTS, agent_id)

    def deregister_agent(self, agent_id: str):
        self.redis_instance.srem(self.REGISTERED_AGENTS, agent_id)

    def registered_agents(self) -> list[str]:
        return [
            member.decode("utf-8")
            for member in self.redis_instance.smembers(self.REGISTERED_AGENTS)
        ]

    def broadcast_state(self, agent_id: str, state: State):
        state_key: str = self._get_state_key(agent_id)
        self.send(state_key, state)

    def broadcast_action(self, agent_id: str, action: Action):
        action_key: str = self._get_action_key(agent_id)
        self.send(action_key, action)

    def get_state(self, agent_id: str) -> State:
        return self.recv(self._get_state_key(agent_id))

    def get_action(self, agent_id: str) -> Action:
        return self.recv(self._get_action_key(agent_id))

    @staticmethod
    def _get_action_key(agent_id: str) -> str:
        return f"{agent_id}_action"

    @staticmethod
    def _get_state_key(agent_id: str) -> str:
        return f"{agent_id}_state"


def get_communicator(init_config: dict | ObjectInitConfig) -> Communicator:
    if isinstance(init_config, dict):
        init_config: ObjectInitConfig = ObjectInitConfig(**init_config)

    communicator_class: type[Communicator] = globals()[init_config.class_name]
    assert issubclass(
        communicator_class, Communicator
    ), f"Communicator [{init_config.class_name}] was not found"
    return communicator_class(**init_config.params)
