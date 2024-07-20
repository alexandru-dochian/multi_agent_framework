import pickle
import string
import time
import random


def get_current_time() -> int:
    return int(round(time.time() * 1000))


def generate_random_string(length: int = 8) -> str:
    characters = string.ascii_letters + string.digits
    return "".join(random.choice(characters) for _ in range(length))


class DiskUtils:
    @staticmethod
    def store(path, content):
        with path.open("wb") as f:
            pickle.dump(content, f)

    @staticmethod
    def load(path):
        with path.open("rb") as file:
            return pickle.load(file)
