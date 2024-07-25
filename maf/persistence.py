import logging
import os.path
import pickle
from os import listdir
from os.path import isfile

from maf import utils, logger_config

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

logger: logging.Logger = logger_config.get_logger(__name__)


def store(experiment_dir: str, logger: str, content: dict):
    dir_path: str = os.path.join(
        DATA_DIR,
        experiment_dir,
        logger,
    )
    os.makedirs(dir_path, exist_ok=True)

    current_time: int = utils.get_current_time()
    random_string: str = utils.generate_random_string()
    file_name: str = f"{current_time}_{random_string}.pkl"
    file_path: str = os.path.join(dir_path, file_name)

    content.update({"current_time": current_time})
    with open(file_path, "wb") as f:
        pickle.dump(content, f)


def list_files(experiment_dir, logger: str) -> list[str]:
    dir_path: str = os.path.join(DATA_DIR, experiment_dir, logger)
    return [f for f in listdir(dir_path) if isfile(os.path.join(dir_path, f))]


def load(experiment_dir: str, logger: str, file_name: str) -> object:
    file_path: str = os.path.join(DATA_DIR, experiment_dir, logger, file_name)
    with open(file_path, "rb") as file:
        return pickle.load(file)
