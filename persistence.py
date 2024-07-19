import glob
import os.path
import pickle

import utils

DATA_DIR = "data"


def create_directory(directory: str):
    os.makedirs(directory, exist_ok=True)


def store(experiment_dir: str, logger: str, content: dict):
    dir_path: str = os.path.join(
        DATA_DIR,
        experiment_dir,
        logger,
    )
    create_directory(dir_path)

    current_time: int = utils.get_current_time()
    random_string: str = utils.generate_random_string()
    file_name: str = f"{current_time}_{random_string}.pkl"
    file_path: str = os.path.join(
        dir_path,
        file_name
    )

    content.update({'current_time': current_time})
    with open(file_path, "wb") as f:
        pickle.dump(content, f)


def list_files(experiment_dir, logger: str):
    dir_path: str = os.path.join(
        DATA_DIR,
        experiment_dir,
        logger
    )
    return glob.glob(dir_path)


def load(experiment_dir: str, logger: str, file_name: str) -> object:
    file_path: str = os.path.join(
        DATA_DIR,
        experiment_dir,
        logger,
        file_name
    )
    with open(file_path, 'rb') as file:
        return pickle.load(file)
