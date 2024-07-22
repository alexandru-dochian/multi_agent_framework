import argparse
import json
import logging
import os

from maf.app import App

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "maf/config")


def parse_arguments():
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Multi agent framework for collective intelligence research",
    )
    parser.add_argument(
        "experiment_config_file", nargs="?", default=f"{CONFIG_DIR}/default.json"
    )
    arguments = parser.parse_args()
    logging.info(f"Using config_file = [{arguments.experiment_config_file}]")

    if not arguments.experiment_config_file.endswith(".json"):
        raise Exception("experiment_config_file should be a json file")

    return arguments.experiment_config_file


def load_config(file_path: str) -> dict:
    abs_file_path = os.path.abspath(file_path)
    expected_directory_path = os.path.abspath(CONFIG_DIR)
    assert (
        os.path.commonprefix([abs_file_path, expected_directory_path])
        == expected_directory_path
    ), f"Config file should be placed under `{CONFIG_DIR}` directory!"
    with open(os.path.join(CONFIG_DIR, abs_file_path)) as fp:
        return json.load(fp)


if __name__ == "__main__":
    App(load_config(parse_arguments())).start()
