import argparse
import json
import logging
import os

from maf import logger_config
from maf.app import App

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "maf/config")

logger: logging.Logger = logger_config.get_logger(__name__)


def print_banner(experiment: str):
    text_in_banner: str = f" experiment = [{experiment}] "
    custom_line_in_banner: str = (
        "=" * 8 + text_in_banner + "=" * (159 - len(text_in_banner))
    )
    logger.info(
        f"""\n
    =======================================================================================================================================================================
    ███╗   ███╗██╗   ██╗██╗  ████████╗██╗     █████╗  ██████╗ ███████╗███╗   ██╗████████╗    ███████╗██████╗  █████╗ ███╗   ███╗███████╗██╗    ██╗ ██████╗ ██████╗ ██╗  ██╗
    ████╗ ████║██║   ██║██║  ╚══██╔══╝██║    ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝    ██╔════╝██╔══██╗██╔══██╗████╗ ████║██╔════╝██║    ██║██╔═══██╗██╔══██╗██║ ██╔╝
    ██╔████╔██║██║   ██║██║     ██║   ██║    ███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║       █████╗  ██████╔╝███████║██╔████╔██║█████╗  ██║ █╗ ██║██║   ██║██████╔╝█████╔╝ 
    ██║╚██╔╝██║██║   ██║██║     ██║   ██║    ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║       ██╔══╝  ██╔══██╗██╔══██║██║╚██╔╝██║██╔══╝  ██║███╗██║██║   ██║██╔══██╗██╔═██╗ 
    ██║ ╚═╝ ██║╚██████╔╝███████╗██║   ██║    ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║       ██║     ██║  ██║██║  ██║██║ ╚═╝ ██║███████╗╚███╔███╔╝╚██████╔╝██║  ██║██║  ██╗
    ╚═╝     ╚═╝ ╚═════╝ ╚══════╝╚═╝   ╚═╝    ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝       ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝ ╚══╝╚══╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝
    =======================================================================================================================================================================
    {custom_line_in_banner}
    =======================================================================================================================================================================
    """
    )


def parse_arguments():
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Multi agent framework for collective intelligence research",
    )
    parser.add_argument(
        "experiment_config_file", nargs="?", default=f"{CONFIG_DIR}/default.json"
    )
    arguments = parser.parse_args()

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
    config_file: str = parse_arguments()
    print_banner(config_file)
    App(load_config(parse_arguments())).start()
