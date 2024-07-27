import argparse
import json
import logging
import os

from maf import logger_config
from maf.app import App

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "maf/config")

logger: logging.Logger = logger_config.get_logger(__name__)


def read_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


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

    # Adding arguments for displaying legal texts
    parser.add_argument(
        "show",
        choices=["l", "n"],
        nargs="?",
        help="Display legal texts: 'l' for LICENSE, 'n' for NOTICE",
    )

    parser.add_argument(
        "config",
        nargs="?",
        default=f"{CONFIG_DIR}/default.json",
        help="Path to any maf/config/*.json configuration file",
    )
    arguments = parser.parse_args()

    if arguments.show == "l":
        copying_text = read_text_file("LICENSE")
        print(copying_text)
        exit(0)
    elif arguments.show == "n":
        notice_text = read_text_file("NOTICE")
        print(notice_text)
        exit(0)

    if not arguments.config.endswith(".json"):
        raise Exception("config should be a json file")

    return arguments.config


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
    config: str = parse_arguments()
    print_banner(config)
    App(load_config(parse_arguments())).start()
