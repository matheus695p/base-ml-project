import warnings
from argparse import ArgumentParser

warnings.filterwarnings("ignore")


def get_command_line_arguments():
    parser = ArgumentParser()
    parser.add_argument("--env", default="base", type=str)
    return parser.parse_args()
