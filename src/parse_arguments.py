import argparse


def parse_argument():
    parse = argparse.ArgumentParser()
    parse.add_argument(
        "--model",
        type = str,
        default = "Qwen/Qwen3-0.6B"
    )
    parse.add_argument(
        "--input",
        type = str,
        default = "data/input/function_calling_tests.json"
    )
    parse.add_argument(
        "--functions_definition",
        type = str,
        default = "data/input/functions_definition.json"
    )
    return parse.parse_args()
