import argparse


def parse_argument():
    parse = argparse.ArgumentParser()
    parse.add_argument(
        "--model",
        type = str,
        default = "Qwen/Qwen3-0.6B",
        help = "choose the model to use"
    )
    parse.add_argument(
        "--input",
        type = str,
        default = "data/input/function_calling_tests.json",
        help = "choose the input file"
    )
    parse.add_argument(
        "--functions_definition",
        type = str,
        default = "data/input/functions_definition.json",
        help = "choose the function definition file"
    )
    parse.add_argument(
        "--output",
        type = str,
        default = "output/output.json",
        help = "choose the output file"
    )
    return parse.parse_args()
