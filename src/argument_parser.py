import argparse


def argument_parser():
    parse = argparse.ArgumentParser()
    parse.add_argument(
        "--input",
        type = str,
        help = "Path to the input JSON file containing prompts (function_calling_tests.json). Defaults to data/input/function_calling_tests.json",
        default = "data/input/function_calling_tests.json"
    )

    parse.add_argument(
        "--output",
        type = str,
        help = "Path to the output JSON file where results will be written. Defaults to data/output/output.json",
        default = "output.json"
    )

    parse.add_argument(
        "--functions_definition",
        type = str,
        help = "Path to the JSON file defining available functions, their parameters, and types (functions_definition.json).",
        default = "data/input/functions_definition.json"
    )

    parse.add_argument(
        "--model",
        type = str,
        help = "Hugging Face model identifier to use for generation (e.g. Qwen/Qwen3-0.6B). Optional, not required by the subject.",
        default = "Qwen/Qwen3-0.6B"
    )

    return parse.parse_args()
