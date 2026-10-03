import argparse


def argument_parser() -> argparse.Namespace:
    """Parse the command line arguments of the program.

    Returns:
        The parsed arguments with the attributes ``input``, ``output``,
        ``functions_definition`` and ``model``. Each one has a default
        value, so the program can run without any argument.
    """
    parse = argparse.ArgumentParser()
    parse.add_argument(
        "--input",
        type=str,
        help=(
            "Path to the input JSON file containing prompts "
            "(function_calling_tests.json). Defaults to %(default)s"
        ),
        default="data/input/function_calling_tests.json"
    )

    parse.add_argument(
        "--output",
        type=str,
        help=(
            "Path to the output JSON file where results will be written. "
            "Defaults to %(default)s"
        ),
        default="data/output/function_calls.json"
    )

    parse.add_argument(
        "--functions_definition",
        type=str,
        help=(
            "Path to the JSON file defining available functions, their "
            "parameters, and types (functions_definition.json). "
            "Defaults to %(default)s"
        ),
        default="data/input/functions_definition.json"
    )

    parse.add_argument(
        "--model",
        type=str,
        help=(
            "Hugging Face model identifier to use for generation "
            "(e.g. Qwen/Qwen3-0.6B). Optional, not required by the "
            "subject. Defaults to %(default)s"
        ),
        default="Qwen/Qwen3-0.6B"
    )

    return parse.parse_args()
