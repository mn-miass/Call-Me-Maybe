import logging
import sys
import time

from .parsing_files import ParsingFiles
from .parsing_data import Parse_data
from .display_info import display_banner, display_checking_file, display_data
from .display_info import display_model, display_model_error, display_summary
from .display_info import display_decoding_start
from .constrained_decoding import ConstrainedDecoding
from llm_sdk import Small_LLM_Model  # type: ignore


def main() -> None:
    """Run the whole pipeline: parse, load the model, decode, save.

    Steps:
        1. Check the files and validate the prompts and functions.
        2. Load the language model.
        3. Run the constrained decoding and write the output JSON.

    The program exits with status 1 if the inputs are invalid or the
    model cannot be loaded. The total running time is printed at the end.
    """
    start = time.perf_counter()
    logging.basicConfig(
        filename="./src/log.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="w"
    )

    display_banner()
    logging.info(f"\n\n{' PARSING PART ':=^70}\n\n")

    files = ParsingFiles()
    display_checking_file(
        files.input_file.name,
        files.input_file.error
    )

    display_checking_file(
        files.functions_definition_file.name,
        files.functions_definition_file.error
    )

    display_checking_file(
        files.output_file.name,
        files.output_file.error
    )

    data = Parse_data(
        files.input_file.data,
        files.functions_definition_file.data
    )

    display_data(
        files.input_file.name,
        data.errors_input
    )

    display_data(
        files.functions_definition_file.name,
        data.errors_functions
    )

    if not data.is_valid():
        logging.critical("EXITING THE PROGRAM")
        sys.exit(1)

    logging.info(f"\n\n{' Loading Model ':=^70}\n\n")
    try:
        model_start = time.perf_counter()
        model = Small_LLM_Model(model_name=files.model)
        model_end = time.perf_counter()
        display_model(files.model, model_end - model_start)
        logging.info(
            f"{files.model} Was Loaded successfully"
        )
    except Exception as e:
        display_model_error(files.model)
        logging.critical(
            f"EXITING THE PROGRAM: {e}"
        )
        display_model_error(files.model)
        sys.exit(1)

    logging.info(f"\n\n{' Constrained Decoding ':=^70}\n\n")

    display_decoding_start(len(data.input_data), len(data.functions_data))
    cd = ConstrainedDecoding(
        data.parsed_function_data,
        data.parsed_input_data,
        files.output_file.name,
        model,
    )

    cd.main_loop()
    cd.display_json()
    end = time.perf_counter()
    display_summary(len(data.input_data), len(data.functions_data), end-start,
                    files.output_file.name)


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:
        print(e)
