
import logging
import sys


from .parsing_files import ParsingFiles
from .parsing_data import Parse_data
from .display_info import display_data,display_model, display_model_error
from .constrained_decoding import ConstrainedDecoding
from llm_sdk import Small_LLM_Model


logging.basicConfig(
    filename = "log.log",
    level = logging.INFO,
    format = "%(asctime)s - %(levelname)s - %(message)s",
    filemode = "w"
)

logging.info(
    "\n\n=========================================PARSING PART===========================================\n\n"
)

files = ParsingFiles()
data = Parse_data(
    files.input_file.data,
    files.functions_definition_file.data
)


display_data(
    files.input_file.name,
    data.errors_input
)

display_data(
    files.input_file.name,
    data.errors_functions
)

if not data.is_valid:
    logging.critical("EXITING THE PROGRAM")
    sys.exit()


logging.info(
    "\n\n=========================================Loading Model===========================================\n\n"
)
try:
    model = Small_LLM_Model(model_name=files.model)
    display_model(files.model)
    logging.info(
        f"{files.model} Was Loaded successfully"
    )
except Exception:
    display_model_error(files.model)
    logging.critical(
        "EXITING THE PROGRAM"
    )
    sys.exit()

logging.info(
    "\n\n=========================================Constrained Decoding===========================================\n\n"
)

cd = ConstrainedDecoding(
    model,
    data.parsed_input_data,
    data.parsed_function_data
)

cd.generate_prompt()
cd.get_vocabulary()
cd.processing_system_prompt()
cd.main_loop()