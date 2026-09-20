# import sys
# from pydantic import ValidationError

# from .parse_arguments import parse_argument
# from .load_file import load_file
# from .parse_file import parse_input, parse_functions_definition
# from .parse_data import ValidateData
# from .display import display_loading, display_parsing, display_model, display_model_error
# from .build_prompt import build_prompt
# from .constrained_decoding import ConstrainedDecoding


# from llm_sdk import Small_LLM_Model


# import json
# import time


# if __name__ == "__main__":
#     start = time.perf_counter()

#     parse = parse_argument()

#     model = parse.model
#     input = parse.input
#     functions_definition = parse.functions_definition
#     output = parse.output

#     input_data = load_file(input)
#     display_loading(input, input_data)

#     functions_definition_data = load_file(functions_definition)
#     display_loading(functions_definition, functions_definition_data)

#     try:
#         ValidateData(data=input_data)
#         ValidateData(data=functions_definition_data)
#     except ValidationError as e:
#         print(e.errors()[0]["msg"] + str(e.errors()[0]["loc"]) + e.errors()[0]["type"])
#         sys.exit()

#     input_data = parse_input(input_data)
#     display_parsing(input, input_data)
#     functions_definition_data = parse_functions_definition(functions_definition_data)
#     display_parsing(functions_definition, functions_definition_data)

#     if not input_data or not functions_definition_data:
#         sys.exit()

#     prompt = build_prompt(functions_definition_data)

#     display_model(model)
#     try:
#         model = Small_LLM_Model(model_name=model)
#     except Exception:
#         display_model_error(model)
#         sys.exit()

#     constraineddecoding = ConstrainedDecoding(
#         model,
#         input_data,
#         prompt
#     )

#     end = time.perf_counter()
#     constraineddecoding.constrained_decoding()
#     print(end - start)




from .parsing_files import ParsingFiles
from .parsing_data import Prompt, FunctionDefinition, DataStructur


files = ParsingFiles()
