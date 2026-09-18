import sys
from pydantic import ValidationError

from .parse_arguments import parse_argument
from .load_file import load_file
from .parse_file import parse_input, parse_functions_definition
from .parse_prompt import ValidateData
from .display import display_loading, display_parsing

if __name__ == "__main__":
    parse = parse_argument()

    model = parse.model
    input = parse.input
    functions_definition = parse.functions_definition

    input_data = load_file(input)
    display_loading(input, input_data)

    functions_definition_data = load_file(functions_definition)
    display_loading(functions_definition, functions_definition_data)

    try:
        ValidateData(data=input_data)
        ValidateData(data=functions_definition_data)
    except ValidationError:
        sys.exit()

    input_data = parse_input(input_data)
    display_parsing(input, input_data)
    functions_definition_data = parse_functions_definition(functions_definition_data)
    display_parsing(functions_definition, functions_definition_data)

    if not input_data or not functions_definition_data:
        sys.exit()