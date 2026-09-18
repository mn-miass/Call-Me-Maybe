from pydantic import ValidationError
from .parse_prompt import Prompt, FunctionDefinition



def parse_input(input_data):
    lst = []
    for data in input_data:
        try:
            lst.append(Prompt(**data))
        except ValidationError as e:
            return e.errors()[0]["msg"] + str(e.errors()[0]["loc"]) + e.errors()[0]["type"]
    return lst


def parse_functions_definition(functions_definition_data):
    lst = []
    for data in functions_definition_data:
        try:
            lst.append(FunctionDefinition(**data))
        except ValidationError as e:
            return e.errors()[0]["msg"] + str(e.errors()[0]["loc"]) + e.errors()[0]["type"]
    return lst

