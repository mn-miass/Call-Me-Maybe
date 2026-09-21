from pydantic import BaseModel, Field, ConfigDict, ValidationError
from typing import Dict
from enum import Enum
import logging

logging.basicConfig(
    filename = "log.log",
    level = logging.INFO,
    format = "%(asctime)s - %(levelname)s - %(message)s"
)


class DataStructur(BaseModel):
    #model_config = ConfigDict(extra="forbid")
    data: list[Dict] = Field(min_length = 1)


class Type(Enum):
    NUMBER = "number"
    STRING = "string"
    BOOLEAN = "boolean"


class Prompt(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length = 1)


class Parameter(BaseModel):
    type: Type


class ReturnType(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Type


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Parameter]
    returns: ReturnType


class Parse_data():
    def __init__(self, input_data, functions_data):
        self.input_data = input_data
        self.functions_data = functions_data
        self.errors_input = []
        self.errors_functions = []
        self.check_input()
        self.check_functions()
        self.is_valid()

    def check_input(self):
        for element in self.input_data:
            try:
                Prompt(**element)
                logging.info(f"{element} Was Validated Successfully")
            except ValidationError as e:
                self.errors_input.append(
                    e.errors()[0]["msg"] + str(e.errors()[0]["loc"]) + e.errors()[0]["type"]
                )
                logging.error(f"{element} {self.errors_input[-1]}")

    def check_functions(self):
        for element in self.functions_data:
            try:
                FunctionDefinition(**element)
                logging.info(f"{element} Was Validated Successfully")
            except ValidationError as e:
                self.errors_functions.append(
                    e.errors()[0]["msg"] + str(e.errors()[0]["loc"]) + e.errors()[0]["type"]
                )
                logging.error(f"{element} {self.errors_functions[-1]}")

    def is_valid(self):
        self.valid = not self.errors_input and not self.errors_functions