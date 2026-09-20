from pydantic import BaseModel, Field, ConfigDict
from typing import Dict
from enum import Enum


class DataStructur(BaseModel):
    #model_config = ConfigDict(extra="forbid")
    data: list[Dict] = Field(min_length = 1)

class Type(Enum):
    "NUMBER" = "number"
    "STRING" = "string"
    "BOOLEAN" = "boolean"

class Prompt(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length = 1)


class Parameter(BaseModel):
    type: Type = Field(min_length = 1)


class ReturnType(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Type = Field(min_length = 1)


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Parameter]
    returns: ReturnType