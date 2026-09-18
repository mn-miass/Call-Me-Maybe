from pydantic import BaseModel, ConfigDict, Field, model_validator, ValidationError
from typing import Dict
from enum import Enum
from keyword import iskeyword


class Type(Enum):
    NUMBER = "number"
    STRING = "string"


class Prompt(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_legth = 1)

    @model_validator(mode="after")
    def check(self):
        if len(self.prompt) == 0:
            raise ValidationError("Error:")
        return self


class Parameter(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Type


class ReturnType(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Type


class FunctionDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: Dict[str, Parameter] = Field(min_length=1)
    returns: ReturnType

    @model_validator(mode="after")
    def check(self):
        if iskeyword(self.name):
            raise ValueError(f"Name Cant be a keyword {self.name}")
        if not self.name.isidentifier():
            raise ValueError(f"Name Must Be a real Identifier {self.name}")
        return self


class ValidateData(BaseModel):
    data: list[Dict] = Field(min_length=1)
