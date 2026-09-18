from pydantic import BaseModel, ConfigDict, Field
from typing import Dict



class Prompt(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str

class Parameter(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: str

class ReturnType(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: str

class FunctionDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    description: str
    parameters: Dict[str, Parameter]
    returns: ReturnType

class ValidateData(BaseModel):
    data: list[Dict] = Field(min_length=1)