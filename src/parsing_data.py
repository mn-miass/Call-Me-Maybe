import keyword
import logging
from enum import Enum
from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic import model_validator

logging.basicConfig(
    filename="./src/log.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w"
)


class DataStructur(BaseModel):
    """A non-empty list of raw JSON objects."""

    data: List[Dict[str, Any]] = Field(min_length=1)


class Type(Enum):
    """Value types allowed for function parameters and return values."""

    NUMBER = "number"
    STRING = "string"
    BOOLEAN = "boolean"
    INTEGER = "integer"


class Prompt(BaseModel):
    """One user request. Extra keys are rejected."""

    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1)


class Parameter(BaseModel):
    """The type of one function parameter."""

    type: Type


class ReturnType(BaseModel):
    """The return type of a function. Extra keys are rejected."""

    model_config = ConfigDict(extra="forbid")
    type: Type


class FunctionDefinition(BaseModel):
    """A function the model can choose: name, description and types."""

    name: str
    description: str
    parameters: Dict[str, Parameter]
    returns: ReturnType

    @model_validator(mode="after")
    def check(self) -> "FunctionDefinition":
        """Check that the name is a valid, non-keyword identifier.

        Returns:
            The validated function definition.

        Raises:
            ValueError: If the name is a Python keyword or not an
                identifier.
        """
        if keyword.iskeyword(self.name):
            raise ValueError(
                f"{self.name} cant be a keyword"
            )
        if not self.name.isidentifier():
            raise ValueError(
                f"{self.name} Cant be function name"
            )
        return self


class Parse_data():
    """Validate the prompts and the function definitions.

    Valid items are stored as pydantic objects. For invalid items an
    error message is recorded and logged, and parsing continues.
    """

    def __init__(
        self, input_data: List[Any], functions_data: List[Any]
    ) -> None:
        """Validate both inputs right away.

        Args:
            input_data: Raw prompt objects loaded from the input JSON.
            functions_data: Raw function objects loaded from the
                function definitions JSON.
        """
        self.input_data = input_data
        self.functions_data = functions_data
        self.errors_input: List[str] = []
        self.errors_functions: List[str] = []
        self.parsed_input_data: List[Prompt] = []
        self.parsed_function_data: List[FunctionDefinition] = []
        self.check_input()
        self.check_functions()
        self.is_valid()

    @staticmethod
    def _format_error(error: ValidationError) -> str:
        """Return a readable message for the first validation error."""
        first = error.errors()[0]
        return f"{first['msg']} {first['loc']} ({first['type']})"

    def check_input(self) -> None:
        """Validate every prompt and fill ``parsed_input_data``."""
        for element in self.input_data:
            try:
                data = Prompt(**element)
                self.parsed_input_data.append(data)
                logging.info(f"{element} Was Validated Successfully")
            except ValidationError as e:
                self.errors_input.append(self._format_error(e))
                logging.error(f"{element} {self.errors_input[-1]}")
            except TypeError:
                self.errors_input.append(
                    f"Expected a JSON object, got {type(element).__name__}"
                )
                logging.error(f"{element} {self.errors_input[-1]}")

    def check_functions(self) -> None:
        """Validate every function and fill ``parsed_function_data``."""
        for element in self.functions_data:
            try:
                data = FunctionDefinition(**element)
                self.parsed_function_data.append(data)
                logging.info(f"{element} Was Validated Successfully")
            except ValidationError as e:
                self.errors_functions.append(self._format_error(e))
                logging.error(f"{element} {self.errors_functions[-1]}")
            except TypeError:
                self.errors_functions.append(
                    f"Expected a JSON object, got {type(element).__name__}"
                )
                logging.error(f"{element} {self.errors_functions[-1]}")

    def is_valid(self) -> bool:
        """Tell whether both inputs were fully valid.

        Returns:
            ``True`` if no prompt and no function had an error.
        """
        self.valid = not self.errors_input and not self.errors_functions
        return self.valid
