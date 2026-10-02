import json
import logging
import sys
from typing import Any, List, Union

from .argument_parser import argument_parser
from .display_info import display_checking_file


logging.basicConfig(
    filename="./src/log.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w"
)


class File():
    """A file the program needs, with its access mode and its content."""

    def __init__(self, path: str, permission: str) -> None:
        """Store the path and the mode; nothing is opened yet.

        Args:
            path: Path of the file.
            permission: Mode used to open it (``"r"`` or ``"w"``).
        """
        self.name = path
        self.permission = permission
        self.data: Any = None
        self.error: Union[Exception, bool, None] = None
        logging.debug(
            f"Creating File Object {self.name} "
            f"with permissions {self.permission}"
        )


class ParsingFiles():
    """Read the command line arguments and check all the files.

    The program exits with an error code as soon as one file cannot be
    opened or, for the input files, cannot be parsed as JSON.
    """

    def __init__(self) -> None:
        """Read the arguments, check the files and exit on any error."""
        self.valid = True
        self.get_arguments()
        self.check_files()
        self.exist_if_error()

    def get_arguments(self) -> None:
        """Parse the command line and store the values on ``self``."""
        parse = argument_parser()
        self.input = parse.input
        self.output = parse.output
        self.functions_definition = parse.functions_definition
        self.model = parse.model
        logging.debug(
            f"Getting all the arguments input={self.input} "
            f"output={self.output} functions={self.functions_definition}"
        )

    def check_files(self) -> None:
        """Check the input, output and function definition files.

        Input files are loaded as JSON. The output file is opened in
        write mode to make sure it can be created.
        """
        self.input_file = File(self.input, "r")
        self.output_file = File(self.output, "w")
        self.functions_definition_file = File(self.functions_definition, "r")

        self.files: List[File] = [
            self.input_file,
            self.output_file,
            self.functions_definition_file,
        ]
        for file in self.files:
            self.check_file(file)

        for file in self.files:
            display_checking_file(file.name, file.error)

    def exist_if_error(self) -> None:
        """Exit the program with status 1 if any file has an error."""
        for file in self.files:
            if file.error:
                logging.error(
                    f"{file.name}: {file.error}"
                )
                logging.critical(
                    "EXITING THE FILE"
                )
                sys.exit(1)
            logging.info(
                f"{file.name} Passed"
            )

    @staticmethod
    def check_file(file: File) -> None:
        """Open one file and load its JSON content if it is read-only.

        Any exception is stored in ``file.error`` instead of being raised.

        Args:
            file: The file to check. ``file.data`` and ``file.error`` are
                updated.
        """
        try:
            with open(file.name, file.permission, encoding="utf-8") as f:
                if file.permission == "r":
                    file.data = json.load(f)
                    file.error = False
                logging.info(
                    f"File {file.name} was Loaded Successfully"
                )
        except Exception as e:
            logging.error(
                e
            )
            file.error = e
