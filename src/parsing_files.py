import json
import sys
import logging

from .argument_parser import argument_parser
from .display_info import display_checking_file



logging.basicConfig(
        filename ="log.log",
        level = logging.DEBUG,
        format = "%(asctime)s - %(levelname)s - %(message)s",
        filemode = "w"
    )

class File():
    def __init__(self, path, permission):
        self.name = path
        self.permission = permission
        self.data = None
        self.error = None
        logging.debug(
            f"Creating File Object {self.name} with permissions {self.permission}"
        )


class ParsingFiles():
    def __init__(self):
        self.valid = True
        self.get_arguments()
        self.check_files()
        self.exist_if_error()

    def get_arguments(self):
        parse = argument_parser()
        self.input = parse.input
        self.output = parse.output
        self.functions_definition = parse.functions_definition
        self.model = parse.model
        logging.debug(
            f"Getting all the arguments input={self.input} output={self.output} functions={self.functions_definition}"
        )

    def check_files(self):
        self.input_file = File(self.input, "r")
        self.output_file = File(self.output, "w")
        self.functions_definition_file = File(self.functions_definition, "r")

        self.files = [self.input_file, self.output_file, self.functions_definition_file]
        for file in self.files:
            self.check_file(file)

        for file in self.files:
            display_checking_file(file.name, file.error)

    def exist_if_error(self):
        for file in self.files:
            if file.error:
                logging.error(
                    f"{file.name}"
                )
                logging.critical(
                    "EXITING THE FILE"
                )
                sys.exit()
            logging.info(
                f"{file.name} Passed"
            )

    @staticmethod
    def check_file(file):
        try:
            with open(file.name, file.permission) as f:
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
