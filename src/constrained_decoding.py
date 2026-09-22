import logging
import json

from llm_sdk import Small_LLM_Model
from .display_info import display_building_prompt

logging.basicConfig(
    filename = "log.log",
    level = logging.INFO,
    format = "%(asctime)s - %(levelname)s - %(message)s",
    filemode = "w"
)


class ConstrainedDeoding():
    def __init__(self, model: Small_LLM_Model, input_data, functions):
        self.model = model
        self.input_data = input_data
        self.functions = functions

    def generate_prompt(self):
        logging.debug(
            f"Creating the prompt"
        )
        prompt = "Use only a matching function from the list below" \
        "if no function matches the user intent (even if types match), set name \"None\""
        "never use unrelated function for a different task."
        "Available Functions: "

        for function in self.functions:
            parameters = ""
            for parameter in function.parameters:
                parameters += f"{parameter}: {function.parameters[parameter].type.value} "
            function_prompt = f"\n-   {function.name}: {function.description} have parameter(s) {parameters} with return type {function.returns.type.value}"
            prompt += function_prompt
        prompt += '\nOutput only vlid json: {"name":"<fn>, "args": {<args>}}'
        self.prompt = prompt
        display_building_prompt()
        logging.info(
            f"\nthe prompt was created: \n{self.prompt}"
        )

    def get_vocabulary(self):
        vocab_path = self.model.get_path_to_vocab_file()
        with open(vocab_path, "r") as file:
            self.vocab_data = json.load(file)

    def processing_prompts(self):
        encoded_prompts = []
        logging.debug(
            f"Encoding all prompts"
        )
        for prompt in self.input_data:
            encoded_prompts.append(
                self.encode_prompt("User: prompt: "+ prompt.prompt)
            )

    def encode_prompt(self, prompt):
        logging.debug(
            f"encoding the text {prompt}"
        )
        encoded = self.model.encode(prompt)
        logging.info(
            f"text was encoded into {encoded}"
        )
        return encoded

