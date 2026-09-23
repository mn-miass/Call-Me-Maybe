import logging
import json
import torch
import numpy as np


from llm_sdk import Small_LLM_Model
from .display_info import display_building_prompt


logging.basicConfig(
    filename = "log.log",
    level = logging.INFO,
    format = "%(asctime)s - %(levelname)s - %(message)s",
    filemode = "w"
)


class ConstrainedDecoding():
    def __init__(self, model: Small_LLM_Model, input_data, functions):
        self.model = model
        self.input_data = input_data
        self.functions = functions

    def generate_prompt(self):
        logging.debug(
            f"Creating the prompt"
        )
        prompt = "You will be given a prompt and match it with the correct function" \
        "Use only a matching function from the list below" \
        "if no function matches the user intent (even if types match), set name \"None\""
        "never use unrelated function for a different task."
        "Available Functions: "
        "the answer should be as this format:"

        for function in self.functions:
            parameters = ""
            for parameter in function.parameters:
                parameters += f"{parameter}: {function.parameters[parameter].type.value} "
            function_prompt = f"\n-   {function.name}: {function.description} have parameter(s) {parameters} with return type {function.returns.type.value}"
            prompt += function_prompt
        prompt += '\nOutput only and ONLY The valid json: {"name":"<fn>, "args": {<args>}} Nothing else'
        self.system_prompt = prompt
        display_building_prompt()
        logging.info(
            f"\nthe prompt was created: \n{self.system_prompt}"
        )

    def get_vocabulary(self):
        vocab_path = self.model.get_path_to_vocab_file()
        with open(vocab_path, "r") as file:
            self.vocab_data = json.load(file)

    def processing_system_prompt(self):
        self.encoded_system_prompt = self.encode_prompt(self.system_prompt)


    def encode_prompt(self, prompt):
        logging.debug(
            f"encoding the text :\n{prompt}"
        )
        encoded = self.model.encode(prompt).tolist()[0]
        logging.info(
            f"text was encoded into :\n{encoded}"
        )
        return encoded

    def get_max_value_id(self, logits):
        logging.debug(
            "Getting Max Logit"
        )
        return np.argmax(logits)

    """
        Encode the system prompt only once (reduce the time)
        Encode each prompt only when its needed
    """
    def main_loop(self):
        for prompt in self.input_data:
            logging.debug(
                f"Processing the prompt {prompt.prompt}"
            )
            encoded_prompt = self.encode_prompt(prompt.prompt + "[ {")
            the_current_prompt = self.encoded_system_prompt + self.encode_prompt("[ {")
            logging.debug(
                f"Merging The text with the system_prompt :\n{the_current_prompt}"
            )
            for _ in range(50):
                logits = self.model.get_logits_from_input_ids(the_current_prompt)
                logging.debug(
                    f"Getting logits"
                )
                picked_logit = self.get_max_value_id(logits)
                the_current_prompt.append(picked_logit)
                picked = self.model.decode(picked_logit)
                print(picked, end = "", flush=True)
                if "]" in picked:
                    print()
                    break
                


                

            

