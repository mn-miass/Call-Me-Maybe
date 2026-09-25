import logging
import json
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
    def __init__(self, model: Small_LLM_Model, input_data, functions, output_file):
        self.model = model
        self.input_data = input_data
        self.functions = functions
        self.output_file = output_file

    def generate_prompt(self):
        logging.debug(
            f"Creating the prompt"
        )
        prompt =    "You are a function selector. Given a user request, output ONLY the name " \
                    "of the single best matching function from the list below. " \
                    "Do not explain. Do not add punctuation. Output nothing except the exact function name.\n" \
                    "If nothing matches, output: None\n" \
                    "Any description that use words replace or subtitute or anything simualar its a regex" \
                    "Functions:\n"

        for function in self.functions:
            parameters = ""
            for parameter in function.parameters:
                parameters += f"{parameter}: {function.parameters[parameter].type.value} "
            function_prompt = f"\n-   function name: {function.name}, function description: {function.description} have parameter(s) {parameters} with return type {function.returns.type.value}"
            prompt += function_prompt
        self.system_prompt = prompt
        display_building_prompt()
        logging.info(
            f"\nthe system prompt was created: \n{self.system_prompt}"
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

    def set_all_logits_to_invalid_except_valid_one(self, logits, valid_id, valid_value):
        logging.debug(
            f"Set alllogits to -inf except {valid_value}"
        )
        mask = np.full(len(logits, -np.inf))
        mask[valid_id] = 0
        logits = logits + mask
        return logits

    def check_if_max_logit_is_valid(self, logits, valid_id):
        return np.argmax(logits) == valid_id

    """
        Encode the system prompt only once (reduce the time)
        Encode each prompt only when its needed
    """
    def main_loop(self):
        result = []
        self.get_parameters_for_each_function()
        for prompt in self.input_data:
            logging.debug(
                f"Processing the prompt {prompt.prompt}"
            )
            encoded_prompt = self.encode_prompt(prompt.prompt)
            the_current_prompt = self.encoded_system_prompt + encoded_prompt
            logging.debug(
                f"Merging The text with the system_prompt :\n{the_current_prompt}"
            )
            prompt_data = {
                "prompt": prompt.prompt
            }
            name = self.get_function_name(the_current_prompt)
            parameters = self.get_parameters(name, the_current_prompt)
            prompt_data["name"] = name
            prompt_data["parameters"] = parameters
            result.append(prompt_data)
            self.result = result

    def get_function_name(self, encoded_prompt):
        prompt = 'Answer with exactly valid json from the list above in the correct format {"name": "<fn>"}, nothing else.\n' \
                 '{"name": "'
        aditional_prompt = self.encode_prompt(prompt)
        general_prompt = encoded_prompt + aditional_prompt
        output = []
        while True:
            logits = self.model.get_logits_from_input_ids(general_prompt)
            max_logit = self.get_max_value_id(logits)
            if "\"" in self.model.decode(max_logit):
                break 
            output.append(max_logit)
            general_prompt.append(max_logit)
        logging.info(
            f"Function name :{self.model.decode(output)}"
        )
        return self.model.decode(output)

    def get_parameters(self, function_name, encoded_prompt):
        if function_name == None:
            return None
        result = self.parameters[function_name]
        for element in result:
            prompt =    f'Answer with exactly one JSON value for the parameter "{element}".\n' \
                        f'Output only the value, nothing else, no quotes unless it is a string.\n' \
                        f'{element} = '
            output = []
            the_current_prompt = encoded_prompt + self.encode_prompt(prompt)
            print(self.model.decode(the_current_prompt), flush=True, end="")
            while True:
                logits = self.model.get_logits_from_input_ids(the_current_prompt)
                max_logit = self.get_max_value_id(logits)
                encoded_max_logits = self.model.decode(max_logit)
                print(encoded_max_logits, flush=True, end="")
                if "}" in encoded_max_logits or " " in encoded_max_logits  or "\n" in encoded_max_logits or "\t" in encoded_max_logits:
                    break
                if "\"" in encoded_max_logits:
                    output = "".join(output)
                    break
                output.append(max_logit)
                the_current_prompt.append(max_logit)
        return result

            
    def display_json(self):
        with open(self.output_file.name, "w") as file:
            json.dump(self.result, file, indent=2)

    def get_parameters_for_each_function(self):
        parameters = {}
        for function in self.functions:
            parameters[function.name] = {}
            for parameter in function.parameters:
                parameters[function.name][parameter] = None
        self.parameters = parameters