import logging
import json
import string
import numpy as np


from llm_sdk import Small_LLM_Model
from .display_info import display_building_prompt


logging.basicConfig(
    filename = "log.log",
    level = logging.INFO,
    format = "%(asctime)s - %(levelname)s - %(message)s",
    filemode = "w"
)


# class ConstrainedDecoding():
#     def __init__(self, model: Small_LLM_Model, input_data, functions, output_file):
#         logging.info(
#             f"Initializing ConstrainedDecoding | functions_count={len(functions)} "
#             f"input_count={len(input_data)} output_file={getattr(output_file, 'name', output_file)}"
#         )
#         self.model = model
#         self.input_data = input_data
#         self.functions = functions
#         self.output_file = output_file
#         logging.debug("ConstrainedDecoding instance attributes set")

#     def generate_prompt(self):
#         logging.info("ENTER generate_prompt")
#         logging.debug("Creating the prompt")
#         prompt =    "You are a function selector. Given a user request, output ONLY the name " \
#                     "of the single best matching function from the list below. " \
#                     "Do not explain. Do not add punctuation. Output nothing except the exact function name.\n" \
#                     "Read each function's description carefully and match it to the user's request by meaning.\n" \
#                     "If nothing matches, output: None\n" \
#                     "Functions:\n"

#         logging.debug(f"Base prompt initialized, length={len(prompt)}")
#         for function in self.functions:
#             logging.debug(f"Appending function to prompt: {function.name}")
#             parameters = ", ".join(
#                 f"{name} ({function.parameters[name].type.value})"
#                 for name in function.parameters
#             )
#             function_prompt = (
#                 f"\n- name: {function.name}\n"
#                 f"  description: {function.description}\n"
#                 f"  parameters: {parameters}\n"
#                 f"  returns: {function.returns.type.value}\n"
#             )
#             prompt += function_prompt
#             logging.debug(f"  parameters added: {parameters}")
#         self.system_prompt = prompt
#         display_building_prompt()
#         logging.info(f"\nthe system prompt was created: \n{self.system_prompt}")
#         logging.info(f"EXIT generate_prompt | final system_prompt length={len(self.system_prompt)}")

#     def get_vocabulary(self):
#         logging.info("ENTER get_vocabulary")
#         vocab_path = self.model.get_path_to_vocab_file()
#         logging.debug(f"Vocabulary path resolved: {vocab_path}")
#         with open(vocab_path, "r") as file:
#             self.vocab_data = json.load(file)
#         with open("vocab.json", "w") as file:
#             json.dump(self.vocab_data, file, indent=2)
#         logging.info(f"Vocabulary loaded | entries={len(self.vocab_data) if hasattr(self.vocab_data, '__len__') else 'unknown'}")
#         logging.info("EXIT get_vocabulary")

#     def processing_system_prompt(self):
#         logging.info("ENTER processing_system_prompt")
#         self.encoded_system_prompt = self.encode_prompt(self.system_prompt)
#         logging.info(f"System prompt encoded | token_count={len(self.encoded_system_prompt)}")
#         logging.info("EXIT processing_system_prompt")


#     def encode_prompt(self, prompt):
#         logging.debug(
#             f"encoding the text :\n{prompt}"
#         )
#         encoded = self.model.encode(prompt).tolist()[0]
#         logging.info(
#             f"text was encoded into :\n{encoded}"
#         )
#         return encoded

#     def get_max_value_id(self, logits):
#         logging.debug(
#             "Getting Max Logit"
#         )
#         result = np.argmax(logits)
#         logging.debug(f"Max logit id found: {result}")
#         return result

#     def set_all_logits_to_invalid_except_valid(self, logits, valid_id):
#         mask = np.full(len(logits), -np.inf)
#         mask[list(valid_id)] = 0
#         logits = logits + mask
#         return logits

#     def check_if_max_logit_is_valid(self, logits, valid_id):
#         result = np.argmax(logits) == valid_id
#         logging.debug(f"check_if_max_logit_is_valid | valid_id={valid_id} result={result}")
#         return result

#     """
#         Encode the system prompt only once (reduce the time)
#         Encode each prompt only when its needed
#     """
#     def main_loop(self):
#         logging.info("ENTER main_loop")
#         result = []
#         self.get_parameters_for_each_function()
#         logging.info(f"Beginning iteration over input_data | total_prompts={len(self.input_data)}")
#         self.get_parameters_types()
#         self.encode_types()
#         for idx, prompt in enumerate(self.input_data):
#             prompt_data = {
#                 "prompt": prompt.prompt
#             }
#             logging.info(f"--- Processing prompt #{idx} ---")
#             logging.debug(
#                 f"Processing the prompt {prompt.prompt}"
#             )
#             encoded_prompt = self.encode_prompt(prompt.prompt)
#             the_current_prompt = self.encoded_system_prompt + encoded_prompt
#             logging.debug(
#                 f"Merging The text with the system_prompt :\n{the_current_prompt}"
#             )
#             logging.debug(f"Combined prompt token_count={len(the_current_prompt)}")
#             logging.debug(
#                 f"Getting the function name for {prompt.prompt}"
#             )
#             name = self.get_function_name(the_current_prompt)
#             logging.info(f"Prompt #{idx} resolved function name: {name}")
#             logging.debug(
#                 f"Getting the function parameters for {prompt.prompt}"
#             )
#             parameters = self.get_parameters(name, the_current_prompt)
#             logging.info(f"Prompt #{idx} resolved parameters: {parameters}")
#             prompt_data["name"] = name
#             prompt_data["parameters"] = parameters
#             result.append(prompt_data)
#             logging.debug(f"New result is {result}")
#         logging.debug(f"Prompt #{idx} appended to result | result_size={len(result)}")
#         self.result = result
#         logging.info(f"EXIT main_loop | total_results={len(self.result)}")

#     def get_function_name(self, encoded_prompt):
#         logging.info("ENTER get_function_name")
#         prompt = 'Answer with exactly valid json from the list above in the correct format {"name": "<fn>"}, nothing else.\n' \
#                  '{"name": "'
#         aditional_prompt = self.encode_prompt(prompt)
#         general_prompt = encoded_prompt + aditional_prompt
#         logging.debug(f"general_prompt token_count={len(general_prompt)}")
#         output = []
#         iteration = 0
#         while True:
#             iteration += 1
#             logging.debug(f"get_function_name loop iteration={iteration}")
#             logits = self.model.get_logits_from_input_ids(general_prompt)
#             max_logit = self.get_max_value_id(logits)
#             decoded_token = self.model.decode(max_logit)
#             logging.debug(f"Decoded token at iteration {iteration}: {decoded_token}")
#             if "\"" in decoded_token:
#                 logging.debug(f"Closing quote detected at iteration {iteration}, breaking loop")
#                 break 
#             output.append(max_logit)
#             general_prompt.append(max_logit)
#         logging.info(
#             f"Function name :{self.model.decode(output)}"
#         )
#         logging.info(f"EXIT get_function_name | iterations={iteration}")
#         return self.model.decode(output)

#     def get_parameters(self, function_name, encoded_prompt):
#         logging.info(f"ENTER get_parameters | function_name={function_name}")
#         if function_name == None:
#             logging.info("function_name is None, EXIT get_parameters early with None")
#             return None
#         result = dict(self.parameters[function_name])
#         logging.debug(f"Parameter slots for {function_name}: {list(result.keys())}")
#         for element in result:
#             logging.info(f"-- Resolving parameter: {element} --")
#             prompt =    f'Answer with exactly one JSON value for the parameter "{element}".\n' \
#                         f'Output only the value, nothing else, no quotes unless it is a string.\n' \
#                         f'{element} = '
#             output = []
#             the_current_prompt = encoded_prompt + self.encode_prompt(prompt)
#             count = 0
#             iteration = 0
#             type_element = self.types[function_name][element]
#             while True:
#                 iteration += 1
#                 logging.debug(f"get_parameters loop | parameter={element} iteration={iteration}")
#                 logits = self.model.get_logits_from_input_ids(the_current_prompt)
#                 if type_element == "NUMBER":
#                     logits = self.set_all_logits_to_invalid_except_valid(logits, self.numbers_encoded + self.close_encoded)
#                 elif type_element == "STRING":
#                     logits = self.set_all_logits_to_invalid_except_valid(logits, self.strings_encoded + self.close_encoded)
#                 elif type_element == "BOOLEAN":
#                     logits = self.set_all_logits_to_invalid_except_valid(logits, self.boolean_encoded + self.close_encoded)
#                 max_logit = self.get_max_value_id(logits)
#                 encoded_max_logits = self.model.decode(max_logit)
#                 logging.debug(
#                     f"Getting the parameter {encoded_max_logits}"
#                 )
#                 if "}" in encoded_max_logits or " " in encoded_max_logits  or "\n" in encoded_max_logits or "\t" in encoded_max_logits:
#                     logging.debug(
#                         f"Logit was {encoded_max_logits} End of parameter {result}"
#                     )
#                     result[element] = "".join(output)
#                     if type_element == "NUMBER":
#                         result[element] = float(result[element])
#                     logging.info(f"Parameter {element} resolved (list form) via terminator token, iterations={iteration}")
#                     break
#                 if "\"" in encoded_max_logits:
#                     if count:
#                         result[element] = "".join(output)
#                         logging.debug(
#                             f"Logit was {encoded_max_logits} End of parameter {result}"
#                         )
#                         logging.info(f"Parameter {element} resolved (string form) via closing quote, iterations={iteration}")
#                         break
#                     else:
#                         count += 1
#                         logging.debug(f"Opening quote detected for parameter {element}, count={count}")
#                 output.append(encoded_max_logits)
#                 the_current_prompt.append(max_logit)
#             output = []
#         logging.info(f"EXIT get_parameters | function_name={function_name} result={result}")
#         return result


#     def display_json(self):
#         logging.info(f"ENTER display_json | writing to {self.output_file.name}")
#         with open(self.output_file.name, "w") as file:
#             json.dump(self.result, file, indent=2)
#         logging.info("EXIT display_json | write complete")

#     def get_parameters_for_each_function(self):
#         logging.info("ENTER get_parameters_for_each_function")
#         parameters = {}
#         for function in self.functions:
#             logging.debug(f"Initializing parameter slots for function: {function.name}")
#             parameters[function.name] = {}
#             for parameter in function.parameters:
#                 parameters[function.name][parameter] = None
#                 logging.debug(f"  {function.name}.{parameter} = None")
#         self.parameters = parameters
#         logging.info(f"EXIT get_parameters_for_each_function | functions_initialized={len(parameters)}")

#     def get_parameters_types(self):
#         types = {}
#         for function in self.functions:
#             types[function.name] = {}
#             for parameter in function.parameters:
#                 types[function.name][parameter] = function.parameters[parameter].type.name
#         self.types = types

#     def encode_types(self):
#         integers = string.digits + "-+"
#         numbers = string.digits + "-+."
#         numbers = {char for char in numbers}
#         strings = string.printable
#         strings = {char for char in strings}
#         close = string.whitespace + "}\","
#         close = {char for char in close}
#         boolean_true = "True"
#         boolean_false = "False"

#         self.numbers_encoded = self.encode_prompt(list(numbers))
#         self.strings_encoded = self.encode_prompt(list(strings))
#         self.boolean_encoded = self.encode_prompt(list(boolean_true)) + self.encode_prompt(list(boolean_false))
#         self.close_encoded = self.encode_prompt(list(close))


class ConstrainedDecoding():
    def __init__(self, functions, prompts, output_path, model: Small_LLM_Model):
        self.functions = functions
        self.prompts = prompts
        self.output_path = output_path
        self.model = model
        self.output = []
        self._making_decoder_data()
        self._encode_needed_characters()
        self._build_masks()
        self._generate_system_prompt()
        self._encode_system_prompt()
        self._encode_function_names()
        self._encode_get_parameter_prompt()
        self._main_loop()

    def _my_encode(self, input):
        encoded_input = self.model.encode(input)
        return encoded_input.tolist()[0]

    def _making_decoder_data(self):
        vocab_path = self.model.get_path_to_vocab_file()
        with open(vocab_path, "r") as file:
            data = json.load(file)
        self.vocab_len = len(data)
        swaped_data = {}
        swaped_data = {id: token for token, id in data.items()}
        self.id_token_vocab = swaped_data

    def _my_decode(self, input_id):
        return self.id_token_vocab[input_id]

    def _encode_needed_characters(self):
        self.integer_encoded = [self._my_encode(char)[0] for char in string.digits + "-+"]
        self.number_encoded = self.integer_encoded + self._my_encode(".")
        self.characters_encoded = [self._my_encode(char)[0] for char in string.printable]
        self.boolean_encoded = [self._my_encode("True")[0], self._my_encode("False")[0]]
        self.close_encoded = [self._my_encode(char)[0] for char in string.whitespace + "}\","]

    def _build_masks(self):
        self.number_masks = np.full(self.vocab_len, -np.inf)
        self.number_masks[self.number_encoded + self.close_encoded] = 0

        self.characters_masks = np.full(self.vocab_len, -np.inf)
        self.characters_masks[self.characters_encoded + self.close_encoded] = 0

        self.integer_masks = np.full(self.vocab_len, -np.inf)
        self.integer_masks[self.integer_encoded + self.close_encoded] = 0

        self.boolean_mask = np.full(self.vocab_len, -np.inf)
        self.boolean_mask[self.boolean_encoded] = 0

    def _generate_system_prompt(self):
        prompt =    """You are a function selector. Given a user request, output ONLY the name
                    of the single best matching function from the list below.
                    Do not explain. Do not add punctuation.\n
                    Read each function's description carefully and match it to the user's request by meaning.\n
                    If nothing matches, output: None\n" \
                    Functions:\n"""
        for function in self.functions:
            parameters = ", ".join(
                f"{name} ({function.parameters[name].type.value})"
                for name in function.parameters
            )
            function_prompt = (
                f"\n- name: {function.name}\n"
                f"  description: {function.description}\n"
                f"  parameters: {parameters}\n"
                f"  returns: {function.returns.type.value}\n"
            )
            prompt += function_prompt
        self.system_prompt = prompt
        display_building_prompt()

    def _encode_system_prompt(self):
        self.encoded_system_prompt =  self._my_encode(self.system_prompt)

    def _encode_function_names(self):
        function_names = {}
        for function in self.functions:
            function_names[function] = self._my_encode(function)
        self.function_names = function_names 

    def _encode_function_name_prompt(self):
        self.function_name_prompt = self._my_encode('Output only the function name in the folowing form {"name": "<fn>"} {"name": "')

    def _encode_get_parameter_prompt(self):
        pass

    def _main_loop(self):
        for prompt in self.prompts:
            prompt_output = {
                "prompt": prompt.prompt
            }
            encoded_prompt = self._my_encode(prompt.prompt)
            function_name = self.get_function_name(encoded_prompt)
            parameters = self.get_parameters(encoded_prompt, function_name)
            prompt_output["name"] = function_name
            prompt_output["parameters"] = parameters
            self.output.append(prompt_output)

    def _get_function_name(self, encoded_prompt):
        prompt = self.encoded_system_prompt + encoded_prompt + self.function_name_prompt
        output = []
        while True:
            logits = self.model.get_logits_from_input_ids(prompt)
            max_logit = self._get_max_logit_id(logits)
            decoded_max_logit = self._my_decode(max_logit)
            if "}" in decoded_max_logit or "\"" in decoded_max_logit or " " in decoded_max_logit or "\n" in decoded_max_logit or "," in decoded_max_logit:
                return "".join(output)
            prompt.append(max_logit)
            output.append(decoded_max_logit)

    def _get_max_logit_id(self, logits):
        return np.argmax(logits)

    def _get_function_parameters(self, encoded_prompt, function_name):
        return None
