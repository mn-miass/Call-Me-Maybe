# import logging
# import json
# import string
# import numpy as np


# from llm_sdk import Small_LLM_Model
# from .display_info import display_building_prompt


# logging.basicConfig(
#     filename = "log.log",
#     level = logging.INFO,
#     format = "%(asctime)s - %(levelname)s - %(message)s",
#     filemode = "w"
# )


# class ConstrainedDecoding():
#     def __init__(self, functions, prompts, output_path, model: Small_LLM_Model):
#         self.functions = functions
#         self.prompts = prompts
#         self.output_path = output_path
#         self.model = model
#         self.output = []
#         self.masks = None
#         self._making_decoder_data()
#         self._encode_needed_characters()
#         self._generate_system_prompt()
#         self._encode_system_prompt()
#         self._encode_function_names()
#         self._encode_function_name_prompt()
#         self._get_argument_types()
#         self._main_loop()

#     def _my_encode(self, input):
#         encoded_input = self.model.encode(input)
#         return encoded_input.tolist()[0]

#     def _making_decoder_data(self):
#         vocab_path = self.model.get_path_to_vocab_file()
#         with open(vocab_path, "r") as file:
#             data = json.load(file)
#         swaped_data = {}
#         swaped_data = {id: token for token, id in data.items()}
#         self.id_token_vocab = swaped_data

#     def _my_decode(self, input_id):
#         return self.id_token_vocab[input_id]

#     def _encode_needed_characters(self):
#         self.integer_encoded = [self._my_encode(char)[0] for char in string.digits + "-+"]
#         self.number_encoded = self.integer_encoded + self._my_encode(".")
#         self.characters_encoded = [self._my_encode(char)[0] for char in string.printable]
#         self.boolean_encoded = [self._my_encode("True")[0], self._my_encode("False")[0]]
#         self.close_encoded = [self._my_encode(char)[0] for char in string.whitespace + "}\","]

#     def _build_masks(self, size_logits):
#         self.number_masks = np.full(size_logits, -np.inf)
#         self.number_masks[self.number_encoded + self.close_encoded] = 0

#         self.characters_masks = np.full(size_logits, -np.inf)
#         self.characters_masks[self.characters_encoded + self.close_encoded] = 0

#         self.integer_masks = np.full(size_logits, -np.inf)
#         self.integer_masks[self.integer_encoded + self.close_encoded] = 0

#         self.boolean_mask = np.full(size_logits, -np.inf)
#         self.boolean_mask[self.boolean_encoded] = 0

#     def _generate_system_prompt(self):
#         prompt =    """You are a function selector. Given a user request, output ONLY the name
#                     of the single best matching function from the list below.
#                     Do not explain. Do not add punctuation.\n
#                     Read each function's description carefully and match it to the user's request by meaning.\n
#                     If nothing matches, output: None\n" \
#                     Functions:\n"""
#         for function in self.functions:
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
#         self.system_prompt = prompt
#         display_building_prompt()

#     def _encode_system_prompt(self):
#         self.encoded_system_prompt =  self._my_encode(self.system_prompt)

#     def _encode_function_names(self):
#         function_names = {}
#         for function in self.functions:
#             function_names[function.name] = self._my_encode(function.name)
#         self.function_names = function_names 

#     def _encode_function_name_prompt(self):
#         self.function_name_prompt = self._my_encode('Output only the function name in the folowing form {"name": "<fn>"} {"name": "')

#     def _encode_get_parameter_prompt(self, parameter, parameter_type):
#         if parameter_type == "STRING":
#             value_placeholder = '"value"'
#         else:
#             value_placeholder = "value"
#         return self._my_encode(
#             f'Output only the parameter {parameter} in the following form '
#             f'{{"{parameter}": {value_placeholder}}}'
#         )

#     def _get_argument_types(self):
#         types = {}
#         for function in self.functions:
#             types[function.name] = {}
#             for parameter in function.parameters:
#                 types[function.name][parameter] = function.parameters[parameter].type.name
#         self.types = types

#     def _main_loop(self):
#         for prompt in self.prompts:
#             prompt_output = {
#                 "prompt": prompt.prompt
#             }
#             encoded_prompt = self._my_encode(prompt.prompt)
#             function_name = self._get_function_name(encoded_prompt)
#             parameters = self._get_function_parameters(encoded_prompt, function_name)
#             prompt_output["name"] = function_name
#             prompt_output["parameters"] = parameters
#             self.output.append(prompt_output)

#     def _get_function_name(self, encoded_prompt):
#         prompt = self.encoded_system_prompt + encoded_prompt + self.function_name_prompt
#         output = []
#         while True:
#             logits = self.model.get_logits_from_input_ids(prompt)
#             max_logit = self._get_max_logit_id(logits)
#             decoded_max_logit = self.model.decode(max_logit)
#             if "}" in decoded_max_logit or "\"" in decoded_max_logit or " " in decoded_max_logit or "\n" in decoded_max_logit or "," in decoded_max_logit:
#                 return "".join(output)
#             prompt.append(max_logit)
#             output.append(decoded_max_logit)

#     def _get_max_logit_id(self, logits):
#         return np.argmax(logits)

#     def _set_all_logits_to_neg_inf_except_valids(self, logits, valid_id):
#         pass

#     def _get_function_parameters(self, encoded_prompt, function_name):
#         output = {}
#         for parameter in self.types[function_name]:
#             prompt = self.encoded_system_prompt + encoded_prompt
#             type_para = self.types[function_name][parameter]
#             prompt = prompt +  self._encode_get_parameter_prompt(parameter, type_para)
#             output[parameter] = []
#             while True:
#                 logits = self.model.get_logits_from_input_ids(prompt)
#                 if not self.masks:
#                     self._build_masks(len(logits))
#                     self.masks = True
#                 if type_para == "STRING":
#                     logits = logits + self.characters_masks
#                 elif type_para == "NUMBER":
#                     logits = logits + self.number_masks
#                 elif type_para == "INTEGER":
#                     logits = logits + self.integer_masks
#                 else:
#                     logits = logits + self.boolean_mask
#                 max_logit = self._get_max_logit_id(logits)
#                 max_logit_decoded = self.model.decode(max_logit)
#                 if "}" in max_logit_decoded or "\"" in max_logit_decoded or "\n" in max_logit_decoded or "," in max_logit_decoded:
#                     print(max_logit_decoded, flush=True, end="")
#                     output[parameter] = "".join(output[parameter])
#                     break
#                 prompt.append(max_logit)
#                 output[parameter].append(max_logit_decoded)
#         return output

#     def display_output(self):
#         with open(self.output_path, "w") as file:
#             json.dump(self.output, file, indent=2)

# import logging
# import json
# import string
# import numpy as np


# from llm_sdk import Small_LLM_Model
# from .display_info import display_building_prompt


# logging.basicConfig(
#     filename = "log.log",
#     level = logging.INFO,
#     format = "%(asctime)s - %(levelname)s - %(message)s",
#     filemode = "w"
# )


# class ConstrainedDecoding():
#     def __init__(self, functions, prompts, output_path, model: Small_LLM_Model):
#         logging.info(
#             f"Initializing ConstrainedDecoding | functions={len(functions)} "
#             f"prompts={len(prompts)} output_path={output_path}"
#         )
#         self.functions = functions
#         self.prompts = prompts
#         self.output_path = output_path
#         self.model = model
#         self.output = []
#         self.masks = None
#         self._making_decoder_data()
#         self._encode_needed_characters()
#         self._generate_function_prompt()
#         self._generate_system_prompt()
#         self._encode_system_prompt()
#         self._encode_function_names()
#         self._encode_function_name_prompt()
#         self._get_argument_types()
#         logging.info("Setup complete, entering main loop")
#         self._main_loop()

#     def _my_encode(self, input):
#         encoded_input = self.model.encode(input)
#         return encoded_input.tolist()[0]

#     def _making_decoder_data(self):
#         vocab_path = self.model.get_path_to_vocab_file()
#         logging.info(f"Loading vocab file from {vocab_path}")
#         with open(vocab_path, "r") as file:
#             data = json.load(file)
#         swaped_data = {}
#         swaped_data = {id: token for token, id in data.items()}
#         self.id_token_vocab = swaped_data
#         logging.info(f"Vocab loaded | entries={len(swaped_data)}")

#     def _my_decode(self, input_id):
#         return self.id_token_vocab[input_id]

#     def _encode_needed_characters(self):
#         logging.debug("Encoding needed character classes")
#         self.integer_encoded = [self._my_encode(char)[0] for char in string.digits + "-+"]
#         self.number_encoded = self.integer_encoded + self._my_encode(".")
#         self.characters_encoded = [self._my_encode(char)[0] for char in string.printable]
#         self.boolean_encoded = [self._my_encode("True")[0], self._my_encode("False")[0]]
#         self.close_encoded = [self._my_encode(char)[0] for char in string.whitespace + "}\","]
#         logging.info(
#             f"Character classes encoded | integer={len(self.integer_encoded)} "
#             f"number={len(self.number_encoded)} characters={len(self.characters_encoded)} "
#             f"boolean={len(self.boolean_encoded)} close={len(self.close_encoded)}"
#         )

#     def _build_masks(self, size_logits):
#         logging.info(f"Building masks | size_logits={size_logits}")
#         self.number_masks = np.full(size_logits, -np.inf)
#         self.number_masks[self.number_encoded + self.close_encoded] = 0

#         self.characters_masks = np.full(size_logits, -np.inf)
#         self.characters_masks[self.characters_encoded + self.close_encoded] = 0

#         self.integer_masks = np.full(size_logits, -np.inf)
#         self.integer_masks[self.integer_encoded + self.close_encoded] = 0

#         self.boolean_mask = np.full(size_logits, -np.inf)
#         self.boolean_mask[self.boolean_encoded] = 0
#         logging.info("Masks built successfully")

#     def _generate_system_prompt(self):
#         logging.debug("Generating system prompt")
#         header = (
#             "You are a function selector. Given a user request, output ONLY the name "
#             "of the single best matching function from the list below. "
#             "Do not explain. Do not add punctuation.\n"
#             "Read each function's description carefully and match it to the "
#             "request by meaning.\n\nFunctions:"
#         )
#         self.system_prompt = header + "".join(self.function_prompt.values()) + "\n"
#         display_building_prompt()
#         logging.info(f"System prompt built | length_chars={len(self.system_prompt)}")
#         logging.debug(f"System prompt content:\n{self.system_prompt}")

#     def _generate_function_prompt(self):
#         self.function_prompt = {}
#         for function in self.functions:
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
#             self.function_prompt[function.name] = function_prompt
    
#     def _encode_system_prompt(self):
#         self.encoded_system_prompt =  self._my_encode(self.system_prompt)
#         logging.info(f"System prompt encoded | tokens={len(self.encoded_system_prompt)}")

#     def _encode_function_names(self):
#         function_names = {}
#         for function in self.functions:
#             function_names[function.name] = self._my_encode(function.name)
#         self.function_names = function_names
#         logging.debug(f"Function names encoded | functions={list(function_names.keys())}")

#     def _encode_function_name_prompt(self):
#         self.function_name_prompt = self._my_encode('Output only the function name in the folowing form {"name": "<fn>"} {"name": "')
#         logging.debug("Function name prompt encoded")

#     def _encode_get_parameter_prompt(self, parameter, parameter_type, user_prompt_text):
#         if parameter_type == "STRING":
#             text = (
#                 f'extraxt the exact full value of parameter "{parameter}" from the request above based of function parameter. only close \" at the end\n'
#                 f'{{"{parameter}": "'
#             )
#         else:
#             text = (
#                 f'Request: "{user_prompt_text}"\n'
#                 f'Extract the value of parameter "{parameter}" from the request above.\n'
#                 f'{{"{parameter}": '
#             )
#         return self._my_encode(text)

#     def _get_argument_types(self):
#         types = {}
#         for function in self.functions:
#             types[function.name] = {}
#             for parameter in function.parameters:
#                 types[function.name][parameter] = function.parameters[parameter].type.name
#         self.types = types
#         logging.debug(f"Argument types resolved: {types}")

#     def _main_loop(self):
#         logging.info(f"ENTER _main_loop | total_prompts={len(self.prompts)}")
#         for idx, prompt in enumerate(self.prompts):
#             logging.info(f"--- Processing prompt #{idx}: {prompt.prompt!r} ---")
#             prompt_output = {
#                 "prompt": prompt.prompt
#             }
#             encoded_prompt = self._my_encode(prompt.prompt)
#             function_name = self._get_function_name(encoded_prompt)
#             logging.info(f"Prompt #{idx} resolved function name: {function_name!r}")
#             parameters = self._get_function_parameters(encoded_prompt, function_name, prompt.prompt)
#             logging.info(f"Prompt #{idx} resolved parameters: {parameters!r}")
#             prompt_output["name"] = function_name
#             prompt_output["parameters"] = parameters
#             self.output.append(prompt_output)
#         logging.info(f"EXIT _main_loop | total_results={len(self.output)}")

#     def _get_function_name(self, encoded_prompt):
#         logging.debug("ENTER _get_function_name")
#         prompt = self.encoded_system_prompt + encoded_prompt + self.function_name_prompt
#         output = []
#         iterations = 0
#         while True:
#             iterations += 1
#             logging.debug(f"  _get_function_name iteration {iterations}")
#             logits = self.model.get_logits_from_input_ids(prompt)
#             max_logit = self._get_max_logit_id(logits)
#             decoded_max_logit = self.model.decode(max_logit)
#             logging.debug(f"    token id={max_logit} decoded={decoded_max_logit!r}")
#             if "}" in decoded_max_logit or "\n" in decoded_max_logit:
#                 logging.debug(f"  stop token hit after {iterations} iterations")
#                 return "".join(output)
#             prompt.append(max_logit)
#             output.append(decoded_max_logit)

#     def _get_max_logit_id(self, logits):
#         return np.argmax(logits)

#     def _set_all_logits_to_neg_inf_except_valids(self, logits, valid_id):
#         pass

#     def _get_function_parameters(self, encoded_prompt, function_name, row_prompt_text):
#         prompt = self.model.encode(self.function_prompt[function_name]).tolist()[0] + encoded_prompt
#         logging.debug(f"ENTER _get_function_parameters | function_name={function_name}")
#         output = {}
#         for parameter in self.types[function_name]:
#             type_para = self.types[function_name][parameter]
#             logging.debug(f"  resolving parameter {parameter} (type={type_para})")
#             prompt = prompt +  self._encode_get_parameter_prompt(parameter, type_para, row_prompt_text)
#             logging.debug(f"{parameter} prompt: {self.model.decode(prompt)}")
#             output[parameter] = []
#             iterations = 0
#             while True:
#                 iterations += 1
#                 if iterations > 20:
#                     break
#                 logits = self.model.get_logits_from_input_ids(prompt)
#                 if not self.masks:
#                     self._build_masks(len(logits))
#                     self.masks = True
#                 if type_para == "STRING":
#                     logits = logits + self.characters_masks
#                     # if iterations <= 2:
#                     #     logits[self.close_encoded] = -np.inf
#                 elif type_para == "NUMBER":
#                     logits = logits + self.number_masks
#                 elif type_para == "INTEGER":
#                     logits = logits + self.integer_masks
#                 else:
#                     logits = logits + self.boolean_mask
#                 max_logit = self._get_max_logit_id(logits)
#                 max_logit_decoded = self.model.decode(max_logit)
#                 logging.debug(
#                     f"    parameter={parameter} iteration={iterations} "
#                     f"token id={max_logit} decoded={max_logit_decoded!r}"
#                 )
#                 if "}" in max_logit_decoded or "\"" in max_logit_decoded or "\n" in max_logit_decoded or "," in max_logit_decoded:
#                     output[parameter] = "".join(output[parameter])
#                     logging.debug(
#                         f"  parameter {parameter} resolved after {iterations} iterations: "
#                         f"{output[parameter]!r}"
#                     )
#                     break
#                 output[parameter] = "".join(output[parameter])
#                 if "-" in max_logit_decoded or "+" in max_logit_decoded and output[parameter] and type_para != "STRING":
#                     output[parameter] = "".join(output[parameter])
#                     logging.debug(
#                         f"  parameter {parameter} resolved after {iterations} iterations: "
#                         f"{output[parameter]!r}"
#                     )
#                     output[parameter] = "".join(output[parameter])
#                     break
#                 prompt.append(max_logit)
#                 output[parameter] += max_logit_decoded
#                 logging.debug(
#                     f"New input prompt is : \n{self.model.decode(prompt)}\n"
#                 )
#             logging.info(f"Parameter resolved | name={parameter} value={output[parameter]!r}")
#         return output

#     def display_output(self):
#         logging.info(f"Writing output to {self.output_path}")
#         with open(self.output_path, "w") as file:
#             json.dump(self.output, file, indent=2)
#         logging.info("Output written successfully")


## fix the needed characters so it will involve all takens that can include all characters 
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


class ConstrainedDecoding():
    def __init__(self, functions, prompts, output_file, model: Small_LLM_Model):
        self.functions = functions
        self.prompts = prompts
        self.output_file = output_file
        self.model = model
        valid_json = []
        self._get_function_prompts()
        self._get_encoded_function_prompts()
        self._get_system_prompt()
        self._get_encoded_system_prompt()
        self._get_needed_characters()
        self._get_needed_tokens_for_function()
        self._get_encoded_prompt()

    def _get_function_prompts(self):
        self.functions_prompt = {}
        for function in self.functions:
            prompt = f"- {function.name}: {function.description}  parameters: "
            params = ", ".join(f"{parameter} ({value_type.type.value})"for parameter, value_type in function.parameters.items())
            prompt += params
            self.functions_prompt[function.name] = prompt + "\n"
        self.functions_prompt["None"] = "- None: Use ONLY when the request matches none of the functions above."

    def _get_encoded_function_prompts(self):
        self.encoded_function_prompt = {}
        for name, prompt in self.functions_prompt.items():
            self.encoded_function_prompt[name] = self.model.encode(prompt).tolist()[0]
        self.encoded_function_prompt

    def _get_system_prompt(self):
        SELECTOR_HEADER = (
            "You are a function selector. Given a user request, output ONLY the "
            "name of the single best matching function.\n"
            "Rules:\n"
            "- Output must be exactly one of the function names listed below.\n"
            "- Do not explain. Do not add punctuation. Do not add quotes.\n"
            "- If none match, output: fn_invalid_name\n\n"
            "Functions:\n"
        )
        self.system_prompt = (
            SELECTOR_HEADER
            + "".join(f"{prompt}" for prompt in self.functions_prompt.values())
        )

    def _get_encoded_system_prompt(self):
        self.encoded_system_prompt = self.model.encode(self.system_prompt).tolist()[0]

    def _get_needed_characters(self):
        self.integer_characters = [char for char in string.digits]
        self.sign_characters = ["+", "-"]
        self.decimal_point_character = ["."]
        self.strings_characters = [char for char in string.printable]
        self.stop_characters = [",", "\n", " ", "}"]

    def _get_needed_tokens_for_function(self):
        functions_name_tokens = []
        for function in self.functions_prompt:
            tokens = self.model.encode(function).tolist()[0]
            if tokens not in functions_name_tokens:
                functions_name_tokens += tokens
        self.functions_name_tokens = list(functions_name_tokens)
        print(self.functions_name_tokens, flush=True)

    def _get_encoded_prompt(self):
        self.encoded_prompt = {}
        for prompt in self.prompts:
            self.encoded_prompt[prompt.prompt] = self.model.encode(prompt.prompt).tolist()[0]

    def main_loop(self):
        for prompt in self.prompts:
            prompt_data = {
                "prompt": prompt.prompt
            }
            name = self._get_function_name(prompt.prompt)
            prompt_data = {
                "name": name
            }
            print(prompt_data)

    def _get_function_name(self, prompt):
        output = []
        current_prompt = self.encoded_system_prompt +  self.encoded_prompt[prompt]
        text = (
            self.system_prompt
            + "\nUser request: " + prompt + "\n"
            + "Best function name: "
        )
        current_prompt = self.model.encode(text).tolist()[0]
        needed_characters_encoded = [self.model.encode(char).tolist()[0][0] for char in self.stop_characters]
        print(needed_characters_encoded, flush=True)
        needed_characters_encoded += list(self.functions_name_tokens)
        while True:
            logits = self.model.get_logits_from_input_ids(current_prompt)
            mask = np.full(len(logits), -np.inf)
            mask[needed_characters_encoded] = 0
            logits += mask
            max = self._get_max(logits)
            decoded_max = self.model.decode(max)
            if decoded_max in self.stop_characters:
                print(decoded_max)
                break
            output.append(decoded_max)
            current_prompt.append(max)
            print(decoded_max)
        return "".join(output)

    def _get_max(self, logits):
        return np.argmax(logits)