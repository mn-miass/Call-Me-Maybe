import logging
import json
import string
import numpy as np


from llm_sdk import Small_LLM_Model
from .display_info import display_building_prompt


logging.basicConfig(
    filename = "log.log",
    level = logging.DEBUG,
    format = "%(asctime)s - %(levelname)s - %(message)s",
    filemode = "w"
)


class ConstrainedDecoding():
    def __init__(self, functions, prompts, output_file, model: Small_LLM_Model):
        logging.info(
            f"Initializing ConstrainedDecoding | functions={len(functions)} "
            f"prompts={len(prompts)} output_file={output_file}"
        )
        self.functions = functions
        self.prompts = prompts
        self.output_file = output_file
        self.model = model
        self.valid_json = []
        self._get_function_prompts()
        self._get_encoded_function_prompts()
        self._get_system_prompt()
        self._get_encoded_system_prompt()
        self._get_vacabulary()
        self._get_needed_characters()
        self._get_argument_types()
        self._get_needed_tokens_for_function()
        self._get_encoded_prompt()
        logging.info("Setup complete")

    def _get_function_prompts(self):
        self.functions_prompt = {}
        for function in self.functions:
            prompt = f"- {function.name}: {function.description}  parameters: "
            params = ", ".join(f"{parameter} ({value_type.type.value})"for parameter, value_type in function.parameters.items())
            prompt += params
            self.functions_prompt[function.name] = prompt + "\n"
            logging.debug(f"Function prompt built: {self.functions_prompt[function.name]!r}")
        self.functions_prompt["None"] = "- None: Use ONLY when the request matches none of the functions above."
        logging.info(f"Function prompts built | count={len(self.functions_prompt)}")

    def _get_encoded_function_prompts(self):
        self.encoded_function_prompt = {}
        for name, prompt in self.functions_prompt.items():
            self.encoded_function_prompt[name] = self.model.encode(prompt).tolist()[0]
            logging.debug(f"Function prompt encoded | name={name} tokens={len(self.encoded_function_prompt[name])}")

    def _get_system_prompt(self):
        SELECTOR_HEADER = (
            "You are a function selector. Given a user request, output ONLY the "
            "name of the single best matching function.\n"
            "Rules:\n"
            "- Output must be exactly one of the function names listed below.\n"
            "- Do not explain. Do not add punctuation. Do not add quotes.\n"
            "- If the request does not match any function, output: None\n\n"
            "Functions:\n"
        )
        examples = "\nExamples:\n" + "".join(
            f"User request: {f.description}\nBest function name: {f.name}\n"
            for f in self.functions
        )
        examples += (
            "User request: What is the weather today?\nBest function name: None\n"
            "User request: Tell me a joke\nBest function name: None\n"
        )
        self.system_prompt = (
            SELECTOR_HEADER
            + "".join(self.functions_prompt.values())
            + examples
        )
        logging.info(f"System prompt built | length_chars={len(self.system_prompt)}")
        logging.debug(f"System prompt content:\n{self.system_prompt}")

    def _get_encoded_system_prompt(self):
        self.encoded_system_prompt = self.model.encode(self.system_prompt).tolist()[0]
        logging.info(f"System prompt encoded | tokens={len(self.encoded_system_prompt)}")

    def _get_vacabulary(self):
        vocab_path = self.model.get_path_to_vocab_file()
        logging.info(f"Loading vocab file from {vocab_path}")
        with open(vocab_path, "r") as file:
            self.vocab_data = json.load(file)
        logging.info(f"Vocab loaded | entries={len(self.vocab_data)}")

    def _get_needed_characters(self):
        self.integer_characters = [token for token in self.vocab_data if all(char in string.digits for char in token)]
        self.integer_characters = [self.model.encode(token).tolist()[0][0] for token in self.integer_characters]
        self.sign_characters = ["+", "-"]
        self.sign_characters = [self.model.encode(token).tolist()[0][0] for token in self.sign_characters]
        self.decimal_point_character = ["."]
        self.decimal_point_character = self.model.encode(self.decimal_point_character).tolist()[0][0]
        self.strings_characters = [
            token_id
            for token, token_id in self.vocab_data.items()
            if all(char in string.printable for char in token.replace("Ġ", " ").replace("Ċ", "\n"))
        ]
        self.stop_characters = [",", "\n", "}", "}}", ", ", " ,", "\",", "\"}", "\"}\n", "\", ", ')",', "}\n\n"]
        self.stop_characters = [self.model.encode(token).tolist()[0][0] for token in self.stop_characters]
        logging.info(
            f"Character classes built | integer={len(self.integer_characters)} "
            f"sign={len(self.sign_characters)} strings={len(self.strings_characters)} "
            f"stop={len(self.stop_characters)}"
        )
        logging.debug(
            f"decimal_point={self.decimal_point_character!r} "
            f"(type={type(self.decimal_point_character).__name__})"
        )
        logging.debug(f"stop ids={self.stop_characters}")

    def _get_needed_tokens_for_function(self):
        functions_name_tokens = []
        for function in self.functions_prompt:
            tokens = self.model.encode(function).tolist()[0]
            logging.debug(f"Function name tokens | name={function} ids={tokens}")
            if tokens not in functions_name_tokens:
                functions_name_tokens += tokens
        self.functions_name_tokens = list(functions_name_tokens)
        logging.info(f"Function name tokens ready | count={len(self.functions_name_tokens)}")

    def _get_encoded_prompt(self):
        self.encoded_prompt = {}
        for prompt in self.prompts:
            self.encoded_prompt[prompt.prompt] = self.model.encode(prompt.prompt).tolist()[0]
        logging.info(f"User prompts encoded | count={len(self.encoded_prompt)}")

    def main_loop(self):
        logging.info(f"ENTER main_loop | total_prompts={len(self.prompts)}")
        for idx, prompt in enumerate(self.prompts):
            logging.info(f"--- Processing prompt #{idx}: {prompt.prompt!r} ---")
            prompt_data = {}
            prompt_data["prompt"] = prompt.prompt
            name = self._get_function_name(prompt.prompt)
            logging.info(f"Prompt #{idx} resolved function name: {name!r}")
            prompt_data["name"] = name
            if name != "None":
                parameters = self._get_parameters(prompt.prompt, name)
            else:
                logging.warning(f"Prompt #{idx} matched no function")
                parameters = None
            logging.info(f"Prompt #{idx} resolved parameters: {parameters!r}")
            prompt_data["parameters"] = parameters
            self.valid_json.append(prompt_data)
        logging.info(f"EXIT main_loop | total_results={len(self.valid_json)}")


    def _get_function_name(self, prompt):
        logging.debug("ENTER _get_function_name")
        output = []
        current_prompt = self.encoded_system_prompt +  self.encoded_prompt[prompt]
        text = (
            self.system_prompt
            + "\nUser request: " + prompt + "\n"
            + "Best function name: "
        )
        current_prompt = self.model.encode(text).tolist()[0]
        needed_characters_encoded = []
        needed_characters_encoded += self.stop_characters
        needed_characters_encoded += list(self.functions_name_tokens)
        iterations = 0
        while True:
            iterations += 1
            logits = self.model.get_logits_from_input_ids(current_prompt)
            mask = np.full(len(logits), -np.inf)
            mask[needed_characters_encoded] = 0
            logits += mask
            max = self._get_max(logits)
            decoded_max = self.model.decode(max)
            logging.debug(f"  name iteration={iterations} token id={max} decoded={decoded_max!r}")
            if max in self.stop_characters:
                logging.debug(f"  stop token hit after {iterations} iterations")
                break
            output.append(decoded_max)
            current_prompt.append(max)
        return "".join(output)

    def _get_argument_types(self):
        self.types = {}
        for function in self.functions:
            self.types[function.name] = {}
            for parameter in function.parameters:
                self.types[function.name][parameter] = function.parameters[parameter].type.value
        logging.debug(f"Argument types resolved: {self.types}")

    def _get_parameters(self, prompt, function_name):
        logging.debug(f"ENTER _get_parameters | function_name={function_name}")
        output = {}
        main_prompt = (
            "Extract the function arguments from the request as JSON.\n"
            "Copy values exactly from the request. "
            "For a regex, write a short pattern that matches only the parts to replace. "
            "For a replacement, write the exact text that goes in place of each match "
            "If you are asked for a path give the full path"
            "(use the symbol itself when the request names a symbol).\n\n"
            'Request: "Replace all digits in \'a1b2\' with X"\n'
            '{"source_string": "a1b2", "regex": "\\\\d", "replacement": "X"}\n'
            'Request: "Replace all spaces in \'a b\' with underscores"\n'
            '{"source_string": "a b", "regex": "\\\\s", "replacement": "_"}\n'
            'Request: "Replace every lowercase letter in \'aB1\' with #"\n'
            '{"source_string": "aB1", "regex": "[a-z]", "replacement": "#"}\n'
            'Request: "Replace the word \'red\' with \'blue\' in \'a red car\'"\n'
            '{"source_string": "a red car", "regex": "red", "replacement": "blue"}\n'
            'Request: "\'any operation on numbers\' 7 and 8"\n'
            '{"a": 7, "b": 8}\n\n'
            + self.functions_prompt[function_name]
            + f'\nRequest: "{prompt}"\n'
        )
        prefix = "{"
        for parameter in self.types[function_name]:
            para_type = self.types[function_name][parameter]
            logging.debug(f"  resolving parameter {parameter} (type={para_type})")
            prefix += f'"{parameter}": ' + ('"' if para_type == "string" else "")
            current_prompt = self.model.encode(main_prompt + prefix).tolist()[0]
            logging.debug(f"  {parameter} prompt tail: {(main_prompt + prefix)[-200:]!r}")
            if para_type == "integer":
                allowed_char = self.integer_characters + self.sign_characters + self.stop_characters
            if para_type == "number":
                allowed_char = self.integer_characters + self.sign_characters + self.decimal_point_character + self.stop_characters
            else:
                allowed_char = self.strings_characters
            logging.debug(f"  {parameter} allowed tokens={len(allowed_char)}")
            param = ""
            mask = None
            for iteration in range(100):
                logits = self.model.get_logits_from_input_ids(current_prompt)
                if mask is None:
                    mask = np.full(len(logits), -np.inf)
                    mask[allowed_char] = 0
                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    top = np.argsort(logits + mask)[-5:][::-1]
                    logging.debug(
                        f"    {parameter} iteration={iteration} "
                        f"top5={[self.model.decode(int(t)) for t in top]}"
                    )
                max_logit = int(self._get_max(logits + mask))
                decoded_max = self.model.decode(max_logit)
                logging.debug(
                    f"    {parameter} iteration={iteration} "
                    f"token id={max_logit} decoded={decoded_max!r}"
                )
                if para_type == "string":
                    if '"' in decoded_max:
                        param += decoded_max.split('"')[0]
                        logging.debug(f"  {parameter} closing quote found after {iteration} iterations")
                        break
                elif max_logit in self.stop_characters:
                    logging.debug(f"  {parameter} stop token hit after {iteration} iterations")
                    break
                param += decoded_max
                current_prompt.append(max_logit)
            else:
                logging.warning(f"  {parameter} hit the 100 token limit without stopping")
            logging.debug(f"  {parameter} raw value={param!r}")
            if para_type == "string":
                prefix += param + '", '
                try:
                    param = json.loads('"' + param + '"')
                except ValueError:
                    logging.warning(f"  {parameter} json.loads failed, keeping raw value {param!r}")
                param = self._check_string(param)
            elif para_type == "number" or para_type == "integer":
                param = self._check_int(param, para_type)
            else:
                prefix += param + ", "
            logging.info(f"Parameter resolved | name={parameter} type={para_type} value={param!r}")
            output[parameter] = param
        return output

    @staticmethod
    def _get_max(logits):
        return np.argmax(logits)

    def display_json(self):
        logging.info(f"Writing output to {self.output_file}")
        with open(self.output_file, "w") as file:
            json.dump(self.valid_json, file, indent=2)
        logging.info("Output written successfully")


    @staticmethod
    def _check_int(number, para_type):
        logging.debug(f"_check_int raw={number!r} type={para_type}")
        number = number.strip()
        if number.startswith("-"):
            parts = number[1:].split('-', 1)
            number = '-' + parts
        if number.startswith("+"):
            number = number[1:].split('+', 1)
        if para_type == "number":
            return float(number)
        else:
            return int(number)

    @staticmethod
    def _check_string(string):
        string = string.strip()
        if string[0] == "\"":
            string = string[1:]
        return string
