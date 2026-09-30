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
            "- If none match, output: fn_invalid_name\n\n" \
            "Functions:\n"
        )
        self.system_prompt = (
            SELECTOR_HEADER
            + "".join(f"{prompt}" for prompt in self.functions_prompt.values())
        )

    def _get_encoded_system_prompt(self):
        self.encoded_system_prompt = self.model.encode(self.system_prompt).tolist()[0]

    def _get_vacabulary(self):
        vocab_path = self.model.get_path_to_vocab_file()
        with open(vocab_path, "r") as file:
            self.vocab_data = json.load(file)

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
        self.stop_characters = [",", "\n", "}", "}}", ", ", " ,", "\",", "\"}", "\"}\n", "\", ", ')",', ]
        self.stop_characters = [self.model.encode(token).tolist()[0][0] for token in self.stop_characters]

    def _get_needed_tokens_for_function(self):
        functions_name_tokens = []
        for function in self.functions_prompt:
            tokens = self.model.encode(function).tolist()[0]
            if tokens not in functions_name_tokens:
                functions_name_tokens += tokens
        self.functions_name_tokens = list(functions_name_tokens)


    def _get_encoded_prompt(self):
        self.encoded_prompt = {}
        for prompt in self.prompts:
            self.encoded_prompt[prompt.prompt] = self.model.encode(prompt.prompt).tolist()[0]

    def main_loop(self):
        for prompt in self.prompts:
            prompt_data = {}
            prompt_data["prompt"] = prompt.prompt
            name = self._get_function_name(prompt.prompt)
            prompt_data["name"] = name
            parameters = self._get_parameters(prompt.prompt, name)
            prompt_data["parameters"] = parameters
            self.valid_json.append(prompt_data)
    
    def load_string(self):
        with open("string.json", "w") as file:
            json.dump(self.strings_characters, file, indent=2)

    def load_stop(self):
        with open("stop.json", "w") as file:
            json.dump(self.stop_characters, file, indent=2)

    def load_integers(self):
        with open("integer.json", "w") as file:
            json.dump(self.integer_characters, file, indent=2)


    def _get_function_name(self, prompt):
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
        while True:
            logits = self.model.get_logits_from_input_ids(current_prompt)
            mask = np.full(len(logits), -np.inf)
            mask[needed_characters_encoded] = 0
            logits += mask
            max = self._get_max(logits)
            decoded_max = self.model.decode(max)
            if max in self.stop_characters:
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

    def _get_parameters(self, prompt, function_name):
        output = {}
        main_prompt = self.functions_prompt[function_name] + '"' + prompt + '"' + "\n"
        main_prompt = self.model.encode(main_prompt).tolist()[0]
        current_prompt = main_prompt
        for parameter in self.types[function_name]:
            if parameter != "regex":
                current_prompt = current_prompt + self.model.encode("Copy the argument values exactly from the request. Do not calculate or modify {" + f'"{parameter}" : ' + "").tolist()[0]
            else:
                output["regex"] = self.get_regex(prompt)
                continue
            para_type = self.types[function_name][parameter]
            if para_type == "number":
                allowed_char = self.integer_characters + self.sign_characters + self.decimal_point_character + self.stop_characters
            elif para_type == "string":
                allowed_char = self.strings_characters + self.stop_characters
            param = []
            while True:
                logits = self.model.get_logits_from_input_ids(current_prompt)
                mask = np.full(len(logits), -np.inf)
                mask[allowed_char] = 0
                logits += mask
                max_logit = self._get_max(logits)
                decoded_max = self.model.decode(max_logit)
                if max_logit in self.stop_characters:
                    if ")" in decoded_max:
                        param += ")"
                    output[parameter] = "".join(param)
                    break
                current_prompt.append(max_logit)
                param.append(decoded_max)
            if para_type == "string":
                output[parameter] = self._check_string(output[parameter])
            current_prompt = main_prompt
        return output

    @staticmethod
    def _get_max(logits):
        return np.argmax(logits)

    def display_json(self):
        with open(self.output_file, "w") as file:
            json.dump(self.valid_json, file, indent=2)

    def get_regex(self, request):
        REGEX_HINTS = {
            "digit": "\\d",
            "number": "\\d+",
            "vowel": "[aeiouAEIOU]",
            "constant": "[b-df-hj-mp-tv-zB-DF-HJ-NP-TV-Z]",
            "whitespace": "\\s+",
            "space": "\\s+",
            "upercase": "[A-Z]",
            "lowercase": "[a-z]",
            "punctuation": "[^\\w\\s]"
        }
        text = request.lower()
        prompt = "You will be provided by a text: \"text\" try to extract from it the right regex hint from this bellow"
        prompt += f"text: {text}"
        prompt += "".join(f" - {k}\n" for k in REGEX_HINTS)
        prompt += "correct regex hint is: "
        encoded_prompt = self.model.encode(prompt).tolist()[0]
        while True:
            logits = self.model.get_logits_from_input_ids(encoded_prompt)
            max_logit = self._get_max(logits)
            print(self.model.decode(max_logit), flush=True, end="")
            encoded_prompt.append(max_logit)
    


    @staticmethod
    def _check_string(string):
        string = string.strip()
        if string[0] == "\"":
            string = string[1:]
        return string
