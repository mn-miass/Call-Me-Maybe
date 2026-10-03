import json
import logging
import string
from typing import Any, Dict, List, Optional, Union

import numpy as np
import numpy.typing as npt

from llm_sdk import Small_LLM_Model  # type: ignore


logging.basicConfig(
    filename="log.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w"
)


class ConstrainedDecoding():
    """Function calling with schema-guided constrained decoding.

    The JSON structure is written by the code. The language model is
    only asked for the function name and for each argument value, and
    its logits are masked so that only valid tokens can be chosen.
    """

    def __init__(
        self,
        functions: List[Any],
        prompts: List[Any],
        output_file: str,
        model: Small_LLM_Model,
    ) -> None:
        """Prepare prompts, vocabulary and token sets for decoding.

        Args:
            functions: Function definitions (name, description, parameters).
            prompts: User requests, each with a ``prompt`` attribute.
            output_file: Path of the JSON file to write.
            model: The language model wrapper.
        """
        logging.info(
            f"Initializing ConstrainedDecoding | functions={len(functions)} "
            f"prompts={len(prompts)} output_file={output_file}"
        )
        self.functions = functions
        self.prompts = prompts
        self.output_file = output_file
        self.model = model
        self.valid_json: List[Dict[str, Any]] = []
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

    def _get_function_prompts(self) -> None:
        """Build one prompt line per function, plus a ``None`` entry.

        Fills ``self.functions_prompt`` with lines such as
        ``- name: description  parameters: a (number), b (string)``.
        """
        self.functions_prompt: Dict[str, str] = {}
        for function in self.functions:
            prompt = f"- {function.name}: {function.description}  parameters: "
            params = ", ".join(
                f"{parameter} ({value_type.type.value})"
                for parameter, value_type in function.parameters.items()
            )
            prompt += params
            self.functions_prompt[function.name] = prompt + "\n"
            logging.debug(
                "Function prompt built: "
                f"{self.functions_prompt[function.name]!r}"
            )
        self.functions_prompt["None"] = (
            "- None: Use ONLY when the request matches "
            "none of the functions above."
        )
        logging.info(
            f"Function prompts built | count={len(self.functions_prompt)}"
        )

    def _get_encoded_function_prompts(self) -> None:
        """Tokenize every function prompt line.

        Fills ``self.encoded_function_prompt`` (name -> token ids).
        """
        self.encoded_function_prompt: Dict[str, List[int]] = {}
        for name, prompt in self.functions_prompt.items():
            self.encoded_function_prompt[name] = (
                self.model.encode(prompt).tolist()[0]
            )
            logging.debug(
                f"Function prompt encoded | name={name} "
                f"tokens={len(self.encoded_function_prompt[name])}"
            )

    def _get_system_prompt(self) -> None:
        """Build the function-selection system prompt.

        Combines the instructions, the function list and few-shot examples
        (including two ``None`` examples) into ``self.system_prompt``.
        """
        SELECTOR_HEADER = (
            "You are a function selector. Given a user request, output ONLY "
            "the name of the single best matching function.\n"
            "Rules:\n"
            "- Output must be exactly one of the function names listed "
            "below.\n"
            "- Do not explain. Do not add punctuation. Do not add quotes.\n"
            "- If the request does not match any function, output: None\n\n"
            "Functions:\n"
        )
        examples = "\nExamples:\n" + "".join(
            f"User request: {f.description}\nBest function name: {f.name}\n"
            for f in self.functions
        )
        self.system_prompt: str = (
            SELECTOR_HEADER
            + "".join(self.functions_prompt.values())
            + examples
        )
        logging.info(
            f"System prompt built | length_chars={len(self.system_prompt)}"
        )
        logging.debug(f"System prompt content:\n{self.system_prompt}")

    def _get_encoded_system_prompt(self) -> None:
        """Tokenize the system prompt into ``self.encoded_system_prompt``.
        """
        self.encoded_system_prompt: List[int] = (
            self.model.encode(self.system_prompt).tolist()[0]
        )
        logging.info(
            f"System prompt encoded | tokens={len(self.encoded_system_prompt)}"
        )

    def _get_vacabulary(self) -> None:
        """Load the model vocabulary file (token -> id) into
        ``self.vocab_data``.
        """
        vocab_path = self.model.get_path_to_vocab_file()
        logging.info(f"Loading vocab file from {vocab_path}")
        with open(vocab_path, "r") as file:
            self.vocab_data: Dict[str, int] = json.load(file)
        logging.info(f"Vocab loaded | entries={len(self.vocab_data)}")

    def _get_needed_characters(self) -> None:
        """Build the token id sets used to mask the logits.

        Creates the integer, sign, decimal point, string and stop token id
        lists. Digit and string ids come from the vocabulary; sign, decimal
        point and stop ids come from encoding the characters.
        """
        self.integer_characters: List[int] = [
            token_id
            for token, token_id in self.vocab_data.items()
            if token and all(char in string.digits for char in token)
        ]
        self.sign_characters: List[int] = [
            self.model.encode(token).tolist()[0][0] for token in ["+", "-"]
        ]
        self.decimal_point_character: List[int] = [
            self.model.encode(".").tolist()[0][0]
        ]
        self.space_characters: List[int] = [
            token_id
            for token, token_id in self.vocab_data.items()
            if token.replace("Ġ", "") in ("", "-", "+")
        ]
        self.strings_characters: List[int] = [
            token_id
            for token, token_id in self.vocab_data.items()
            if all(
                char in string.printable
                for char in token.replace("Ġ", " ").replace("Ċ", "\n")
            )
        ]
        stop_tokens = [
            ",", "\n", "}", "}}", ", ", " ,", "\",", "\"}", "\"}\n",
            "\", ", ')",', "}\n\n", "}\n"
        ]
        self.stop_characters: List[int] = [
            self.model.encode(token).tolist()[0][0] for token in stop_tokens
        ]
        self.boolean_characters: List[int] = [
            token_id
            for token, token_id in self.vocab_data.items()
            if token.replace("Ġ", "") in ("", "true", "false")
        ]
        logging.info(
            f"Character classes built | "
            f"integer={len(self.integer_characters)} "
            f"sign={len(self.sign_characters)} "
            f"space={len(self.space_characters)} "
            f"strings={len(self.strings_characters)} "
            f"stop={len(self.stop_characters)}"
        )
        logging.debug(
            f"decimal_point={self.decimal_point_character!r} "
            f"(type={type(self.decimal_point_character).__name__})"
        )
        logging.debug(f"stop ids={self.stop_characters}")

    def _get_needed_tokens_for_function(self) -> None:
        """Collect the token ids that can appear in a function name.

        Fills ``self.functions_name_tokens`` without duplicates. The
        ``None`` entry is included so the model can answer ``None``.
        """
        functions_name_tokens: List[int] = []
        for function in self.functions_prompt:
            tokens = self.model.encode(function).tolist()[0]
            logging.debug(
                f"Function name tokens | name={function} ids={tokens}"
            )
            for token in tokens:
                if token not in functions_name_tokens:
                    functions_name_tokens.append(token)
        self.functions_name_tokens: List[int] = list(functions_name_tokens)
        logging.info(
            "Function name tokens ready | "
            f"count={len(self.functions_name_tokens)}"
        )

    def _get_encoded_prompt(self) -> None:
        """Tokenize every user prompt into ``self.encoded_prompt``.
        """
        self.encoded_prompt: Dict[str, List[int]] = {}
        for prompt in self.prompts:
            self.encoded_prompt[prompt.prompt] = (
                self.model.encode(prompt.prompt).tolist()[0]
            )
        logging.info(
            f"User prompts encoded | count={len(self.encoded_prompt)}"
        )

    def main_loop(self) -> None:
        """Process every prompt and store the results in ``self.valid_json``.

        For each prompt, the model first chooses a function name. If the
        name is a known function, its parameters are then extracted;
        otherwise ``parameters`` is ``None``.
        """
        logging.info(f"ENTER main_loop | total_prompts={len(self.prompts)}")
        for idx, prompt in enumerate(self.prompts):
            logging.info(
                f"--- Processing prompt #{idx}: {prompt.prompt!r} ---"
            )
            prompt_data: Dict[str, Any] = {}
            prompt_data["prompt"] = prompt.prompt
            name = self._get_function_name(prompt.prompt)
            logging.info(f"Prompt #{idx} resolved function name: {name!r}")
            prompt_data["name"] = name
            parameters: Optional[Dict[str, Any]]
            if name in self.types:
                parameters = self._get_parameters(prompt.prompt, name)
            else:
                logging.warning(f"Prompt #{idx} matched no function")
                parameters = None
            logging.info(f"Prompt #{idx} resolved parameters: {parameters!r}")
            prompt_data["parameters"] = parameters
            self.valid_json.append(prompt_data)
        logging.info(f"EXIT main_loop | total_results={len(self.valid_json)}")

    def _get_function_name(self, prompt: str) -> str:
        """Let the model choose a function name under a logit mask.

        Only function name tokens and stop tokens are allowed. Generation
        ends on a stop token or after 50 tokens.

        Args:
            prompt: The user request.

        Returns:
            The generated function name (may be ``None`` or empty).
        """
        logging.debug("ENTER _get_function_name")
        output: List[str] = []
        text = (
            self.system_prompt
            + "\nUser request: " + prompt + "\n"
            + "Best function name: "
        )
        current_prompt: List[int] = self.model.encode(text).tolist()[0]
        needed_characters_encoded: List[int] = []
        needed_characters_encoded += self.stop_characters
        needed_characters_encoded += list(self.functions_name_tokens)
        iterations = 0
        while True:
            iterations += 1
            logits = self.model.get_logits_from_input_ids(current_prompt)
            mask = np.full(len(logits), -np.inf)
            mask[needed_characters_encoded] = 0
            logits = logits + mask
            max_id = self._get_max(logits)
            decoded_max = self.model.decode(max_id)
            logging.debug(
                f"  name iteration={iterations} "
                f"token id={max_id} decoded={decoded_max!r}"
            )
            if max_id in self.stop_characters:
                logging.debug(
                    f"  stop token hit after {iterations} iterations"
                )
                break
            output.append(decoded_max)
            current_prompt.append(max_id)
            if iterations >= 50:
                logging.warning("  function name hit the 50 token limit")
                break
        name = "".join(output)
        if name == "None":
            print("Model couldnt detect the function name for"
                  f" the prompt {prompt}")
            logging.critical(f"Unknown function name for the prompt {prompt}")
            exit()
        return name

    def _get_argument_types(self) -> None:
        """Map each function to the type of each of its parameters.

        Fills ``self.types`` as ``{function: {parameter: type}}``.
        """
        self.types: Dict[str, Dict[str, str]] = {}
        for function in self.functions:
            self.types[function.name] = {}
            for parameter in function.parameters:
                self.types[function.name][parameter] = (
                    function.parameters[parameter].type.value
                )
        logging.debug(f"Argument types resolved: {self.types}")

    def _get_parameters(
        self, prompt: str, function_name: str
    ) -> Dict[str, Any]:
        """Extract the arguments of a function with typed logit masks.

        The JSON prefix is built by the code. For each parameter the model
        generates the value token by token: digits and signs for integers,
        plus a decimal point for numbers, printable tokens until a closing
        quote for strings. Generation is capped at 100 tokens per value.

        Args:
            prompt: The user request.
            function_name: Name of the chosen function.

        Returns:
            A dictionary mapping parameter names to typed values.
        """
        logging.debug(f"ENTER _get_parameters | function_name={function_name}")
        output: Dict[str, Any] = {}
        main_prompt = (
            "Extract the function arguments from the request as JSON.\n"
            "Copy values exactly from the request. "
            "For a regex, write a short pattern that matches only the "
            "parts to replace. "
            "For a replacement, write the exact text that goes in place "
            "of each match. "
            "If you are asked for a path, give the full path "
            "(use the symbol itself when the request names a symbol). "
            "Keep the sign of numbers: a negative number must keep its "
            "minus sign.\n\n"
            'Request: "Replace all digits in \'a1b2\' with X"\n'
            '{"source_string": "a1b2", "regex": "\\\\d", '
            '"replacement": "X"}\n'
            'Request: "Replace all spaces in \'a b\' with underscores"\n'
            '{"source_string": "a b", "regex": "\\\\s", '
            '"replacement": "_"}\n'
            'Request: "Replace every lowercase letter in \'aB1\' with #"\n'
            '{"source_string": "aB1", "regex": "[a-z]", '
            '"replacement": "#"}\n'
            'Request: "Replace the word \'red\' with \'blue\' in '
            '\'a red car\'"\n'
            '{"source_string": "a red car", "regex": "red", '
            '"replacement": "blue"}\n'
            'Request: "Combine the numbers -2 and 3"\n'
            '{"a": -2, "b": 3}\n'
            'Request: "Combine the numbers 7 and 8"\n'
            '{"a": 7, "b": 8}\n\n'
            + self.functions_prompt[function_name]
            + f'\nRequest: "{prompt}"\n'
        )
        prefix = "{"
        for parameter in self.types[function_name]:
            para_type = self.types[function_name][parameter]
            logging.debug(
                f"  resolving parameter {parameter} (type={para_type})"
            )
            prefix += f'"{parameter}":'
            prefix += ' "' if para_type == "string" else ""
            current_prompt: List[int] = (
                self.model.encode(main_prompt + prefix).tolist()[0]
            )
            logging.debug(
                f"  {parameter} prompt tail: "
                f"{(main_prompt + prefix)[-200:]!r}"
            )
            allowed_char: List[int]
            if para_type == "integer":
                allowed_char = (
                    self.integer_characters
                    + self.sign_characters
                    + self.space_characters
                    + self.stop_characters
                )
            elif para_type == "number":
                allowed_char = (
                    self.integer_characters
                    + self.sign_characters
                    + self.space_characters
                    + self.decimal_point_character
                    + self.stop_characters
                )
            elif para_type == "boolean":
                allowed_char = (
                    self.boolean_characters
                    + self.stop_characters
                )
            else:
                allowed_char = self.strings_characters
            logging.debug(f"  {parameter} allowed tokens={len(allowed_char)}")
            param: Any = ""
            mask: Optional[npt.NDArray[Any]] = None
            for iteration in range(100):
                logits = self.model.get_logits_from_input_ids(current_prompt)
                if mask is None:
                    mask = np.full(len(logits), -np.inf)
                    mask[allowed_char] = 0
                max_logit = self._get_max(logits + mask)
                decoded_max = self.model.decode(max_logit)
                logging.debug(
                    f"    {parameter} iteration={iteration} "
                    f"token id={max_logit} decoded={decoded_max!r}"
                )
                if para_type == "string":
                    if '"' in decoded_max:
                        param += decoded_max.split('"')[0].strip()
                        logging.debug(
                            f"  {parameter} closing quote found "
                            f"after {iteration} iterations"
                        )
                        break
                elif max_logit in self.stop_characters:
                    logging.debug(
                        f"  {parameter} stop token hit "
                        f"after {iteration} iterations"
                    )
                    break
                param += decoded_max
                current_prompt.append(max_logit)
            else:
                logging.warning(
                    f"  {parameter} hit the 100 token limit without stopping"
                )
            logging.debug(f"  {parameter} raw value={param!r}")
            if para_type == "string":
                prefix += param + '", '
                try:
                    param = json.loads('"' + param + '"')
                except ValueError:
                    logging.warning(
                        f"  {parameter} json.loads failed, "
                        f"keeping raw value {param!r}"
                    )
                param = self._check_string(param)
            elif para_type == "number" or para_type == "integer":
                prefix += " " + param.strip() + ", "
                param = self._check_int(param, para_type)
            elif para_type == "boolean":
                param = self._check_boolean(param)
                prefix += " " + str(param) + ", "
            else:
                prefix += param + ", "
            logging.info(
                f"Parameter resolved | name={parameter} "
                f"type={para_type} value={param!r}"
            )
            output[parameter] = param
        return output

    @staticmethod
    def _get_max(logits: npt.NDArray[Any]) -> int:
        """Return the index of the highest logit.

        Args:
            logits: Masked logits over the whole vocabulary.

        Returns:
            The id of the selected token.
        """
        return int(np.argmax(logits))

    def display_json(self) -> None:
        """Write the collected results to ``self.output_file`` as JSON.
        """
        logging.info(f"Writing output to {self.output_file}")
        with open(self.output_file, "w") as file:
            json.dump(self.valid_json, file, indent=2)
        logging.info("Output written successfully")

    @staticmethod
    def _check_int(number: str, para_type: str) -> Union[int, float]:
        """Convert generated text to an int or a float.

        Extra signs are dropped. If the text is not a valid number, a
        warning is logged and zero is returned.

        Args:
            number: Raw generated text.
            para_type: ``"number"`` for a float, otherwise an int.

        Returns:
            The parsed value.
        """
        logging.debug(f"_check_int raw={number!r} type={para_type}")
        number = number.strip()
        if number.startswith("-"):
            number = "-" + number[1:].split("-", 1)[0]
        if number.startswith("+"):
            number = number[1:].split("+", 1)[0]
        try:
            if para_type == "number":
                return float(number)
            return int(number)
        except ValueError:
            logging.warning(f"_check_int could not parse {number!r}")
            return 0.0 if para_type == "number" else 0

    @staticmethod
    def _check_string(text: str) -> str:
        """Strip whitespace and a leading quote from a generated string.

        Args:
            text: Decoded string value.

        Returns:
            The cleaned string.
        """
        text = text.strip()
        if text.startswith("\""):
            text = text[1:]
        return text

    @staticmethod
    def _check_boolean(param: str) -> bool:
        param = param.strip()
        if param == "true" or param == "True":
            return True
        return False
