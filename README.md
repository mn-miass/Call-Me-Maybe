*This project has been created as part of the 42 curriculum by mn-miass.*

# Call Me Maybe

Function calling with a small LLM (Qwen3-0.6B) and **constrained decoding**.

## Description

Large language models answer in prose, but programs need structured, typed calls. This project translates a natural-language request into a function call:

```
"What is the sum of 40 and 2?"
        |
        v
{ "name": "fn_add_numbers", "parameters": {"a": 40.0, "b": 2.0} }
```

The program does **not** answer the question. It chooses which function to call and extracts the arguments with the correct types. A 0.6B-parameter model is unreliable at writing JSON when it is only prompted to do so, so the output is produced with **constrained decoding**: at every generation step, the logits of invalid tokens are set to `-inf`, so the model can only choose tokens that keep the output valid and compliant with the schema.

**Goal:** 100% valid, schema-compliant JSON, high function-selection accuracy, and all prompts processed in under 5 minutes.

### Overview of the pipeline

1. Read and validate `functions_definition.json` and `function_calling_tests.json` (pydantic).
2. For each prompt, let the LLM choose the function name (constrained).
3. Let the LLM generate each argument value (constrained by the parameter type).
4. Write all results to a JSON file with `json.dump`.

## Instructions

### Requirements

- Python 3.10 or later
- [uv](https://docs.astral.sh/uv/)
- Dependencies: `numpy`, `pydantic` (installed by `uv sync`). The `llm_sdk` package is provided in the repository.

### Installation

```bash
make install        # or: uv sync
```

### Execution

```bash
make run            # uses the default paths
# or, explicitly:
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calling_results.json
```

| Argument | Description | Default |
|---|---|---|
| `--input` | JSON array of prompts | `data/input/function_calling_tests.json` |
| `--functions_definition` | JSON array of function definitions | `data/input/functions_definition.json` |
| `--output` | Path of the result file | `output.json` |
| `--model` | Hugging Face model name | `Qwen/Qwen3-0.6B` |

The folder of the output file must already exist (for example `data/output/`).

### Makefile rules

| Rule | Action |
|---|---|
| `make install` | Install the dependencies |
| `make run` | Run the program |
| `make debug` | Run the program under `pdb` |
| `make clean` | Remove `__pycache__`, `.mypy_cache` and other caches |
| `make lint` | `flake8 .` and `mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs` |
| `make lint-strict` | `flake8 .` and `mypy . --strict` |

### Repository layout

```
.
├── Makefile
├── README.md
├── pyproject.toml
├── uv.lock
├── llm_sdk/                  # provided SDK (not modified)
├── data/input/               # example input files
└── src/
    ├── __main__.py           # entry point and pipeline
    ├── argument_parser.py    # command line arguments
    ├── parsing_files.py      # open and load the files, exit on error
    ├── parsing_data.py       # pydantic validation of prompts / functions
    ├── constrained_decoding.py   # the constrained decoding algorithm
    └── display_info.py       # terminal messages
```

A log of each run is written to `log.log`.

## Example usage

Input (`function_calling_tests.json`):

```json
[
  {"prompt": "What is the sum of 2 and 3?"},
  {"prompt": "Reverse the string 'hello'"}
]
```

Function definitions (`functions_definition.json`):

```json
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {"a": {"type": "number"}, "b": {"type": "number"}},
    "returns": {"type": "number"}
  },
  {
    "name": "fn_reverse_string",
    "description": "Reverse a string and return the reversed result.",
    "parameters": {"s": {"type": "string"}},
    "returns": {"type": "string"}
  }
]
```

Output:

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  },
  {
    "prompt": "Reverse the string 'hello'",
    "name": "fn_reverse_string",
    "parameters": {"s": "hello"}
  }
]
```

Error handling examples (the program prints a clear message and exits with status 1, without a traceback):

```bash
uv run python -m src --input missing.json      # file not found
uv run python -m src --input broken.json       # invalid JSON
```

## Algorithm explanation

### Constrained decoding in one paragraph

The model produces one logit per vocabulary token. Before choosing the next token, we compute the set of tokens that keep the output valid, add a mask that is `0` for those tokens and `-inf` for all the others, and take the `argmax`. Invalid tokens can never win, so an invalid output is impossible, whatever the model "wants" to say.

### Preparation (once, at startup)

The vocabulary file (`get_path_to_vocab_file()`) maps each token string to its id. From it we precompute lists of token ids:

| Set | Content | Used for |
|---|---|---|
| integer tokens | tokens made only of digits | integer / number values |
| sign tokens | `+`, `-` | integer / number values |
| decimal point | `.` | number values |
| string tokens | tokens whose characters are all printable (after mapping the byte-level markers `Ġ` to a space and `Ċ` to a newline) | string values |
| stop tokens | `,`, newline, `}`, `",` ... | end of a name or a number |
| name tokens | token ids obtained by encoding each function name (and `None`) | function name |

### Step 1: choosing the function (LLM + mask)

The prompt contains the instructions, the list of functions with their description and parameter types, a few examples (including requests that match no function), then the user request and `Best function name: `.

```
allowed = name_tokens + stop_tokens
loop (at most 50 tokens):
    logits = model(prompt_ids)
    logits[not in allowed] = -inf
    token = argmax(logits)
    if token in stop_tokens: break
    name += decode(token); prompt_ids.append(token)
```

The choice is made by the LLM; there are no keyword rules or other heuristics. If the result is not a known function (for example `None`), `parameters` is set to `null` and the program continues with the next prompt.

### Step 2: extracting the arguments (template + typed mask)

The JSON structure is written by the program, the model only generates the values. For each parameter, the JSON text so far is appended to an extraction prompt, for example `{"a": ` and the model continues from there:

| Type | Allowed tokens | Stop condition |
|---|---|---|
| `integer` | digits, signs, stop tokens | a stop token |
| `number` | digits, signs, decimal point, stop tokens | a stop token |
| `string` | printable tokens | a token containing the closing `"` (the text before the quote is kept) |

Every value is limited to 100 tokens. When a value is finished, it is appended to the JSON prefix, so that the next parameter is generated knowing the previous ones (`{"a": 40, "b": `).

### Step 3: cleaning and writing

- strings are unescaped with `json.loads` (so `\\d` becomes `\d`) and trimmed;
- `number` values become `float`, `integer` values become `int`; an unparsable number is logged and replaced by `0`;
- results are collected as Python dictionaries with exactly the keys `prompt`, `name` and `parameters`, and written with `json.dump`.

### Why the output is always valid JSON

The brackets, quotes and key names are never generated by the model, and the file is produced by `json.dump` from typed Python values. The only free text comes from string values, which are inserted by the serializer with correct escaping. So the file can always be parsed, and the keys and types always match the function definition.

## Design decisions

- **Template + masking instead of masking every token.** The structure of the JSON is fixed by the schema, so asking the model to produce `{`, `"` or `:` would waste forward passes on tokens that have exactly one valid choice. The model is only called where a real decision exists: which function, and what each value is. The guarantee is the same, and the program is faster.
- **Two stages (name, then arguments).** The argument prompt only contains the chosen function, which keeps the context short and avoids confusion between functions.
- **Greedy selection (`argmax`).** Deterministic and reproducible: the same input always gives the same output.
- **Constraints derived from the schema.** The token sets depend on the parameter type read from `functions_definition.json`, so changing the function set requires no code change.
- **Few-shot examples in the prompts.** They help a small model: the examples teach the answer format, and the `None` examples teach it to refuse requests that match no function.
- **Validation with pydantic.** Prompts and function definitions are checked before any model call: empty prompts, extra keys, unknown types, and function names that are Python keywords or not valid identifiers are rejected with a readable message.
- **Graceful errors.** Missing files, invalid JSON, a bad schema and a model that cannot be loaded all produce a clear message and exit status 1.
- **Logging.** Every step (prompt, top tokens, chosen token, resolved values) is written to `log.log` for debugging.

## Performance analysis

The numbers below are measured on the provided example files. Fill them in with your own run.

| Metric | Result |
|---|---|
| Number of prompts | _TODO_ |
| Total time (printed at the end of the run) | _TODO_ (limit: 5 minutes) |
| Correct function selected | _TODO_ % (target: 90%+) |
| Correct arguments | _TODO_ % |
| Valid JSON output | 100% (by construction) |

**Accuracy.** Validity is guaranteed by the constraints. Accuracy depends on the model understanding the request; it is mainly influenced by the quality of the prompts and examples.

**Speed.** Each generated token costs one forward pass of the model. Skipping the structural tokens, restricting names and numbers to a few tokens that end with a stop token, and capping every value reduce the number of passes. Most of the time is spent in the model, not in the masking (a numpy vector operation).

**Reliability.** The program never writes invalid JSON. All input errors are caught and reported, and a value that cannot be parsed produces a logged fallback and not a crash.

## Challenges faced

- **Tokens are not characters.** A single token can contain several characters (for example `",` or `"}`), so a closing quote can arrive inside a longer token. For strings, we stop as soon as a token contains `"` and keep only the text before it; the stop tokens for names and numbers were chosen with the same problem in mind.
- **Knowing when to stop.** A masked model has no way to end a value unless a terminating token is allowed. Stop tokens are therefore part of the allowed set for names and numbers, and every loop has a maximum length.
- **Small-model mistakes.** The 0.6B model sometimes picks a wrong function or invents one. The prompt lists the functions with descriptions, includes examples, and offers an explicit `None` option.
- **Context for later parameters.** After a number was generated, the next parameter's prompt was missing that value. The generated value is now added to the JSON prefix before the next parameter is generated.
- **Logits type.** The SDK returns a Python list; using `+=` with a numpy mask extends the list instead of adding to it. The logits are converted with `np.asarray` before the mask is applied.
- **Escaping.** Values such as regular expressions need escaped backslashes. The prompt shows escaped examples and string values are decoded with `json.loads`.
- **Robust input handling.** Input files can be missing, invalid or contain items that are not objects; each case is detected and reported without a traceback.

## Testing strategy

- **End-to-end runs** on the provided example files, checking the output file is created and parseable.
- **Output validation:** load the result, then verify that every entry has exactly `prompt`, `name` and `parameters`, that the name exists in the definitions, and that every argument is present with the right type.
- **Edge cases:** empty strings, large and negative numbers, decimals, special characters and quotes in strings, ambiguous prompts, requests that match no function, and functions with several parameters.
- **Invalid input:** missing files, invalid JSON, empty prompt, extra keys, unknown parameter type, function names that are keywords or invalid identifiers, and items that are not JSON objects.
- **Static checks:** `make lint` (flake8 and mypy with the flags required by the subject) must report no error.

## Limitations

- The function name is chosen from a whitelist of tokens taken from the function names; it does not track the prefix typed so far. An unknown name is detected and handled (`parameters: null`), but it is not prevented during decoding.
- Only flat arguments are supported (no nested objects or arrays). `boolean` parameters are not given a dedicated mask.
- The few-shot examples in the argument prompt are written by hand; generating them from the function definitions would make the prompt fully generic.

## Resources

### References

- Willard & Louf, *Efficient Guided Generation for Large Language Models* (2023): https://arxiv.org/abs/2307.09702
- Sennrich et al., *Neural Machine Translation of Rare Words with Subword Units* (BPE): https://arxiv.org/abs/1508.07909
- Qwen Team, *Qwen3 Technical Report*: https://arxiv.org/abs/2505.09388
- Qwen3-0.6B model card: https://huggingface.co/Qwen/Qwen3-0.6B
- Guidance (template-based constrained generation): https://github.com/guidance-ai/guidance
- llama.cpp grammars (GBNF): https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md
- OpenAI, *Function calling guide*: https://platform.openai.com/docs/guides/function-calling
- pydantic: https://docs.pydantic.dev/ · uv: https://docs.astral.sh/uv/ · flake8: https://flake8.pycqa.org/ · mypy: https://mypy.readthedocs.io/ · PEP 257: https://peps.python.org/pep-0257/

### Use of AI

AI was used as an assistant, never as a replacement for understanding:

- **Learning:** explaining constrained decoding and its variants (logit masking, grammar/FSM-based, template-based, prefix constraints) and how they relate to this subject.
- **Code review:** finding bugs in my decoding loop (wrong `if/elif` chain, type errors in the number parsing, missing values in the prompt prefix, list/array addition of logits).
- **Tooling:** fixing flake8 and mypy errors, adding docstrings and type hints, and improving error handling.
- **Documentation:** drafting this README from the subject requirements.

All the code was read, tested and discussed with peers before being kept, and I can explain every part of it.