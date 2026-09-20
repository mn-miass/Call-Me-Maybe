from llm_sdk import Small_LLM_Model
import json
import time


class ConstrainedDecoding():
    def __init__(self, model: Small_LLM_Model, input_prompt, functions):
        self.model = model
        self.input_prompt = input_prompt
        self.functions = functions
        self.encoded_prompts = self.encode_prompts()
        self.encoded_functions = self.encode_functions()
        self.valid_vocabulary = self.get_valid_vocab()
        self.swaped_vocabulary = self.swap_vocabulary()

    def encode_prompts(self):
        encoded_prompts = []
        for prompt in self.input_prompt:
            encoded_prompts.append(self.model.encode(prompt.prompt))
        return encoded_prompts

    def encode_functions(self):
        return self.model.encode(self.functions)

    def get_valid_vocab(self):
        #need to check valid_tokens
        valid_tokens = "abcdefghijklmnopqrstuvwxyz0123456789" \
        '{}[]"/\'+*_.,-!?'
        valid_model_tokens = {}
        model_vocab_file = self.model.get_path_to_vocab_file()
        with open(model_vocab_file) as file:
            model_vocab_data = json.load(file)
            with open("vocab.json", "w") as file:
                json.dump(model_vocab_data, file, indent=2)
        for key, value in model_vocab_data.items():
            if key and all(c in valid_tokens for c in key):
                valid_model_tokens[value] = key
        return valid_model_tokens

    def swap_vocabulary(self):
        swaped = {}
        for key, value in self.valid_vocabulary.items():
            swaped[value] = key
        return swaped


    def constrained_decoding(self):
        start = time.time()
        all_result = []
        for prompt in self.encoded_prompts:
            print(self.model.decode(prompt))
        print(self.model.decode(self.encoded_functions))

    # def my_decoder(self, lst_ids):
    #     s = ""
    #     for id in lst_ids:
    #         for i in id:
    #             s += self.swaped_vocabulary[i.item()]
    #     return s