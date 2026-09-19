import json



def encode_input(model, input_data):
    lst = []
    for data in input_data:
        lst.append(model.encode(data.prompt))
    return lst


def load_vocab(model):
    print(model.get_path_to_vocab_file())