import json



def load_file(input_path):
    try:
        with open(input_path) as file:
            data = json.load(file)
            return data
    except json.decoder.JSONDecodeError as e:
         return e.args
    except Exception as e:
        return e.args







    