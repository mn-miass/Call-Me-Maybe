def display_checking_file(file_path, result):
    print(f"📁 Loading File {file_path}")
    if result:
        print(f"❌ Failed To Load The File {file_path} For The Folowing Reason {result}\n")
    else:
        print(f"📂 File {file_path} Was Loaded Successfully\n")

def display_data(file_path, errors):
    print(f"📄 Reading The File {file_path}")
    if errors:
        print(f"❌ Parsing Error in the file {file_path} For The Folowing")
        for error in errors:
            print(error)
    else:
        print(f"✅ Parsing looks clean in the {file_path}\n")

def display_model(model_name):
    print(f"🤖 Loading Model {model_name}")


def display_model_error(model):
    print(f"❌ {model} Model not found or failed to download")


def display_building_prompt():
    print("⚙️ Building system prompt ...")