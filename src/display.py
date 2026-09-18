def display_loading(file_path, result):
    print(f"📁 Loading File {file_path}")
    if not isinstance(result, list):
        print(f"❌ Failed To Load The File {file_path} For The Folowing Reason {result}\n")
    else:
        print(f"📂 File {file_path} Was Loaded Successfully\n")


def display_parsing(file_path, result):
    print(f"📄 Reading The File {file_path}")
    if not isinstance(result, list):
        print(f"❌ Parsing Error in the file {file_path} For The Folowing {result}\n")
    else:
        print(f"✅ Parsing looks clean in the {file_path}\n")

