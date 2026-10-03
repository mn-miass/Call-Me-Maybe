"""Messages printed in the terminal while the program runs."""


def display_banner() -> None:
    """Print the title of the program."""
    print("=" * 52)
    print("📞 Call Me Maybe - function calling with constrained decoding")
    print("=" * 52 + "\n")


def display_checking_file(file_path: str, result: object) -> None:
    """Print whether a file could be loaded."""
    print(f"📁 Loading File {file_path}")
    if result:
        print(
            f"❌ Failed To Load The File {file_path} "
            f"For The Following Reason: {result}\n"
        )
    else:
        print(f"📂 File {file_path} Was Loaded Successfully\n")


def display_data(file_path: str, errors: list[str]) -> None:
    """Print the validation result of one input file."""
    print(f"📄 Reading The File {file_path}")
    if errors:
        print(f"❌ Parsing Error in the file {file_path} For The Following:")
        for number, error in enumerate(errors, 1):
            print(f"   {number}. {error}")
        print()
    else:
        print(f"✅ Parsing looks clean in the {file_path}\n")


def display_model(model_name: str, seconds: float | None = None) -> None:
    """Print that the model is ready (with the loading time if given)."""
    if seconds is None:
        print(f"🤖 Model {model_name} Loaded\n")
    else:
        print(f"🤖 Model {model_name} Loaded in {seconds:.1f}s\n")


def display_model_error(model: str) -> None:
    """Print that the model could not be loaded."""
    print(f"❌ {model} Model not found or failed to download")


def display_decoding_start(total_prompts: int, total_functions: int) -> None:
    """Print what is about to be processed."""
    print(
        f"🧠 Constrained decoding: {total_prompts} prompts, "
        f"{total_functions} functions available\n"
    )


def display_summary(
    prompts: int, functions: int, elapsed: float, output_file: str
) -> None:
    """Print the final statistics of the run."""
    print(f"📊 Prompts processed : {prompts}")
    print(f"✅ Function given    : {functions}")
    print(f"⏱️  Total time        : {elapsed:.1f}s ")
    print(f"💾 Output file       : {output_file}")
    print("=" * 48)
