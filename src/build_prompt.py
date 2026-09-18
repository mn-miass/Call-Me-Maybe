def build_prompt(function_calling):

    prompt = [
        "Much only From The Folowing Function Never use UNRELATED FUNCTIONS \n"
    ]
    available_functions = []

    for fn in function_calling:
        parms = ", ".join(
            f"{key}: {value.type.value}" for key, value in fn.parameters.items()
        )
        available_functions.append(
            f"- {fn.name}({parms}): ({fn.description}) return_type: ({fn.returns.type.value})"
        )

    output_form = '\nOutput Only Valid Json: {"name": "<function_name>", "args": {<args>}\n'
    prompt.extend(available_functions)
    prompt.append(output_form)

    return "\n".join(prompt)