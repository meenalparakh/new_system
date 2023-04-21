from templates import TEMPLATE_DICT


def replace(input_str, replacements):
    for k, v in replacements.items():
        input_str = input_str.replace(k, v)

    return input_str

def extract_code_from_str(response_str, fn_name):
    
    first_occurrence = response_str.find("```")
    assert first_occurrence >= 0

    second_occurrence = response_str.find("```", first_occurrence + 3)
    assert second_occurrence >= 0

    code = response_str[first_occurrence: second_occurrence+3]

    python_occurrence = code.find("python")
    if python_occurrence >= 0:
        code = "```" + code[6:]

    elts = code.split("python\n")
    result = "".join(elts)


    first_occurence_fn = result.find(f"{fn_name}()")
    assert first_occurence_fn >= 0

    second_occurrence_fn = result.find(f"{fn_name}()", first_occurence_fn + 2)
    if second_occurrence_fn < 0:
        return result
    
    return result[:second_occurrence_fn]


def get_plan(
    scene_description,
    task_prompt,
    llm,
    function_name,
    primitives_lst,
    primitives_description,
):
    replacements = {
        "PROMPT": task_prompt,
        "SCENE_DESCRIPTION": scene_description,
        "TASK_NAME": function_name,
        "PRIMITIVES_LST": primitives_lst,
        "PRIMITIVES_DESCRIPTION": primitives_description,
    }

    command_query = TEMPLATE_DICT["command_template"].replace(
        "PROMPT", replacements["PROMPT"]
    )
    command = llm(command_query)
    command = command.replace('"', "")

    replacements["COMMAND"] = command.lower()

    plan_query = TEMPLATE_DICT["plan_template"].replace(
        "SCENE_DESCRIPTION", scene_description
    )
    plan_query = plan_query.replace("COMMAND", command.lower())
    plan = llm(plan_query)

    task_name = function_name
    print("function name: ", task_name)

    code_query = replace(TEMPLATE_DICT["code_template_3"], replacements)
    code_str = llm(code_query)

    code_rectify_query = TEMPLATE_DICT["code_rectify_template"]
    code_rectified = llm(code_rectify_query)

    return task_name, extract_code_from_str(code_rectified, function_name)


def execute_plan(robot, task_name, code_rectified):
    primitives = robot.get_primitives()
    context_str = ""
    for fn_name in primitives:
        context_str = context_str + fn_name + f" = primitives['{fn_name}']['fn']\n"

    context_str = context_str + ""

    function_string = task_name + "()"
    code_str_with_fn_call = context_str + code_rectified + "\n" + function_string

    print("Executing the following code:\n" + code_str_with_fn_call)
    exec(code_str_with_fn_call, locals())
    print("code executed")
