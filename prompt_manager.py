from templates import TEMPLATE_DICT

def get_plan(scene_description, task_prompt, llm, function_name, primitives):
    command_query = TEMPLATE_DICT["command_template"].replace("PROMPT", task_prompt)
    command = llm(command_query)
    command = command.replace('"', "")

    plan_query = TEMPLATE_DICT["plan_template"].replace(
        "SCENE_DESCRIPTION", scene_description
    )
    plan_query = plan_query.replace("COMMAND", command.lower())
    plan = llm(plan_query)

    task_name = function_name
    print("function name: ", task_name)

    code_query = TEMPLATE_DICT["code_template"].replace("TASK_NAME", task_name)
    code_query = code_query.replace("FUNCTION_DESCRIPTIONS", primitives)
    code_str = llm(code_query)

    code_rectify_query = TEMPLATE_DICT["code_rectify_template"]
    code_rectified = llm(code_rectify_query)

    return task_name, code_rectified


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

