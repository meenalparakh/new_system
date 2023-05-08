from templates import TEMPLATE_DICT


def replace(input_str, replacements):
    for k, v in replacements.items():
        input_str = input_str.replace(k, v)

    return input_str


def extract_code_from_str(response_str, fn_name=None):
    first_occurrence = response_str.find("```")
    assert first_occurrence >= 0

    total_code = ""

    while first_occurrence >= 0:
        second_occurrence = response_str.find("```", first_occurrence + 3)
        assert second_occurrence >= 0

        code = response_str[first_occurrence + 3 : second_occurrence]

        python_occurrence = code.find("python")
        if python_occurrence >= 0:
            code = "```" + code[6:]

        elts = code.split("python\n")
        result = "".join(elts)

        total_code += result + "\n"

        response_str = response_str[second_occurrence + 3 :]
        first_occurrence = response_str.find("```")

    total_code = "```" + total_code + "```"

    if fn_name is None:
        return total_code

    first_occurence_fn = total_code.find(f"{fn_name}()")
    assert first_occurence_fn >= 0

    second_occurrence_fn = total_code.find(f"{fn_name}()", first_occurence_fn + 2)
    if second_occurrence_fn < 0:
        return result

    return total_code[:second_occurrence_fn] + "```"


def test_extract_code():
    test_str = """    
First, we need to find the mug and tray:
```
start_task()

mug_id = find("mug", "a ceramic coffee mug")
tray_id = find("tray", "a rectangular serving tray")
```
We have used the find function to locate the mug and tray and stored their IDs for later use. Next, we will pick up the mug:
```
pick(mug_id)
```
After executing the above code, we need to confirm that the robot has picked up the mug. Once confirmed, we will move onto the next step where we find the location to place the mug in the tray.

"""
    result = extract_code_from_str(test_str)
    print(result)


def get_plan(
    scene_description,
    task_prompt,
    llm,
    function_name,
    primitives_lst,
    primitives_description,
    code_rectification=False,
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

    code_query = replace(TEMPLATE_DICT["code_template"], replacements)
    code_str = llm(code_query)

    if code_rectification:
        code_rectify_query = TEMPLATE_DICT["code_rectify_template"]
        code_rectified = llm(code_rectify_query)
    else:
        code_rectified = code_str

    return task_name, extract_code_from_str(code_rectified, function_name)


def get_plan_loop(
    scene_description,
    task_prompt,
    llm,
    function_name,
    primitives_lst,
    primitives_description,
    code_rectification=False,
    first_run=True,
    verbal_query=False,
    ask_plan=False,
):
    if not verbal_query:
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

        replacements["COMMAND"] = command.lower()[:-1]

        if first_run:
            plan_query = TEMPLATE_DICT["plan_template"].replace(
                "SCENE_DESCRIPTION", scene_description
            )
            plan_query = plan_query.replace("COMMAND", command.lower())
            plan = llm(plan_query)

            task_name = function_name
            print("function name: ", task_name)

            code_query = replace(TEMPLATE_DICT["code_template_3"], replacements)
            code_str = llm(code_query)

            if code_rectification:
                code_rectify_query = TEMPLATE_DICT["code_rectify_template"]
                code_rectified = llm(code_rectify_query)
            else:
                code_rectified = code_str

            return task_name, extract_code_from_str(code_rectified, function_name)

        else:
            continued_code_query = replace(
                TEMPLATE_DICT["continued_tasks"],
                {
                    "SCENE_CHANGES": replacements["SCENE_DESCRIPTION"],
                    "COMMAND": replacements["COMMAND"],
                    "TASK_NAME": replacements["TASK_NAME"],
                    "PRIMITIVES_LST": replacements["PRIMITIVES_LST"],
                },
            )
            continued_code = llm(continued_code_query)

            return function_name, extract_code_from_str(continued_code, function_name)

    elif verbal_query:
        verbal_query_template = TEMPLATE_DICT["verbal_query"].replace(
            "SCENE_DESCRIPTION", scene_description
        )
        verbal_query_template = verbal_query_template.replace("QUERY", task_prompt)

        response = llm(verbal_query_template)
        return response


def execute_plan(robot, task_name, code_rectified):
    # primitives = robot.get_primitives()
    context_str = ""
    for fn_name in robot.primitives:
        context_str = (
            context_str + fn_name + f" = robot.primitives['{fn_name}']['fn']\n"
        )

    context_str = context_str + ""

    function_string = task_name + "()"
    code_str_with_fn_call = context_str + code_rectified + "\n" + function_string

    print("Executing the following code:\n" + code_str_with_fn_call)
    exec(code_str_with_fn_call, locals())
    print("code executed")


def execute_plan_new(robot, task_name, code_rectified, prev_code_str=""):
    # primitives = robot.get_primitives()
    context_str = prev_code_str + "\n"

    for fn_name in robot.primitives_running_lst:
        context_str = (
            context_str + fn_name + f" = robot.primitives['{fn_name}']['fn']\n"
        )

    robot.primitives_running_lst = []

    context_str = context_str + ""

    function_string = task_name + "()"
    code_str_with_fn_call = context_str + code_rectified + "\n" + function_string

    print("Executing the following code:\n" + code_str_with_fn_call)
    exec(code_str_with_fn_call, locals())
    print("code executed")

    return context_str + code_rectified


def plan_and_execute(robot, description, chat_module, chat_session="test_run"):
    chat_module.start_session(chat_session)

    first_code_run = False
    prev_code_exec = ""
    max_num_prompts = 3
    prompt_idx = 0

    user_input = input("Press enter to proceed")

    object_dicts = robot.get_object_dicts()
    scene_description, object_dicts = robot.get_scene_description(object_dicts)
    robot.init(object_dicts)
    robot.print_object_dicts(robot.object_dicts)

    while user_input != "quit" and (prompt_idx < max_num_prompts):
        is_verbal = input("Do you want to make a verbal query? y or n: ")
        is_verbal = (is_verbal == "y") or (is_verbal == "Y")

        task_prompt = input("Enter the prompt: ")

        if is_verbal:
            task_name = ""
        else:
            task_name = input("Enter task name: ")

        response = get_plan_loop(
            scene_description,
            task_prompt,
            chat_module.chat,
            task_name,
            robot.primitives_lst,
            robot.primitives_description,
            code_rectification=first_code_run,
            first_run=first_code_run,
            verbal_query=is_verbal,
        )

        if is_verbal:
            print(
                " FINAL ANSWER >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
            )
            print(response)
            print(
                " >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
            )
        else:
            print(
                " FINAL ANSWER >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
            )
            task_name, code_str = response
            print("TASK NAME:", task_name)
            print("CODE:", code_str)
            print(
                " >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
            )

            final_code = code_str.replace("`", "")

            prev_code_exec = execute_plan_new(
                robot, task_name, final_code, prev_code_str=prev_code_exec
            )
            print(robot.primitives.keys())

        if not is_verbal:
            first_code_run = False

            scene_description = robot.get_current_scene_description()
            robot.print_object_dicts(robot.object_dicts)

        prompt_idx += 1
        user_input = input("Press enter to continue")


if __name__ == "__main__":
    robot_type = "sim"

    if robot_type == "sim":
        from robot_env import MyRobot

        robot = MyRobot(
            gui=True,
            grasper=True,
            magnetic_gripper=True,
            clip=True,
            meshcat_viz=True,
            device="cpu",
            skill_learner=True,
        )

        robot.reset("bowl_over_cup")

    elif robot_type == "real":
        from real_env import RealRobot
        robot = RealRobot(gui=False, realsense_cams=True, sam=True, clip=True, grasper=True, cam_idx=[2, 3], device="cpu")


        