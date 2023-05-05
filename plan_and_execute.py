from prompt_manager import get_plan_loop, execute_plan_new, extract_code_from_str
from templates import TEMPLATE_DICT


def plan_and_execute(robot, description, chat_module):
    chat_module.start_session("test_run")

    first_code_run = False
    prev_code_exec = ""
    max_num_prompts = 3
    prompt_idx = 0

    user_input = input("Press enter to proceed")

    while user_input != "quit" and (prompt_idx < max_num_prompts):
        scene_description = description
        is_verbal = input("Is it a verbal query? y or n: ")
        is_verbal = (is_verbal == "y") or (is_verbal == "Y")

        task_prompt = input("Enter the task prompt: ")

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

        description = ""
        prompt_idx += 1
        user_input = input("Press enter to continue")


def plan_and_execute_feedback(robot, scene_description, task_prompt, chat_module):
    command_query = TEMPLATE_DICT["command_template"].replace("PROMPT", task_prompt)
    command = chat_module.chat(command_query)
    command = command.replace('"', "")
    command = command.lower().replace(".", "")

    plan_query = TEMPLATE_DICT["plan_template"].replace(
        "SCENE_DESCRIPTION", scene_description
    )
    plan_query = plan_query.replace("COMMAND", command.lower())

    ####### plan has been obtained #####################################################
    plan = chat_module.chat(plan_query)
    ####################################################################################

    feedback_template = (
        TEMPLATE_DICT["feedback_code_query"]
        .replace("PRIMITIVES_LST", robot.primitives_lst)
        .replace(
            "PRIMITIVES_DESCRIPTION",
            robot.primitives_description,
        )
    )

    ## TODO: Add automatic task completion
    for i in range(20):
        response = chat_module.chat(feedback_template)
        code = extract_code_from_str(response_str=response)

        


        




