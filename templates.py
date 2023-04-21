TEMPLATE_DICT = {
  "command_template": """Convert the following sentence into a command: "PROMPT". Do not change the verb.""",

  # ///////////////////////////////////////////////////////////////////////////////////////////////////////
  "plan_template": """SCENE_DESCRIPTION\n Tell me step by step, how to COMMAND?""",
  # ///////////////////////////////////////////////////////////////////////////////////////////////////////

  "code_template": """Convert the above steps into a python function called `TASK_NAME()`. The following functions are already implemented: 
FUNCTION_DESCRIPTIONS

Use these functions to implement `TASK_NAME()`.""",
  # ///////////////////////////////////////////////////////////////////////////////////////////////////////

  "code_template_2":"""Convert the above steps into a python function called `TASK_NAME()`. The following functions are already implemented: PRIMITIVES_LST. You can ask for new skills using the function `learn_skill`, in case the skills mentioned before are not sufficient to complete the task. Assume the relevant objects are within robot's arm's reach.
PRIMITIVES_DESCRIPTION
Write a python function `TASK_NAME()` using the above API function calls to COMMAND, and follow the steps you said previously. Assume that the above functions are already defined.
""",

  # /////////////////////////////////////////////////////////////////////////////////////////////////////////

  "code_template_3":"""Convert the above steps into a python function called `TASK_NAME()`. The following functions are already implemented: PRIMITIVES_LST. Assume the relevant objects are within robot's arm's reach.
PRIMITIVES_DESCRIPTION
Write a python function `TASK_NAME()` using the above API function calls to COMMAND, and follow the steps you said previously. Assume that the above functions are already defined. You cannot use functions other than above, or those that you define.
""",
  # ////////////////////////////////////////////////////////////////////////////////////////////////////////

  "code_rectify_template": \
"""Change the above code to ensure the following conditions hold:
1. Only picked objects can be placed. 
2. After every pick action, there should be a corresponding place action for the object, that helps to achieve the task.
3. There can be no two consecutive pick actions, and similarly no two consecutive place actions. 
4. The new program should still follow the steps given earlier as closely as possible while taking into account the above requirements.
"""
}
