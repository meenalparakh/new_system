TEMPLATE_DICT = {
  "command_template": """Convert the following sentence into a command: "PROMPT". Do not change the verb.""",

  "plan_template": """SCENE_DESCRIPTION How can we COMMAND? Answer in steps.""",
  
  "code_template": """Convert the above steps into a python function called `TASK_NAME()`. The following functions are already implemented: 
FUNCTION_DESCRIPTIONS

Use these functions to implement `TASK_NAME()`.""",

  "code_rectify_template": \
"""Change the program to ensure the following conditions hold:
1. Only picked objects can be placed. 
2. After every pick action, there should be a corresponding place action for the object, that helps to achieve the task.
3. There can be no two consecutive pick actions, and similarly no two consecutive place actions. 
4. The new program should still follow the steps given earlier as closely as possible while taking into account the above requirements.
"""
}
