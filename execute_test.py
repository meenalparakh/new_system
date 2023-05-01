
import numpy as np
from robot_env import MyRobot

from prompt_manager import execute_plan_new
from skill_learner import ask_for_skill



class TestExecuteRobot(MyRobot):
    def __init__(self):
        super().__init__(
            gui=False,
            grasper=False,
            magnetic_gripper=False,
            clip=False,
            meshcat_viz=False,
            device=None,
            skill_learner=True,
        )

        self.object_dicts = {
            3: {
                'pcd': np.random.rand(123, 3),
                'label': "bowl"
            },
            4: {
                'pcd': 100 + np.random.rand(451, 3),
                'label': "tray"
            }
        }
        self.robot_id = 124
        self.load_primitives()

    def place(self, obj_id, position):
        print("moving arm, printing robot val:", self.robot_id, "args:", position, obj_id)

    def no_action(self):
        print("Ending ...")

    def learn_skill(self, skill_name):

        if skill_name in self.primitives:
            print("Skill already exists. returning the existing one.")
            return self.primitives[skill_name]["fn"]

        print("Asking to learn the skill:", skill_name)
        fn = ask_for_skill(skill_name)

        def new_skill(obj_id):
            pcd = self.object_dicts[obj_id]["pcd"]
            fn(pcd)

        self.new_skill = new_skill

        self.primitives[skill_name] = {
            "fn": self.new_skill,
            "description": f"""
{skill_name}(object_id):
    performs the task of {skill_name.replace("_", " ")}
    Arguments:
        object_id: int
            Id of the object to perform the task of {skill_name.replace("_", " ")}
    Returns: None
"""}
        self.primitives_running_lst.append(skill_name)

        return self.new_skill


    def load_primitives(self):
        self.primitives = {
            "place": {
                "fn": self.place,
                "description": """
place(object_id, position)
    Moves to the position and places the object `object_id`, at the location given by `position`
    Arguments:
        object_id: int
            Id of the object to place
        position: 3D array
            the place location
    Returns: None
""",
            },

            "no_action": {
                "fn": self.no_action,
                "description": """
no_action()
    The function marks the end of a program.
    Returns: None
""",
            },
        }


        if self.skill_learner:
            self.primitives["learn_skill"] = {
                "fn": self.learn_skill,
                "description": """
learn_skill(skill_name)
    adds a new skill to the current list of skills
	Arguments:
	    skill_name: str
            a short name for the skill to learn, must be a string that can 
            represent a function name (only alphabets and underscore can be used)
	Returns:
        skill_function: method
            a function that takes as input an object_id and 
            performs the skill on the object represented by the object_id
"""
            }

        fn_lst = list(self.primitives.keys())
        self.primitives_lst = ",".join([f"`{fn_name}`" for fn_name in fn_lst])
        self.primitives_description = "\n".join(
            [self.primitives[fn]["description"] for fn in fn_lst]
        )
        self.primitives_description = "```\n" + self.primitives_description + "\n```"

        self.primitives_running_lst = list(self.primitives.keys())

        return self.primitives



def main():
    robot = TestExecuteRobot()
    # robot.load_primitives()

    code_rectified = """
```
import numpy as np
def trial_run():
    place(57, np.array([0.1, 0.2, 1.5]))
    no_action()
    place(67, np.array([0.3, 0.2, 0.1]))
    tilt_bowl = learn_skill("tilt_bowl")
    tilt_bowl(3)

    tilt_bowl(4)

```"""

    final_code = code_rectified.replace("`", "")
    task_name = "trial_run"

    code_executed = execute_plan_new(robot, task_name, final_code, prev_code_str="")

    print("starting second round of execution")

    code_rectified = """
```
import numpy as np
trial_run()

def trial_run2():
    no_action()
    tilt_bowl = learn_skill("tilt_bowl")
    tilt_bowl(4)

```"""

    final_code = code_rectified.replace("`", "")
    task_name = "trial_run2"

    execute_plan_new(robot, task_name, final_code, prev_code_str=code_executed)
    print(robot.primitives.keys())



if __name__ == "__main__":
    main()