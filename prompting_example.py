
import numpy as np
from scipy.spatial.transform import Rotation as R

from airobot import Robot
from airobot import log_warn
from airobot.utils.common import euler2quat
from prompt_manager import execute_plan, get_plan
from tasks import task_lst as TASK_LST

class MyRobot(Robot):
    def __init__(self, gui=False):
        super().__init__("franka", pb_cfg={"gui": gui})
        success = self.arm.go_home()
        if not success:
            log_warn("Robot go_home failed!!!")
        ori = euler2quat([0, 0, np.pi / 2])
        self.table_id = self.pb_client.load_urdf(
            "table/table.urdf", [0.6, 0, 0.4], ori, scaling=0.9
        )
        self.pb_client.changeDynamics(self.table_id, 0, mass=0, lateralFriction=2.0)
        self.bounds = np.array([[0.1, 1.1], [-0.6, 0.6], [0.9, 1.4]])

        self.object_dicts = {}

    def move_arm(self, position):
        current_position = self.arm.get_ee_pose()[0]
        direction = np.array(position) - current_position
        self.arm.move_ee_xyz(direction)
        return self.arm.get_ee_pose()[0]

    def no_action(self):
        print("Ending ...")

    def get_primitives(self):
        return {
            "move_arm": {
                "fn": self.move_arm,
                "description": """```
move_arm(position)
    The function moves the robotic end effector (ee) to a new 3D position, given by `position`
    Arguments:
        position: Array-like, length 3
            it represents the target position to move the ee to.
    Returns:
        new_position: Array-like, length 3
            it is the final position of the ee, after performing the move action.
```""",
            },
            "no_action": {
                "fn": self.no_action,
                "description": """```
no_action()
    The function marks the end of a program.
    Returns: None
```""",
            },
        }


if __name__ == "__main__":
    robot = MyRobot(gui=True)

    code_rectified = """
```
import numpy as np
def trial_run():
    move_arm(np.array([0.1, 0.2, 1.5]))
    no_action()
```"""

    final_code = code_rectified.replace("`", "")
    task_name = "trial_run"

    execute_plan(robot, task_name, final_code)
