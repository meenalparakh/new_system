from robot_env import MyRobot
import numpy as np
from scipy.spatial.transform import Rotation as R
import signal
from colorama import Fore

from matplotlib import pyplot as plt

ASSET_LOCATION = "../object-relations/data_new/"

bowls_urdf = [
    "shapenet_objects/02880940/4b32d2c623b54dd4fe296ad57d60d898/models/model_normalized.urdf",
    "shapenet_objects/02880940/2a1e9b5c0cead676b8183a4a81361b94/models/model_normalized.urdf",
    "shapenet_objects/02880940/5b6d840652f0050061d624c546a68fec/models/model_normalized.urdf",
]

mugs_urdf = [
    "shapenet_objects/03797390/1a97f3c83016abca21d0de04f408950f/models/model_normalized.urdf",
    "shapenet_objects/03797390/6e884701bfddd1f71e1138649f4c219/models/model_normalized.urdf",
    "shapenet_objects/03797390/8012f52dd0a4d2f718a93a45bf780820/models/model_normalized.urdf",
]

OBJECT_AREA = np.array([[0.35, 0.65],
                        [-0.3, 0.3]])

def get_random_position():
    x_coord = np.random.uniform(OBJECT_AREA[0][0], OBJECT_AREA[0][1])
    y_coord = np.random.uniform(OBJECT_AREA[1][0], OBJECT_AREA[1][1])
    return [x_coord, y_coord]

OBJECTS = {
    "mug": {
        "pos_ht": 1.07,
        # "pos_ht": 1.30,
        "scale": 0.2,
        "urdfs": mugs_urdf,
        "mass": 0.2,
        "ori": R.from_euler("xyz", [np.pi/2, 0, 0]).as_quat(),
    },
    "bowl": {
        "pos_ht": 1.2,
        "scale": 0.2,
        "urdfs": bowls_urdf,
        "mass": 0.2,
        "ori": R.from_euler("xyz", [np.pi/2, 0, 0]).as_quat(),
    },
}


class ObjectsPick:
    def __init__(self, robot, name, idx, random_position):
        self.robot = robot
        self.task_name = name + "_pick"
        self.obj_urdf = OBJECTS[name]["urdfs"][idx]
        self.name = name
        self.scale = OBJECTS[name]["scale"]
        self.pos_ht = OBJECTS[name]["pos_ht"]
        self.ori = OBJECTS[name]["ori"]
        self.mass = OBJECTS[name]["mass"]
        self.pos = random_position

    def reset(self):
        obj_id = self.robot.pb_client.load_urdf(
            ASSET_LOCATION + self.obj_urdf,
            # base_pos=[0.4, 0.28, 1.11],
            base_pos=[*(self.pos), self.pos_ht],
            base_ori=self.ori,
            scaling=self.scale,
            useFixedBase=False,
        )
        self.robot.pb_client.changeDynamics(obj_id, 1, mass=self.mass, lateralFriction=1.0)

        object_dict = {
            "name": self.name,
            "mask_id": obj_id
        }

        self.robot.sim_dict["object_dicts"][obj_id] = object_dict



if __name__ == "__main__":

    robot = MyRobot(
        gui=True, 
        grasper=True, 
        clip=True, 
        meshcat_viz=True, 
        magnetic_gripper=True
    )
    # for i in range(3):

    count = 0
    total_count = 0

    lower = "mug"
    upper = "bowl"

    num_lower = len(OBJECTS[lower]["urdfs"])
    num_upper = len(OBJECTS[upper]["urdfs"])

    for il in range(num_lower):
        for iu in range(num_upper):

            random_position = np.array([0.4, 0.28])

            robot.reset(ObjectsPick, lower, il, random_position)
            for _ in range(500):
                robot.pb_client.stepSimulation()

            robot.reset(ObjectsPick, upper, iu, random_position)
            for _ in range(500):
                robot.pb_client.stepSimulation()

            object_dicts = robot.get_object_dicts()
            robot.scene_description = robot.get_scene_description(object_dicts)[0]
            print(Fore.RED + robot.scene_description)
            print(Fore.BLACK)

            robot.init_dicts(object_dicts)
            robot.print_object_dicts(object_dicts)

            robot.start_task()
            
            obj_id = robot.find(object_label="bowl")
            robot.pick(obj_id)
            
            if robot.gripper.activated:
                print(Fore.RED + "Pick Bowl Success")

                position = robot.get_place_position_new(obj_id, obj_id, "to the left")
                robot.place(obj_id, position)
            
                mug_id = robot.find(object_label="mug")
                robot.pick(mug_id)
                robot.end_task()

                if robot.gripper.activated:
                    count += 1

            input("Press enter to continue")
            robot.gripper.release()
            robot.remove_objects()
            total_count += 1
            print(Fore.GREEN + f"Current success rate: {count/total_count}")
            print(Fore.BLACK)


