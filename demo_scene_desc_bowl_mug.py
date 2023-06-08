from robot_env import MyRobot
import numpy as np
from scipy.spatial.transform import Rotation as R
import signal
from colorama import Fore
import random
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
        "scale": 0.25,
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
    position = np.array([0.5, 0.3])


    num_lower = len(OBJECTS[lower]["urdfs"])
    num_upper = len(OBJECTS[upper]["urdfs"])

    for _ in range(5):
        il = random.choice(range(num_lower))
        iu = random.choice(range(num_upper))

        robot.reset(ObjectsPick, lower, il, position)
        for _ in range(500):
            robot.pb_client.stepSimulation()

        robot.reset(ObjectsPick, upper, iu, position)
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


# ////////////////////////// No Scene Description /////////////////////////////

# def fetch_mug():
#     # Step 1: Start the task
#     start_task()

#     # Step 2: Get all objects present in the scene
#     all_object_ids = get_all_object_ids()

#     # Step 3: Identify the mug using the label description
#     mug_id = find(object_label="mug", object_ids=all_object_ids)
#     if mug_id is None:
#         print("No mug found on the table.")
#         return

#     # Step 4: Get the mug's current location
#     mug_location = get_location(mug_id)

#     # Step 5: Ensure there are no objects over the mug
#     objects_over_mug = get_objects_contained_and_over(mug_id)
#     if len(objects_over_mug) > 0:
#         print("Objects are found over the mug. Please remove them before proceeding.")
#         return

#     # Step 6: Pick the mug
#     pick(mug_id)

#     # Step 7: Define the place where the robot will move the mug. This can be a pre-defined location or another object.
#     # Here we assume that the destination_id is pre-defined and represents the desired location for the mug.
#     destination_id = get_container_id(mug_id)
#     if destination_id is None:
#         print("No destination defined for the mug.")
#         return

#     # Get the place position for the mug relative to the destination
#     place_position = get_place_position(mug_id, destination_id, "on top")

#     # Step 8: Move the mug to the designated location
#     place(mug_id, place_position)

#     # Step 9: End the task
#     end_task()

#     print("Mug fetched successfully.")


