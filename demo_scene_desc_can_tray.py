from robot_env import MyRobot
import numpy as np
from scipy.spatial.transform import Rotation as R
import signal
from colorama import Fore

from matplotlib import pyplot as plt

ASSET_LOCATION = "../object-relations/data_new/"

trays_urdf = [
    "shapenet_objects/02801938/d224635923b9ec4637dc91749a7c4915/models/model_normalized.urdf",
    "shapenet_objects/02801938/a8943da1522d056650da8b99982a3057/models/model_normalized.urdf",
    "shapenet_objects/02801938/e57f5dcfc6c4185dc2a6a08aa01a9e9/models/model_normalized.urdf",
    "shapenet_objects/02801938/2ae75c0f4bf43142e76bc197b3a3ffc0/models/model_normalized.urdf",
]

cans_urdf = [
    "shapenet_objects/02946921/10c9a321485711a88051229d056d81db/models/model_normalized.urdf",
    "shapenet_objects/02946921/203c5e929d588d07c6754428123c8a7b/models/model_normalized.urdf",
    "shapenet_objects/02946921/7b643c8136a720d9db4a36333be9155/models/model_normalized.urdf",
]

OBJECT_AREA = np.array([[0.35, 0.65],
                        [-0.3, 0.3]])

def get_random_position():
    x_coord = np.random.uniform(OBJECT_AREA[0][0], OBJECT_AREA[0][1])
    y_coord = np.random.uniform(OBJECT_AREA[1][0], OBJECT_AREA[1][1])
    return [x_coord, y_coord]

OBJECTS = {
    "can": {
        "pos_ht": 1.07,
        "scale": 0.13,
        "urdfs": cans_urdf,
        "mass": 0.2,
        "ori": R.from_euler("xyz", [np.pi/2, 0, np.random.uniform(0, 2*np.pi)]).as_quat(),
    },
    "tray": {
        # "pos_ht": 1.20,
        "pos_ht": 1.07,
        "scale": 0.4,
        "urdfs": trays_urdf,
        "mass": 2.0,
        "ori": R.from_euler("xyz", [np.pi/2, 0, np.random.uniform(0, 2*np.pi)]).as_quat(),
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
        # self.ori = OBJECTS[name]["ori"]
        self.ori = R.from_euler("xyz", [np.pi/2, 0, np.random.uniform(0, 2*np.pi)]).as_quat(),
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

    receptacle = "tray"
    obj = "can"

    num_recp = len(OBJECTS[receptacle]["urdfs"])
    num_obj = len(OBJECTS[obj]["urdfs"])

    for im in range(num_recp):
        for il in range(num_recp):
            for iu in range(num_obj):
                for ik in range(num_obj):

                    pos0 = np.array([0.4, -0.2])
                    pos1 = np.array([0.4, 0.2])
                    pos2 = np.array([0.6, -0.1])

                    err = np.random.uniform(-0.02, 0.02, size=6).reshape((3, 2))

                    robot.reset(ObjectsPick, receptacle, im, pos0 + err[0])
                    robot.reset(ObjectsPick, receptacle, il, pos1 + err[1])
                    robot.reset(ObjectsPick, obj, iu, pos2 + err[2])

                    for _ in range(500):
                        robot.pb_client.stepSimulation()

                    robot.reset(ObjectsPick, obj, ik, pos1 + err[1])
                    for _ in range(500):
                        robot.pb_client.stepSimulation()

                    object_dicts = robot.get_object_dicts()
                    robot.scene_description = robot.get_scene_description(object_dicts)[0]
                    print(Fore.RED + robot.scene_description)
                    print(Fore.BLACK)

                    robot.init_dicts(object_dicts)
                    robot.print_object_dicts(object_dicts)



                    robot.start_task()

                    # Step 1: Identify all objects
                    all_object_ids = robot.get_all_object_ids()

                    # Step 2: Identify the first tray and the can it contains
                    first_tray_id = robot.find(object_label="the first tray", object_ids=all_object_ids)
                    objects_in_first_tray = robot.get_objects_contained_and_over(first_tray_id)

                    # Check if the first tray already contains a can
                    for obj_id in objects_in_first_tray:
                        if robot.get_container_id(obj_id) == first_tray_id:
                            first_can_id = obj_id
                            break

                    # Step 3: Identify the second can
                    second_can_id = robot.find(object_label="the second can", object_ids=all_object_ids)

                    # Step 4: Pick the second can
                    robot.pick(second_can_id)

                    # Step 5: Identify the second tray
                    second_tray_id = robot.find(object_label="the second tray", object_ids=all_object_ids)

                    # Step 6: Get the place position in the second tray
                    place_position = robot.get_place_position_new(second_can_id, second_tray_id, "inside")

                    # Step 7: Place the second can in the second tray
                    robot.place(second_can_id, place_position)

                    robot.end_task()


                    robot.remove_objects()


