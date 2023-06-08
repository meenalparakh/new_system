from robot_env import MyRobot
import numpy as np
from scipy.spatial.transform import Rotation as R
import signal
from colorama import Fore
import random
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
    "can": {
        "pos_ht": 1.07,
        "scale": 0.13,
        "urdfs": cans_urdf,
        "mass": 0.2,
        "ori": R.from_euler("xyz", [np.pi/2, 0, np.random.uniform(-np.pi, np.pi)]).as_quat(),
    },
    "tray": {
        # "pos_ht": 1.20,
        "pos_ht": 1.07,
        "scale": 0.5,
        "urdfs": trays_urdf,
        "mass": 2.0,
        "ori": R.from_euler("xyz", [np.pi/2, 0, np.random.uniform(-np.pi, np.pi)]).as_quat(),
    },
    "mug": {
        "pos_ht": 1.07,
        # "pos_ht": 1.30,
        "scale": 0.2,
        "urdfs": mugs_urdf,
        "mass": 0.2,
        "ori": R.from_euler("xyz", [np.pi/2, 0, np.random.uniform(-np.pi, np.pi)]).as_quat(),
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
        # self.ori = R.from_euler("xyz", [np.pi/2, 0, np.random.rand()*np.pi]).as_quat(),
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
    pos0 = np.array([0.4, -0.2])
    pos1 = np.array([0.4, 0.2])
    pos1a = np.array([0.425, 0.225])
    pos1b = np.array([0.34, 0.16])
    pos2 = np.array([0.6, -0.1])

    receptacle = "tray"
    obj1 = "mug"
    obj2 = "can"

    for _ in range(5):

        num_recp = len(OBJECTS[receptacle]["urdfs"])
        num_obj1 = len(OBJECTS[obj1]["urdfs"])
        num_obj2 = len(OBJECTS[obj2]["urdfs"])

        tray_idx1 = random.choice(range(num_recp))
        tray_idx2 = random.choice(range(num_recp))
        mug_idx = random.choice(range(num_obj2))
        can_idx1 = random.choice(range(num_obj1))
        can_idx2 = random.choice(range(num_obj1))

        err = np.random.uniform(-0.02, 0.02, size=8).reshape((4, 2))

        robot.reset(ObjectsPick, receptacle, tray_idx1, pos1)
        robot.reset(ObjectsPick, receptacle, tray_idx2, pos0 + err[0])
        robot.reset(ObjectsPick, obj2, mug_idx, pos2 + err[1])

        for _ in range(500):
            robot.pb_client.stepSimulation()

        robot.reset(ObjectsPick, obj1, can_idx1, pos1a)
        robot.reset(ObjectsPick, obj1, can_idx2, pos1b)

        for _ in range(500):
            robot.pb_client.stepSimulation()

        object_dicts = robot.get_object_dicts()
        robot.scene_description = robot.get_scene_description(object_dicts)[0]
        print(Fore.RED + robot.scene_description)
        print(Fore.BLACK)

        robot.init_dicts(object_dicts)
        robot.print_object_dicts(object_dicts)


    # # ////////////////////////////////// from scene description
    # # Start the task
        robot.start_task()
        
        # Identify the objects
        can_id = robot.find(object_label="the can")
        second_tray_id = robot.find(object_label="the second tray")
        
        # Check if there's nothing in the second tray already
        objects_in_second_tray = robot.get_objects_contained_and_over(second_tray_id)
        if len(objects_in_second_tray) > 0:
            raise ValueError("The second tray is not empty.")

        # Check if the can is not already in some container
        can_container_id = robot.get_container_id(can_id)
        if can_container_id is not None:
            raise ValueError("The can is already inside some container.")
        
        # Pick the can
        robot.pick(can_id)
        
        # Get place position in the second tray for the can
        place_position_in_second_tray = robot.get_place_position_new(can_id, second_tray_id, "inside")
        
        # Place the can inside the second tray
        robot.place(can_id, place_position_in_second_tray, skip_update=True)

        # End the task
        robot.end_task()


        robot.remove_objects()


    # # /////////////////////////// without scene description //////////////////////

    # # File "/Users/meenalp/Desktop/MEng/system_repos/new_system/demo_scene_desc_can_mug.py", line 208, in <module>
    # # tray_without_mugs_id = [id for id in tray_ids if id != mug_container_id][0]
    # # IndexError: list index out of range
    # # ////////////////////////////////////////////////////////////////////////////

    # for obj_id, info in robot.object_dicts.items():
    #    info["used_name"] = info["label"][0]

    # # Start the task
    # robot.start_task()

    # # Get the list of all object ids
    # all_object_ids = robot.get_all_object_ids()

    # # Find the ids of the mugs and trays
    # mug_ids = [robot.find(object_label="mug", object_ids=all_object_ids) for _ in range(2)]
    # tray_ids = [robot.find(object_label="tray", object_ids=all_object_ids) for _ in range(2)]

    # # Find which tray the mugs are in
    # mug_container_id = robot.get_container_id(mug_ids[0])

    # # Get the id of the tray that doesn't contain the mugs
    # tray_without_mugs_id = [id for id in tray_ids if id != mug_container_id][0]

    # # Find the can
    # can_id = robot.find(object_label="can", object_ids=all_object_ids)

    # # Pick up the can
    # robot.pick(can_id)

    # # Get the position to place the can inside the tray without mugs
    # place_position = robot.get_place_position(can_id, tray_without_mugs_id, "inside")

    # # Place the can in the tray
    # robot.place(can_id, place_position)

    # # End the task
    # robot.end_task()

    # /////////////////////////// without scene description //////////////////////
    # for obj_id, info in robot.object_dicts.items():
    #    info["used_name"] = info["label"][0]

    # # Start the task
    # robot.start_task()

    # # Get the list of all object ids
    # all_object_ids = robot.get_all_object_ids()

    # # Find the ids of the mugs and trays
    # mug_id1 = robot.find(object_label="mug", object_ids=all_object_ids)
    # tray_id1 = robot.find(object_label="tray", object_ids=all_object_ids)

    # all_object_ids.remove(mug_id1)
    # all_object_ids.remove(tray_id1)

    # mug_id2 = robot.find(object_label="mug", object_ids=all_object_ids)
    # tray_id2 = robot.find(object_label="tray", object_ids=all_object_ids)

    # mug_ids = [mug_id1, mug_id2]
    # tray_ids = [tray_id1, tray_id2]

    # # Find which tray the mugs are in
    # mug_container_id = robot.get_container_id(mug_ids[0])

    # # Get the id of the tray that doesn't contain the mugs
    # tray_without_mugs_id = [id for id in tray_ids if id != mug_container_id][0]

    # # Find the can
    # can_id = robot.find(object_label="can", object_ids=all_object_ids)

    # # Pick up the can
    # robot.pick(can_id)

    # # Get the position to place the can inside the tray without mugs
    # place_position = robot.get_place_position_new(can_id, tray_without_mugs_id, "inside")

    # # Place the can in the tray
    # robot.place(can_id, place_position, skip_update=True)

    # # End the task
    # robot.end_task()
    # ////////////////////////////////////////////////////////////////////////////


