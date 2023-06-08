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

cans_urdf = [
    "shapenet_objects/02946921/10c9a321485711a88051229d056d81db/models/model_normalized.urdf",
    "shapenet_objects/02946921/203c5e929d588d07c6754428123c8a7b/models/model_normalized.urdf",
    "shapenet_objects/02946921/7b643c8136a720d9db4a36333be9155/models/model_normalized.urdf",
]

bottles_urdf = [
    "shapenet_objects/02876657/e017cd91e485756aef026123226f5519/models/model_normalized.urdf",
    "shapenet_objects/02876657/1ffd7113492d375593202bf99dddc268/models/model_normalized.urdf",
    "shapenet_objects/02876657/1ae823260851f7d9ea600d1a6d9f6e07/models/model_normalized.urdf"
]

trays_urdf = [
    "shapenet_objects/02801938/d224635923b9ec4637dc91749a7c4915/models/model_normalized.urdf",
    "shapenet_objects/02801938/a8943da1522d056650da8b99982a3057/models/model_normalized.urdf",
    "shapenet_objects/02801938/e57f5dcfc6c4185dc2a6a08aa01a9e9/models/model_normalized.urdf",
    "shapenet_objects/02801938/2ae75c0f4bf43142e76bc197b3a3ffc0/models/model_normalized.urdf",
]
shelfs_urdf = [
    "shapenet_objects/02871439/fbc6b4e0aa9cc13d713e7f5d7ea85661/models/model_normalized.urdf",
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
        "scale": 0.15,
        "urdfs": cans_urdf,
        "mass": 0.2,
        "ori": R.from_euler("xyz", [np.pi/2, 0, 0]).as_quat(),
    },
    "mug": {
        "pos_ht": 1.07,
        # "pos_ht": 1.30,
        "scale": 0.2,
        "urdfs": mugs_urdf,
        "mass": 0.2,
        "ori": R.from_euler("xyz", [np.pi/2, 0, 0]).as_quat(),
    },
    "bottle": {
        "pos_ht": 1.07,
        # "pos_ht": 1.30,
        "scale": 0.2,
        "urdfs": bottles_urdf,
        "mass": 0.2,
        "ori": R.from_euler("xyz", [np.pi/2, 0, 0]).as_quat(),
    },
    "bowl": {
        "pos_ht": 1.07,
        "scale": 0.2,
        "urdfs": bowls_urdf,
        "mass": 0.2,
        "ori": R.from_euler("xyz", [np.pi/2, 0, 0]).as_quat(),
    },
    "tray": {
        # "pos_ht": 1.20,
        "pos_ht": 1.07,
        "scale": 0.4,
        "urdfs": trays_urdf,
        "mass": 2.0,
        "ori": R.from_euler("xyz", [np.pi/2, 0, np.pi/2]).as_quat(),
    },
    "shelf": {
        "pos_ht": 1.10,
        "scale": 0.5,
        "urdfs": shelfs_urdf,
        "mass": 2.0,
        "ori": R.from_euler("xyz", [np.pi/2, 0, 0]).as_quat(),
    },
}


class ObjectsPick:
    def __init__(self, robot, name, idx , obj_loc, ht = None):
        self.robot = robot
        self.task_name = name + "_pick"
        self.obj_urdf = OBJECTS[name]["urdfs"][idx]
        self.name = name
        self.scale = OBJECTS[name]["scale"]
        if isinstance(ht, type(None)):
            self.pos_ht = OBJECTS[name]["pos_ht"]
        else:
            self.pos_ht = ht
        self.ori = OBJECTS[name]["ori"]
        self.mass = OBJECTS[name]["mass"]
        self.obj_loc = obj_loc

    def reset(self):
        obj_id = self.robot.pb_client.load_urdf(
            ASSET_LOCATION + self.obj_urdf,
            # base_pos=[0.4, 0.28, 1.11],
            base_pos=[*(self.obj_loc), self.pos_ht],
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

    np.random.seed(5)
    robot = MyRobot(
        gui=True, 
        grasper=True, 
        clip=True, 
        meshcat_viz=True, 
        magnetic_gripper=True
    )
    # for i in range(3):

    pos_1 = [0.45,-0.15]
    pos_2 = [0.65,-0.1]
    pos_3 = [0.6,-0.1]
    pos_4 = [0.45,-0.25]
    pos_5 = [0.55,-0.15]
    res_1 = [0.65, 0.3]
    res_2 = [0.35, 0.3]

    robot.reset(ObjectsPick, "bottle", 0, pos_1)
    # robot.reset(ObjectsPick, "bottle", 1, pos_2)
    robot.reset(ObjectsPick, "bottle", 2, pos_3, 1.09)
    robot.reset(ObjectsPick, "can", 0, pos_4)
    # robot.reset(ObjectsPick, "can", 1, pos_5)
    receptacle_1 = "tray"
    robot.reset(ObjectsPick, receptacle_1,1,res_1)
    robot.reset(ObjectsPick, receptacle_1,3,res_2)

    
    for _ in range(500):
        robot.pb_client.stepSimulation()
    

    object_dicts = robot.get_object_dicts()
    robot.scene_description = robot.get_scene_description(object_dicts)[0]
    print(Fore.RED + robot.scene_description)
    print(Fore.BLACK)

    robot.init_dicts(object_dicts)
    robot.print_object_dicts(object_dicts)
    robot.visualize_object_dicts(object_dicts)

    first_tray_id = robot.find(object_label="the first tray")

    # find the id of the second tray
    second_tray_id = robot.find(object_label="the second tray")

    # find the ids of the bottles
    first_bottle_id = robot.find(object_label="the first bottle")
    second_bottle_id = robot.find(object_label="the second bottle")
    # third_bottle_id = robot.find(object_label="the third bottle")

    # find the ids of the cans
    first_can_id = robot.find(object_label="the first can")
    # second_can_id = robot.find(object_label="the second can")
    


    robot.pick(first_bottle_id, visualize = True)

    if robot.gripper.activated:
    
        place_position = robot.get_place_position(first_bottle_id, first_tray_id, "inside")
        robot.place(first_bottle_id, place_position)




    robot.pick(second_bottle_id, visualize = True)

    if robot.gripper.activated:
    
        place_position = robot.get_place_position(second_bottle_id, first_tray_id, "inside")
        robot.place(second_bottle_id, place_position)


    robot.pick(first_can_id, visualize = True)

    if robot.gripper.activated:
    
        place_position = robot.get_place_position(first_can_id, second_tray_id, "inside")
        robot.place(first_can_id, place_position)
   
   
        

