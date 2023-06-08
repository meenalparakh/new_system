from robot_env import MyRobot
import numpy as np
from scipy.spatial.transform import Rotation as R
import signal
from colorama import Fore

from matplotlib import pyplot as plt
import random

ASSET_LOCATION = "../object-relations/data_new/"

bowls_urdf = [
    "shapenet_objects/02880940/4b32d2c623b54dd4fe296ad57d60d898/models/model_normalized.urdf",
    "shapenet_objects/02880940/2a1e9b5c0cead676b8183a4a81361b94/models/model_normalized.urdf",
    "shapenet_objects/02880940/5b6d840652f0050061d624c546a68fec/models/model_normalized.urdf",
]

mugs_urdf = [
    # "shapenet_objects/03797390/1a97f3c83016abca21d0de04f408950f/models/model_normalized.urdf",
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
    "shapenet_objects/02876657/ed8aff7768d2cc3e45bcca2603f7a948/models/model_normalized.urdf",
]

trays_urdf = [
    "shapenet_objects/02801938/d224635923b9ec4637dc91749a7c4915/models/model_normalized.urdf",
    # "shapenet_objects/02801938/a8943da1522d056650da8b99982a3057/models/model_normalized.urdf",
    "shapenet_objects/02801938/e57f5dcfc6c4185dc2a6a08aa01a9e9/models/model_normalized.urdf",
    "shapenet_objects/02801938/2ae75c0f4bf43142e76bc197b3a3ffc0/models/model_normalized.urdf",
]
shelfs_urdf = [
    "shapenet_objects/02871439/fbc6b4e0aa9cc13d713e7f5d7ea85661/models/model_normalized.urdf",
]

OBJECT_AREA = np.array([[0.4, 0.6],
                        [-0.3, 0.3]])
SAME = False
POS = None

def get_random_position():
    x_coord = np.random.uniform(OBJECT_AREA[0][0], OBJECT_AREA[0][1])
    y_coord = np.random.uniform(OBJECT_AREA[1][0], OBJECT_AREA[1][1])
    return [x_coord, y_coord]

OBJECTS = {
    "can": {
        "pos_ht": 1.2,
        "scale": 0.15,
        "urdfs": cans_urdf,
        "mass": 0.2,
        "ori": R.from_euler("xyz", [np.pi/2, 0, 0]).as_quat(),
    },
    "mug": {
        "pos_ht": 1.2,
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
    "tray": {
        # "pos_ht": 1.20,
        "pos_ht": 1.07,
        "scale": 0.4,
        "urdfs": trays_urdf,
        "mass": 2.0,
        "ori": R.from_euler("xyz", [np.pi/2, 0, 0]).as_quat(),
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
    def __init__(self, robot, o, recap, im, il, iu, ik):
        self.robot = robot

        self.obj_urdf = OBJECTS[o]["urdfs"][iu]
        self.name = o
        self.scale = OBJECTS[o]["scale"]
        self.pos_ht = 1.05
        self.ori = OBJECTS[o]["ori"]
        self.mass = OBJECTS[o]["mass"]

        self.obj_urdf1 = OBJECTS[o]["urdfs"][ik]
        self.name1 = o
        self.scale1 = OBJECTS[o]["scale"]
        self.pos_ht1 = 1.15
        self.ori1 = OBJECTS[o]["ori"]
        self.mass1 = OBJECTS[o]["mass"]

        self.obj_urdf2 = OBJECTS[recap]["urdfs"][im]
        self.name2 = recap
        self.scale2 = OBJECTS[recap]["scale"]
        self.pos_ht2 = 1.15
        self.ori2 = OBJECTS[recap]["ori"]
        self.mass2 = OBJECTS[recap]["mass"]

        self.obj_urdf3 = OBJECTS[recap]["urdfs"][il]
        self.name3 = recap
        self.scale3 = OBJECTS[recap]["scale"]
        self.pos_ht3 = 1.15
        self.ori3 = OBJECTS[recap]["ori"]
        self.mass3 = OBJECTS[recap]["mass"]

    def reset(self):
        pos0 = np.array([0.4, -0.2])
        pos1 = np.array([0.4, 0.2])
        pos2 = np.array([0.6, -0.1])
        err = np.random.uniform(-0.02, 0.02, size=6).reshape((3, 2))


        obj_id = self.robot.pb_client.load_urdf(
            ASSET_LOCATION + self.obj_urdf2,
            # base_pos=[0.4, 0.28, 1.11],
            base_pos=[*(pos0 + err[0]), self.pos_ht],
            base_ori=self.ori2,
            scaling=self.scale2,
            useFixedBase=False,
        )
        self.robot.pb_client.changeDynamics(obj_id, 1, mass=self.mass2, lateralFriction=1.0)
        object_dict = {
            "name": self.name2,
            "mask_id": obj_id
        }
        self.robot.sim_dict["object_dicts"][obj_id] = object_dict


        obj_id = self.robot.pb_client.load_urdf(
            ASSET_LOCATION + self.obj_urdf3,
            # base_pos=[0.4, 0.28, 1.11],
            base_pos=[*(pos1 + err[1]), self.pos_ht3],
            base_ori=self.ori3,
            scaling=self.scale3,
            useFixedBase=False,
        )
        self.robot.pb_client.changeDynamics(obj_id, 1, mass=self.mass3, lateralFriction=1.0)
        object_dict = {
            "name": self.name3,
            "mask_id": obj_id
        }
        self.robot.sim_dict["object_dicts"][obj_id] = object_dict

        obj_id = self.robot.pb_client.load_urdf(
            ASSET_LOCATION + self.obj_urdf,
            # base_pos=[0.4, 0.28, 1.11],
            base_pos=[*(pos2 + err[2]), self.pos_ht1],
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

        obj_id = self.robot.pb_client.load_urdf(
            ASSET_LOCATION + self.obj_urdf1,
            # base_pos=[0.4, 0.28, 1.11],
            base_pos=[*(pos1 + err[1]), self.pos_ht1],
            base_ori=self.ori1,
            scaling=self.scale1,
            useFixedBase=False,
        )
        self.robot.pb_client.changeDynamics(obj_id, 1, mass=self.mass1, lateralFriction=1.0)
        object_dict = {
            "name": self.name1,
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

    receptacle = ["tray", "shelf"]
    obj = ["bowl", "mug", "can"]

    position = np.array([0.5, 0.3])

    for recap in receptacle:
        for o in obj:

            num_o = len(OBJECTS[o]["urdfs"])
            num_recap = len(OBJECTS[recap]["urdfs"])

            im = random.choice(range((num_recap)))
            il = random.choice(range((num_recap)))
            iu = random.choice(range((num_o)))
            ik = random.choice(range((num_o)))

            print(f"Run {total_count} ..................................")

            robot.reset(ObjectsPick, o, recap, im, il, iu, ik)

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
            first_tray_id = robot.find(object_label="the first "+recap, object_ids=all_object_ids)
            objects_in_first_tray = robot.get_objects_contained_and_over(first_tray_id)

            # Check if the first tray already contains a can
            for obj_id in objects_in_first_tray:
                if robot.get_container_id(obj_id) == first_tray_id:
                    first_can_id = obj_id
                    break

            # Step 3: Identify the second can
            second_can_id = robot.find(object_label="the second "+ o, object_ids=all_object_ids)

            # Step 4: Pick the second can
            robot.pick(second_can_id)

            # Step 5: Identify the second tray
            second_tray_id = robot.find(object_label="the second "+recap, object_ids=all_object_ids)

            # Step 6: Get the place position in the second tray
            place_position = robot.get_place_position_new(second_can_id, second_tray_id, "over")

            # Step 7: Place the second can in the second tray
            robot.place(second_can_id, place_position, skip_update=True)

            robot.end_task()



            cam = robot.cams[1]
            rgb, _ = cam.get_images(get_rgb=True, get_depth=False, get_seg=False)
            plt.imsave(f"pick_images/{total_count}.png", rgb.astype(np.uint8))

            input("Press enter to continue")
            robot.gripper.release()
            robot.remove_objects()
            total_count += 1
            print(Fore.GREEN + f"Current success rate: {count/total_count}")
            print(Fore.BLACK)

