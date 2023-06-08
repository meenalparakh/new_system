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
    "shapenet_objects/02876657/ed8aff7768d2cc3e45bcca2603f7a948/models/model_normalized.urdf",
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

OBJECT_AREA = np.array([[0.35, 0.7],
                        [-0.35, 0.35]])

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
    def __init__(self, robot, name, idx):
        self.robot = robot
        self.task_name = name + "_pick"
        self.obj_urdf = OBJECTS[name]["urdfs"][idx]
        self.name = name
        self.scale = OBJECTS[name]["scale"]
        self.pos_ht = OBJECTS[name]["pos_ht"]
        self.ori = OBJECTS[name]["ori"]
        self.mass = OBJECTS[name]["mass"]

    def reset(self):
        obj_id = self.robot.pb_client.load_urdf(
            ASSET_LOCATION + self.obj_urdf,
            # base_pos=[0.4, 0.28, 1.11],
            base_pos=[*(get_random_position()), self.pos_ht],
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


class ComplexPlacement:
    def __init__(self, robot, name, idx):
        self.robot = robot
        self.task_name = name + "_pick"
        self.obj_urdf = OBJECTS[name]["urdfs"][idx]
        self.name = name
        self.scale = OBJECTS[name]["scale"]
        self.pos_ht = OBJECTS[name]["pos_ht"]
        self.ori = OBJECTS[name]["ori"]
        self.mass = OBJECTS[name]["mass"]

    def reset(self):
        obj_id = self.robot.pb_client.load_urdf(
            ASSET_LOCATION + self.obj_urdf,
            base_pos=[0.4, 0.28, self.pos_ht],
            # base_pos=[*(get_random_position()), self.pos_ht],
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


def handler(signum, frame):
    msg = "Ctrl-c was pressed. Exiting."
    robot.arm.go_home()
    obj_id, current_pose = robot.predicted_pose

    cur_pos = current_pose[:3, 3]
    cur_rotation = current_pose[:3, :3]

    yaw_mat = R.from_euler('xyz', [0, 0, np.pi]).as_matrix()
    new_rotation = np.matmul(cur_rotation, yaw_mat)

    new_pose = np.eye(4)
    new_pose[:3, 3] = cur_pos
    new_pose[:3, :3] = new_rotation
    robot.pick(obj_id, grasp_pose=new_pose)

# signal.signal(signal.SIGINT, handler)

def create_scene(robot):
    pass

if __name__ == "__main__":

    robot = MyRobot(
        gui=False, 
        grasper=True, 
        clip=True, 
        meshcat_viz=True, 
        magnetic_gripper=True
    )
    # for i in range(3):

    count = 0
    total_count = 0
    receptacle = "shelf"

    for obj_name in OBJECTS:
        num_instances = len(OBJECTS[obj_name]["urdfs"])

        for idx in range(num_instances):
            print(f"Run {total_count} ..................................")
            robot.reset(ObjectsPick, obj_name, idx)
            robot.reset(ObjectsPick, receptacle, 
                        np.random.randint(0, len(OBJECTS[receptacle]["urdfs"])))

            # robot.reset(ComplexPlacement, "shelf", 0)
            for _ in range(500):
                robot.pb_client.stepSimulation()
            # robot.reset(ComplexPlacement, "tray", 0)
            # for _ in range(500):
            #     robot.pb_client.stepSimulation()
            # robot.reset(ComplexPlacement, "mug", 0)
            # for _ in range(500):
            #     robot.pb_client.stepSimulation()
          

            object_dicts = robot.get_object_dicts()
            robot.scene_description = robot.get_scene_description(object_dicts)[0]
            print(Fore.RED + robot.scene_description)
            print(Fore.BLACK)

            robot.init_dicts(object_dicts)

            obj_id = robot.find(object_label=obj_name)
            robot.pick(obj_id)

            if robot.gripper.activated:
                count += 1
                print(Fore.RED + "Pick Success")

                recept_id = robot.find(object_label=receptacle)
                position = robot.get_place_position_new(obj_id, recept_id, "farther away")
                
                robot.place(obj_id, position, skip_update=True)
            
                cam = robot.cams[1]
                rgb, _ = cam.get_images(get_rgb=True, get_depth=False, get_seg=False)
                plt.imsave(f"pick_images/{total_count}.png", rgb.astype(np.uint8))

            input("press enter to continue")
            robot.gripper.release()
            robot.remove_objects()
            total_count += 1
            print(Fore.GREEN + f"Current success rate: {count/total_count}")
            print(Fore.BLACK)


