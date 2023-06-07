from robot_env import MyRobot
import numpy as np
from scipy.spatial.transform import Rotation as R

ASSET_LOCATION = "../object-relations/data_new/"

mugs_urdf = [
    "shapenet_objects/02880940/4b32d2c623b54dd4fe296ad57d60d898/models/model_normalized.urdf",
    "shapenet_objects/02880940/2a1e9b5c0cead676b8183a4a81361b94/models/model_normalized.urdf",
    "shapenet_objects/02880940/5b6d840652f0050061d624c546a68fec/models/model_normalized.urdf",
]


class MugsPick:
    def __init__(self, robot, mug_urdf):
        self.robot = robot
        self.task_name = "mugs"
        self.mug_urdf = mug_urdf

    def reset(self):
        mug_id = self.robot.pb_client.load_urdf(
            ASSET_LOCATION + self.mug_urdf,
            base_pos=[0.4, 0.28, 1.11],
            base_ori=[np.pi / 2, 0, 0, 1],
            scaling=0.2,
            useFixedBase=False,
        )
        self.robot.pb_client.changeDynamics(mug_id, 1, mass=0.1, lateralFriction=1.0)

        object_dict = {
            "name": "mug",
            "mask_id": mug_id
        }

        self.robot.sim_dict["object_dicts"] = {0: object_dict}


        for _ in range(500):
            self.robot.pb_client.stepSimulation()


if __name__ == "__main__":
    robot = MyRobot(
        gui=False, 
        grasper=True, 
        clip=True, 
        meshcat_viz=True, 
        magnetic_gripper=True
    )
    robot.reset(MugsPick, mugs_urdf[0])

    object_dicts = robot.get_object_dicts()
    robot.scene_description = robot.get_scene_description(object_dicts)

    robot.init_dicts(object_dicts)

    mug_id = robot.find(object_label="mug")
    robot.pick(mug_id)

