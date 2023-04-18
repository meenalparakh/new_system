import numpy as np
from scipy.spatial.transform import Rotation as R

ASSET_LOCATION = "../object-relations/data_new/"

class CupOverBowl:
    def __init__(self, robot):
        self.robot = robot
        self.task_name = "cup_over_bowl"

    def reset(self):
        cup_quat = R.from_euler("xyz", [np.pi / 2, 0, 0]).as_quat()
        print("Cup quat", cup_quat)
        cup_id = self.robot.pb_client.load_urdf(
            ASSET_LOCATION
            + "shapenet_objects/03797390/3d1754b7cb46c0ce5c8081810641ef6/models/model_normalized.urdf",
            base_pos=[0.3, 0.24, 1.01],
            base_ori=cup_quat,
            scaling=0.2,
            useFixedBase=True,
        )
        self.robot.pb_client.changeDynamics(cup_id, 1, mass=0.2, lateralFriction=2.0)

        bowl_id = self.robot.pb_client.load_urdf(
            ASSET_LOCATION
            + "shapenet_objects/02880940/4b32d2c623b54dd4fe296ad57d60d898/models/model_normalized.urdf",
            base_pos=[0.3, 0.19, 1.11],
            base_ori=[1, 0, 0, 1],
            scaling=0.2,
            useFixedBase=False,
        )

        cup_quat = R.from_euler("xyz", [np.pi / 2, 0, 0]).as_quat()
        print("Cup quat", cup_quat)

        basket_id = self.robot.pb_client.load_urdf(
            ASSET_LOCATION
            + "shapenet_objects/02801938/1f6046149060eb81cbde89e0c48a01bf/models/model_normalized.urdf",
            base_pos=[0.7, 0.24, 1.01],
            base_ori=cup_quat,
            scaling=0.2,
            useFixedBase=True,
        )
        self.robot.pb_client.changeDynamics(basket_id, 1, mass=0.2, lateralFriction=2.0)

        bowl_id2 = self.robot.pb_client.load_urdf(
            ASSET_LOCATION
            + "shapenet_objects/02880940/4b32d2c623b54dd4fe296ad57d60d898/models/model_normalized.urdf",
            base_pos=[0.7, 0.19, 1.11],
            base_ori=[1, 0, 0, 1],
            scaling=0.2,
            useFixedBase=False,
        )



        for i in range(4):
            self.robot.sim_dict['object_dicts'][i] = {
                "name": "unknonwn",
                "used_name": "unknown",
                "mask_id": None,
                "object_center": None,
                "pixel_center": None,
                "contains": set(),
                "edges": set(),
                "lies_over": set(),
                "lies_below": set(),
            }

        self.robot.sim_dict['object_dicts'][0]["name"] = "cup"
        self.robot.sim_dict['object_dicts'][0]["mask_id"] = cup_id
        self.robot.sim_dict['object_dicts'][1]["name"] = "bowl"
        self.robot.sim_dict['object_dicts'][1]["mask_id"] = bowl_id

        self.robot.sim_dict['object_dicts'][2]["name"] = "basket"
        self.robot.sim_dict['object_dicts'][2]["mask_id"] = basket_id
        self.robot.sim_dict['object_dicts'][3]["name"] = "bowl"
        self.robot.sim_dict['object_dicts'][3]["mask_id"] = bowl_id2


class RealTask:
    def __init__(self, robot):
        self.robot = robot

    def reset(self):
        print("Real Env")


task_lst = {"cup_over_bowl": CupOverBowl, "real_env": RealTask}
