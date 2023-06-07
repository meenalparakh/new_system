from robot_env import MyRobot, print_object_dicts
import numpy as np
from scipy.spatial.transform import Rotation as R

ASSET_LOCATION = "../object-relations/data_new/"


class BowlsInBin:
    def __init__(self, robot):
        self.robot = robot
        self.task_name = "bowls_in_bin"

    def reset(self):

        apple_id = self.robot.pb_client.load_urdf(
            ASSET_LOCATION
            + "shapenet_objects/02880940/2a1e9b5c0cead676b8183a4a81361b94/models/model_normalized.urdf",
            base_pos=[0.4, 0.28, 1.11],
            base_ori=[np.pi / 2, 0, 0, 1],
            scaling=0.2,
            useFixedBase=False,
        )
        self.robot.pb_client.changeDynamics(apple_id, 1, mass=0.1, lateralFriction=10.0)

        tray_quat = R.from_euler("xyz", [np.pi / 2, 0, 0]).as_quat()
        print("tray quat", tray_quat)

        tray_id = self.robot.pb_client.load_urdf(
            ASSET_LOCATION
            + "shapenet_objects/02801938/3ebac03588e8a43450da8b99982a3057/models/model_normalized.urdf",
            base_pos=[0.7, -0.34, 1.01],
            base_ori=tray_quat,
            scaling=0.5,
            useFixedBase=True,
        )
        self.robot.pb_client.changeDynamics(tray_id, 1, mass=0.2, lateralFriction=2.0)

        for i in range(2):
            self.robot.sim_dict["object_dicts"][i] = {
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


        # for i in range(3):
        self.robot.sim_dict["object_dicts"][0]["name"] = "apple"
        self.robot.sim_dict["object_dicts"][0]["mask_id"] = apple_id

        self.robot.sim_dict["object_dicts"][1]["name"] = "tray"
        self.robot.sim_dict["object_dicts"][1]["mask_id"] = tray_id

        for _ in range(500):
            self.robot.pb_client.stepSimulation()


if __name__ == "__main__":
    robot = MyRobot(
        gui=True, grasper=True, clip=True, meshcat_viz=True, magnetic_gripper=True
    )
    robot.reset("bowls_in_bin")
    # robot.reset("one_object")

    obs = robot.get_obs()

    combined_pts, combined_rgb = robot.get_combined_pcd(
        obs["colors"], obs["depths"], idx=None
    )
    combined_pts, combined_rgb, _ = robot.crop_pcd(combined_pts, combined_rgb, None)
    robot.viz.view_pcd(combined_pts, combined_rgb)

    segs, info_dict = robot.get_segment_labels_and_embeddings(
        obs["colors"], obs["depths"], robot.clip
    )

    object_dicts = robot.get_segmented_pcd(
        obs["colors"],
        obs["depths"],
        segs,
        remove_floor_ht=1.0,
        std_threshold=0.02,
        label_infos=info_dict,
        visualization=True,
        process_pcd_fn=robot.crop_pcd,
    )

    description, object_dicts = robot.get_scene_description(object_dicts)
    robot.object_dicts = object_dicts
    print_object_dicts(object_dicts)

    print("-----------------------------------------------------------------")
    print(description)
    print("-----------------------------------------------------------------")

    # //////////////////////////////////////////////////////////////////////////////
    # To show Grasps for all objects
    # //////////////////////////////////////////////////////////////////////////////

    # all_grasps = []; all_scores = []
    # for obj_id in robot.object_dicts:
    #     grasps, scores = robot.get_grasp(obj_id,  threshold=0.95)

    #     if scores is None:
    #         print("No grasps to show.")

    #     else:
    #         all_grasps.append(grasps)
    #         all_scores.append(scores)
    #         best_id = np.argmax(scores)
    #         chosen_grasp = grasps[best_id: best_id+1]
    #         # chosen_grasp = grasps
    #         robot.viz.view_grasps(chosen_grasp, name=robot.object_dicts[obj_id]["used_name"].replace(" ", "_"), freq=100)

    # //////////////////////////////////////////////////////////////////////////////
    # Custom Plan check
    # //////////////////////////////////////////////////////////////////////////////


    first_bowl_id = robot.find("first bowl", "lying to the right")

    robot.pick(first_bowl_id, visualize=True)
    bowl_place_pos = robot.get_place_position(first_bowl_id, "on the tray")
    robot.place(first_bowl_id, bowl_place_pos)

    robot.update_dicts()



    # print("id of the object being picked", bowl_id)
    # robot.pick(bowl_id, visualize=True)

    # robot.place(bowl_id, bowl_place_pos)
    # robot.update_dicts()

    # robot.pick(cup_id, visualize=True)
    # cup_place_pos = robot.get_place_position(cup_id, "over the bowl")
    # robot.place(cup_id, cup_place_pos)
    # robot.update_dicts()