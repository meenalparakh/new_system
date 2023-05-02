import numpy as np
from robot_env import MyRobot
import cv2
from multiprocessing.pool import ThreadPool
import threading
from threading import Thread
import pickle
import os

np.random.seed(10)
# class MyObjectTracker:
#     def __init__(self, tracker="goturn"):
#         self.tracker = cv2.TrackerGOTURN_create()

#     def register_bbs(self, image, bb):
#         success = self.tracker.init(image, bb)
#         return success

#     def track(self):
#         ok, bbox = self.tracker.update(frame)

# def store_obs(robot, n=50):
#     for i in range(n):
#         obs = robot.get_obs()
#         robot.obs_lst.append(obs)

if __name__ == "__main__":
    robot = MyRobot(
        gui=False, grasper=True, magnetic_gripper=True, clip=True, meshcat_viz=False
    )
    robot.reset("cup_over_bowl")

    obs = robot.get_obs()
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
    # print_object_dicts(object_dicts)
    print(description)

    robot.obs_lst = []
    # thread = Thread(target=store_obs, args=(robot,))
    # thread.start()

    bowl_id = robot.find("bowl", "lies to the right of the table")
    print("bowl_id", bowl_id)
    robot.pick(3)

    with open("obs_lst.pkl", "wb") as f:
        pickle.dump(robot.obs_lst, f)

    # thread.join()

    # os.makedirs("threading_check", exists_ok=True)

    # colors = [o["colors"] for o in robot.obs_lst]
    # for idx, c in enumerate(colors):
    #     cv2.imwrite(f"threading_check/{idx}.png")
