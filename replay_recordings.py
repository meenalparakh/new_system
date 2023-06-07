from robot_env import MyRobot
from recorder import Recorder

import pickle
import os
import shutil
from real_env import RealRobot, Camera
from IPython import embed
import cv2

if __name__ == "__main__":

    robot = MyRobot(gui=False, grasper=False, 
                    magnetic_gripper=False, 
                    clip=True, 
                    meshcat_viz=True, 
                    skill_learner=False)
    
    recorder = Recorder("recording_check")
    # recorder.replay_recording(robot, with_actions=False)

    # exit()

    obs, object_dict = recorder.get_obs()
    robot.object_dicts = object_dict

    robot.visualize_object_dicts(robot.object_dicts)
    # robot.get_segmap()
    # print(list(obs.keys()))
    from IPython import embed
    embed()

    # obj_id = robot.find(visual_description="green mug")

    # position = robot.get_place_position_new(4, 2, "over")
    # current_image = cv2.imread("segmap.png")
    # position = robot.get_place_position_new(2, 4, "left")

    # print("obj_id detected:", obj_id)

    exit()

    robot = RealRobot(
                gui=False,
                scene_dir=None,
                realsense_cams=True,
                sam=True,
                clip=True,
                grasper=False,
                cam_idx=[0, 1, 2, 3],
                real_robot_airobot=False,
                real_robot_polymetis=False,
                device=None)
    fname = "/Users/meenalp/Desktop/MEng/system_repos/llmrobot/experiment_results/check_learn_skill/info_20230511_1137.pkl"

    configs = []

    with open("obs2.pkl", 'rb') as f:
        obs = pickle.load(f)

    # # embed()

    robot.cams = []
    configs = obs["configs"]
    colors = obs["colors"]
    for idx, conf in enumerate(configs):
        if idx == 1:
            continue
        H, W = colors[idx].shape[:2]
        cam_extr = robot.realsense_cams.cams.cams[idx].cam_ext_mat
        cam = Camera(
            cam_extr=cam_extr, cam_intr=conf["intrinsics"], H=H, W=W
        )
        robot.cams.append(cam)

    with open(fname, 'rb') as f:
        info = pickle.load(f)

    obs = info["obs"]
    # # segs = info["segs"]

    segs, info_dict = robot.get_segment_labels_and_embeddings(obs["colors"],
                                                              obs["depths"],
                                                              robot.clip,
                                                              "custom",
                                                              "mug,banana,mango,tray")
    with open("info_dict.pkl", 'wb') as f:
        pickle.dump([segs, info_dict], f)

    with open("info_dict.pkl", 'rb') as f:
        segs, info_dict = pickle.load(f)
        
    # # embed()

    object_dicts = robot.get_segmented_pcd(colors=obs["colors"], 
                            depths=obs["depths"],
                            segs=segs,
                            std_threshold=0.03,
                            label_infos=info_dict)
    
    robot.visualize_object_dicts(object_dicts)
