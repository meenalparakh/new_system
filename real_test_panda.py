from robot_env import print_object_dicts
from real_env import RealRobot
from visualize_pcd import VizServer
import distinctipy as dp
import numpy as np
from prompt_manager import get_plan, execute_plan
from gpt_module import ChatGPTModule
import pickle


if __name__ == "__main__":
    robot = RealRobot(
        gui=False,
        scene_dir=None,
        realsense_cams=True,
        sam=True,
        clip=True,
        cam_idx=[0, 1, 2, 3],
    )

    # obs = robot.get_obs(source="obs2.pkl")
    obs = robot.get_obs(source="realsense")

    ###################### combined pcd visualization
    combined_pts, combined_rgb = robot.get_combined_pcd(
        obs["colors"], obs["depths"], idx=[0, 1, 3]
    )
    viz = VizServer()
    viz.view_pcd(combined_pts, combined_rgb)

    segs, info_dict = robot.get_segment_labels_and_embeddings(
        obs["colors"],
        obs["depths"],
        robot.clip,
        vocabulary="custom",
        custom_vocabulary="tray,mug",
    )

    with open("cached_info.pkl", 'wb') as f:
        pickle.dump([segs, info_dict], f)

    with open("cached_info.pkl", 'rb') as f:
        segs, info_dict = pickle.load(f)


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
    print(description)

    # //////////////////////////////////////////////////////////////////////////////
    # Visualization of point clouds
    # //////////////////////////////////////////////////////////////////////////////

    print("Number of objects", len(object_dicts))
    pcds = []; colors = []
    rand_colors = dp.get_colors(len(object_dicts))
    for idx, obj in enumerate(object_dicts):
        label = object_dicts[obj]["label"][0]
        pcd = object_dicts[obj]["pcd"]
        color = object_dicts[obj]["rgb"]

        viz.view_pcd(pcd, color, f"{idx}_{label}")
        pcds.append(pcd)
        colors.append(color)

    pcds = np.vstack(pcds); colors = np.vstack(colors)
    viz.view_pcd(pcds, colors)

    # //////////////////////////////////////////////////////////////////////////////
    # LLM Planning
    # //////////////////////////////////////////////////////////////////////////////


    obj_id = input("obj_id")
    place_position = input("place position:")

    obj_id = int(obj_id)
    place_position = [float(k) for k in place_position.split(" ")]

    robot.pick_place_real(obj_id, place_position)

    # robot.get_primitives()
    # task_name = "place_mug"
    # task_prompt = "place the mug into the tray"

    # chat_module = ChatGPTModule()
    # chat_module.start_session("test_run")
    # task_name, code_rectified = get_plan(description, task_prompt, chat_module.chat, task_name, robot.primitives_lst, robot.primitives_description)

    # print(" FINAL ANSWER >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    # print("TASK:", task_name)
    # print(code_rectified)

    # final_code = code_rectified.replace("`", "")

    # execute_plan(robot, task_name, final_code)
