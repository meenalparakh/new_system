from robot_env import MyRobot, print_object_dicts
from clip_model import MyCLIP
from visualize_pcd import VizServer
import distinctipy as dp
import numpy as np
import pickle

if __name__ == "__main__":
    robot = MyRobot(gui=True, grasper=True, clip=True)
    # robot.reset("cup_over_bowl")
    robot.reset("one_object")

    obs = robot.get_obs()

    combined_pts, combined_rgb = robot.get_combined_pcd(
        obs["colors"], obs["depths"], idx=None
    )
    viz = VizServer()
    viz.view_pcd(combined_pts, combined_rgb)

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
        process_pcd_fn=robot.crop_pcd
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
    # Grasps
    # //////////////////////////////////////////////////////////////////////////////


    all_grasps = []; all_scores = []
    for obj_id in robot.object_dicts:
        grasps, scores = robot.get_grasp(obj_id,  threshold=0.95)

        if scores is None:
            print("No grasps to show.")
        
        else:
            all_grasps.append(grasps)
            all_scores.append(scores)
            best_id = np.argmax(scores)
            chosen_grasp = grasps[best_id: best_id+1]
            # chosen_grasp = grasps
            viz.view_grasps(chosen_grasp, name=robot.object_dicts[obj_id]["used_name"].replace(" ", "_"), freq=100)

    with open("object_grasps.pkl", 'wb') as f:
        pickle.dump([all_grasps, all_scores], f)

    # //////////////////////////////////////////////////////////////////////////////
    # LLM Planning
    # //////////////////////////////////////////////////////////////////////////////

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