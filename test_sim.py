from robot_env import MyRobot, print_object_dicts

# from clip_model import MyCLIP
# from visualize_pcd import VizServer
import distinctipy as dp
import numpy as np
import pickle
from gpt_module import ChatGPTModule
from prompt_manager import get_plan_loop, execute_plan_new

if __name__ == "__main__":
    robot = MyRobot(
        gui=True, grasper=True, clip=True, meshcat_viz=False, magnetic_gripper=True
    )
    robot.reset("cup_over_bowl")
    # robot.reset("one_object")

    obs = robot.get_obs()

    # //////////////////////////////////////////////////////////////////////////////
    # Visualization of initial point cloud
    # //////////////////////////////////////////////////////////////////////////////

    combined_pts, combined_rgb = robot.get_combined_pcd(
        obs["colors"], obs["depths"], idx=None
    )
    combined_pts, combined_rgb, _ = robot.crop_pcd(combined_pts, combined_rgb, None)
    robot.viz.view_pcd(combined_pts, combined_rgb)

    # //////////////////////////////////////////////////////////////////////////////
    # Labelling + segmentation + description
    # //////////////////////////////////////////////////////////////////////////////

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
    print(description)

    # //////////////////////////////////////////////////////////////////////////////
    # Visualization of individual object point clouds
    # //////////////////////////////////////////////////////////////////////////////

    print("Number of objects", len(object_dicts))
    pcds = []
    colors = []
    rand_colors = dp.get_colors(len(object_dicts))
    for idx, obj in enumerate(object_dicts):
        label = object_dicts[obj]["label"][0]
        pcd = object_dicts[obj]["pcd"]
        color = object_dicts[obj]["rgb"]

        robot.viz.view_pcd(pcd, color, f"{idx}_{label}")
        pcds.append(pcd)
        colors.append(color)

    pcds = np.vstack(pcds)
    colors = np.vstack(colors)
    robot.viz.view_pcd(pcds, colors)

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
    #         viz.view_grasps(chosen_grasp, name=robot.object_dicts[obj_id]["used_name"].replace(" ", "_"), freq=100)

    # with open("object_grasps.pkl", 'wb') as f:
    #     pickle.dump([all_grasps, all_scores], f)

    # //////////////////////////////////////////////////////////////////////////////
    # Custom Plan check
    # //////////////////////////////////////////////////////////////////////////////

    bowl_id = robot.find("bowl", "lying on the right side of the table")
    basket_id = robot.find("basket", "lying on the left of the bowl")

    print("id of the object being picked", bowl_id)
    robot.pick(bowl_id, visualize=False)

    basket_location = robot.get_location(basket_id)
    robot.place(bowl_id, basket_location)

    # //////////////////////////////////////////////////////////////////////////////
    # LLM Planning and Execution in Loop
    # //////////////////////////////////////////////////////////////////////////////

    chat_module = ChatGPTModule()
    chat_module.start_session("test_run")

    first_code_run = False
    prev_code_exec = ""
    max_num_prompts = 3
    prompt_idx = 0

    user_input = input("Press enter to proceed")

    while user_input != "quit" and (prompt_idx < max_num_prompts):
        scene_description = description
        is_verbal = input("Is it a verbal query? y or n: ")
        is_verbal = (is_verbal == "y") or (is_verbal == "Y")

        task_prompt = input("Enter the task prompt: ")

        if is_verbal:
            task_name = ""
        else:
            task_name = input("Enter task name: ")

        response = get_plan_loop(
            scene_description,
            task_prompt,
            chat_module.chat,
            task_name,
            robot.primitives_lst,
            robot.primitives_description,
            code_rectification=first_code_run,
            first_run=first_code_run,
            verbal_query=is_verbal,
        )

        if is_verbal:
            print(
                " FINAL ANSWER >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
            )
            print(response)
            print(
                " >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
            )
        else:
            print(
                " FINAL ANSWER >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
            )
            task_name, code_str = response
            print("TASK NAME:", task_name)
            print("CODE:", code_str)
            print(
                " >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
            )

            final_code = code_str.replace("`", "")

            prev_code_exec = execute_plan_new(
                robot, task_name, final_code, prev_code_str=prev_code_exec
            )
            print(robot.primitives.keys())

        if not is_verbal:
            first_code_run = False

        description = ""
        prompt_idx += 1
        user_input = input("Press enter to continue")
