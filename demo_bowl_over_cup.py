from robot_env import MyRobot, print_object_dicts
import numpy as np
from gpt_module import ChatGPTModule
from plan_and_execute import plan_and_execute

if __name__ == "__main__":
    robot = MyRobot(
        gui=True, grasper=True, clip=True, meshcat_viz=True, magnetic_gripper=True
    )
    robot.reset("bowl_over_cup")
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
        std_threshold=0.025,
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
    # Checking place options
    # //////////////////////////////////////////////////////////////////////////////

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

    bowl_id = robot.find("bowl", "lying over the cup")
    cup_id = robot.find("cup", "lying on the right of the table")

    print("id of the object being picked", bowl_id)
    robot.pick(bowl_id, visualize=True)
    bowl_place_pos = robot.get_place_position(bowl_id, "farther from the cup")

    robot.place(bowl_id, bowl_place_pos)
    robot.update_dicts()

    robot.pick(cup_id, visualize=True)
    cup_place_pos = robot.get_place_position(cup_id, "over the bowl")
    robot.place(cup_id, cup_place_pos)
    robot.update_dicts()

    input("wait")

    # //////////////////////////////////////////////////////////////////////////////
    # LLM Planning and Execution in Loop
    # //////////////////////////////////////////////////////////////////////////////

    chat_module = ChatGPTModule()
    plan_and_execute(robot, chat_module)