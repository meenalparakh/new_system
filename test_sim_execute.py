from robot_env import MyRobot, print_object_dicts
from clip_model import MyCLIP
from visualize_pcd import VizServer
import distinctipy as dp
import numpy as np
import pickle
from gpt_module import ChatGPTModule
from prompt_manager import get_plan, execute_plan

if __name__ == "__main__":
    robot = MyRobot(gui=False, grasper=True, clip=True, meshcat_viz=True)
    robot.reset("cup_over_bowl")
    # robot.reset("one_object")

    # obs = robot.get_obs()
    # combined_pts, combined_rgb = robot.get_combined_pcd(
    #     obs["colors"], obs["depths"], idx=None
    # )
    # combined_pts, combined_rgb, _ = robot.crop_pcd(combined_pts, combined_rgb, None)
    # robot.viz.view_pcd(combined_pts, combined_rgb)

    # segs, info_dict = robot.get_segment_labels_and_embeddings(
    #     obs["colors"], obs["depths"], robot.clip
    # )
    # object_dicts = robot.get_segmented_pcd(
    #     obs["colors"],
    #     obs["depths"],
    #     segs,
    #     remove_floor_ht=1.0,
    #     std_threshold=0.02,
    #     label_infos=info_dict,
    #     visualization=True,
    #     process_pcd_fn=robot.crop_pcd,
    # )

    # description, object_dicts = robot.get_scene_description(object_dicts)

    # robot.object_dicts = object_dicts
    # print_object_dicts(object_dicts)
    # print(description)

    # with open("object_dicts.pkl", 'wb') as f:
    #     pickle.dump(object_dicts, f)

    # input("stop")

    with open("object_dicts.pkl", 'rb') as f:
        robot.object_dicts = pickle.load(f)      



    # //////////////////////////////////////////////////////////////////////////////
    # Grasps
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
    
    mug_id = robot.find("bowl", "lying on the right side of the table")
    basket_id = robot.find("basket", "lying on the left of the bowl")

    robot.pick(mug_id, visualize=True)
    
    basket_location = robot.get_location(basket_id)
    robot.place(mug_id, basket_location)


    # //////////////////////////////////////////////////////////////////////////////
    # Execution check
    # //////////////////////////////////////////////////////////////////////////////

    code_str = """
bowl_id = find("bowl")
bowl_pos = get_location(bowl_id)

tray_id = find("tray")
tray_pos = get_location(tray_id)

pick(bowl_id)
move(bowl_pos, tray_pos)

tilt_bowl = learn_skill("tilt_object")
tilt_bowl(bowl_id)

shelf_id = find("shelf")
shelf_pos = get_location(shelf_id)

"""


    # //////////////////////////////////////////////////////////////////////////////
    # LLM Planning
    # //////////////////////////////////////////////////////////////////////////////

    # robot.get_primitives()
    # task_name = "mug_in_basket"
    # task_prompt = "place the mug into the basket"

    # chat_module = ChatGPTModule()
    # chat_module.start_session("test_run")
    # task_name, code_rectified = get_plan(
    #     description,
    #     task_prompt,
    #     chat_module.chat,
    #     task_name,
    #     robot.primitives_lst,
    #     robot.primitives_description,
    # )

    # print(
    #     " FINAL ANSWER >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    # )
    # print("TASK:", task_name)
    # print(code_rectified)

    # final_code = code_rectified.replace("`", "")

    # execute_plan(robot, task_name, final_code)
