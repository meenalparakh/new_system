from robot_env import MyRobot, print_object_dicts
from clip_model import MyCLIP
from visualize_pcd import VizServer
import distinctipy as dp
import numpy as np
import pickle
from gpt_module import ChatGPTModule
from prompt_manager import get_plan, execute_plan

if __name__ == "__main__":
    robot = MyRobot(
        gui=True, grasper=True, clip=True, meshcat_viz=False, magnetic_gripper=True
    )
    robot.reset("cup_over_bowl")
    # robot.reset("one_object")

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
    # print(description)

    # //////////////////////////////////////////////////////////////////////////////
    # Custom Plan check
    # //////////////////////////////////////////////////////////////////////////////
    bowl_id = robot.find("bowl", "lying on the right side of the table")
    basket_id = robot.find("basket", "lying on the left of the bowl")

    print("id of the object being picked", bowl_id)
    robot.pick(bowl_id, visualize=False)   

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
