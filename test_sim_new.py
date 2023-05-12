from robot_env import MyRobot, print_object_dicts

# from clip_model import MyCLIP
# from visualize_pcd import VizServer
import distinctipy as dp
import numpy as np
import pickle
from gpt_module import ChatGPTModule
from plan_and_execute import plan_and_execute

if __name__ == "__main__":
    robot = MyRobot(
        gui=True, grasper=True, clip=True, meshcat_viz=True, magnetic_gripper=True
    )
    robot.reset("bowl_over_cup")

    # //////////////////////////////////////////////////////////////////////////////
    # Labelling + segmentation + description
    # //////////////////////////////////////////////////////////////////////////////

    object_dicts = robot.get_object_dicts()
    scene_description, object_dicts = robot.get_scene_description(object_dicts)
    robot.init_dicts(object_dicts)

    # //////////////////////////////////////////////////////////////////////////////
    # Visualization of individual object point clouds
    # //////////////////////////////////////////////////////////////////////////////

    robot.visualize_object_dicts(object_dicts)

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

    # bowl_id = robot.find("bowl", "lying on the right side of the table")
    # basket_id = robot.find("basket", "lying on the left of the bowl")

    # print("id of the object being picked", bowl_id)
    # robot.pick(bowl_id, visualize=False)

    # basket_location = robot.get_location(basket_id)
    # robot.place(bowl_id, basket_location)

    # //////////////////////////////////////////////////////////////////////////////
    # LLM Planning and Execution in Loop
    # //////////////////////////////////////////////////////////////////////////////

    chat_module = ChatGPTModule()
    plan_and_execute(robot, scene_description, chat_module)


