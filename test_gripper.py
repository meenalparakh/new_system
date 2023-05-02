import pickle
from robot_env import MyRobot


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

    # robot.pick(bowl_id, visualize=True)

    basket_location = robot.get_location(basket_id)
    robot.place(bowl_id, basket_location)
