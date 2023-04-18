from robot_env import MyRobot, print_object_dicts
from clip_model import MyCLIP

if __name__ == "__main__":
    robot = MyRobot(gui=True)
    robot.reset("cup_over_bowl")
    obs = robot.get_obs()
    pts, rgb = robot.get_combined_pcd(obs["colors"], obs["depths"])
    # print(len(obs["segs"]), obs["segs"][0].shape)

    # visualize_pcd(pts, rgb)

    clip = MyCLIP()

    print(clip.image_preprocess)
    segs, info_dict = robot.get_segment_labels_and_embeddings(obs["colors"], obs["depths"], clip)

    object_dicts = robot.get_segmented_pcd(obs["colors"], obs["depths"], segs, remove_floor_ht=1.0, label_infos=info_dict)

    description, new_dcts = robot.get_scene_description(object_dicts)
    print_object_dicts(new_dcts)
    print(description)

    


