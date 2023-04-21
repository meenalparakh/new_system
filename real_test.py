from robot_env import print_object_dicts
from clip_model import MyCLIP
from real_env import RealRobot
from visualize_pcd import VizServer
import distinctipy as dp
import numpy as np

if __name__ == "__main__":
    robot = RealRobot(gui=False, scene_dir=None, realsense_cams=False, sam=True)

    obs = robot.get_obs()


    ###################### combined pcd visualization
    combined_pts, combined_rgb = robot.get_combined_pcd(obs["colors"], obs["depths"], idx=[0, 1, 3])
    viz = VizServer()
    # viz.view_pcd(combined_pts, combined_rgb)

    clip = MyCLIP()

    print(clip.image_preprocess)
    segs, info_dict = robot.get_segment_labels_and_embeddings(
        obs["colors"], obs["depths"], clip, vocabulary="custom", custom_vocabulary="mug,tray"
    )

    object_dicts = robot.get_segmented_pcd(
        obs["colors"],
        obs["depths"],
        segs,
        remove_floor_ht=1.0,
        label_infos=info_dict,
        visualization=True,
        process_pcd_fn=robot.crop_pcd
    )

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


    description, new_dcts = robot.get_scene_description(object_dicts)
    print_object_dicts(new_dcts)
    print(description)


