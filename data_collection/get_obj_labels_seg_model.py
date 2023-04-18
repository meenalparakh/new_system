import sys
sys.path.append('../')

import cv2
import numpy as np
import argparse
import os
import open3d
import random
from matplotlib.gridspec import GridSpec

from data_collection.utils.data_collection_utils import get_intr_from_cam_params
from data_collection.scene_wrapper import Scene
import pickle
from data_collection.utils.pointcloud_utils import project_pcd_to_image
from data_collection.utils.data_post_processing_utils import get_segmentation_data
from matplotlib import pyplot as plt
from language.text_encoder import build_text_encoder
import subprocess

from data_collection.utils.data_collection_utils import save_pcd
import meshcat
import meshcat.geometry as g
from grasping.test_meshcat_pcd import sample_grasp_show as viz_grasps
from obj_label_utils import *
from data_collection.utils.pointcloud_utils import get_pcds
'''
TO DO:
4. find their entropies in text embedding space.
    4a. which embeding to choose? BERT, GPT, CLIP - for now using CLIP embeddings
5. those with high variance should ask for human input
6. parse human input to obtain the final label
7. contactnet.py and mesh_utils.py usses ROOT_FOLDER
'''

NUM_VIEWS = 4
# NUM_POINTS_ON_OBJECT = 10
def view_grasps(scene_dir):
    pcd = open3d.io.read_point_cloud(os.path.join(scene_dir, "cleaned_pcd.ply"))

    with open(os.path.join(scene_dir, "contactnet_grasps.pkl"), 'rb') as f:
        contactnet_grasps = pickle.load(f)

    object_info = {k: {} for k in contactnet_grasps}
    for k in contactnet_grasps:
        object_info[k]["pred_grasps"] = contactnet_grasps[k]["pred_grasps"]
        object_info[k]["pred_grasp_scores"] = contactnet_grasps[k]["pred_success"]

    visualize_labeled_scenes(pcd, object_info, grasp_type="best")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-dir", type=str, default='../data/scene_data/20221030_0540')
    args = parser.parse_args()    
    # skip_generation = True

    # view_grasps(args.scene_dir)
    # exit()

    scene_name = os.path.normpath(args.scene_dir).split(os.sep)[-1]
    # cleaned_pcd = os.path.join(args.scene_dir, 'c')
    image_dir = os.path.join(args.scene_dir, "images")
    image_dir = os.path.join(args.scene_dir, "depth")

    tmp = os.path.join(args.scene_dir, 'transformed_configs.pkl')
    
    with open(tmp, 'rb') as f:
        configs = pickle.load(f)

    # configs = list of config, config is a dict with fields: 
    # intrinsics: 3x3 matrix
    # fname: name of image
    # height:
    # width: 
    # extrinsics: 4x4 matrix

    k0 = min(NUM_VIEWS, len(configs))
    chosen_configs = random.sample(configs, k=k0)
    cam_extrs = [config['extrinsics'] for config in chosen_configs]
    names = [config['fname'] for config in chosen_configs]

    colors = [cv2.imread(os.path.join(image_dir, names[i]))[:,:,::-1] for i in range(k0)]
    depths = [cv2.imread(os.path.join(image_dir, names[i]))[:,:,::-1] for i in range(k0)]
    xyzs, colors, cam_extrs = get_pcds(colors, depths, chosen_configs)

    pcd = save_pcd(xyzs, colors, cam_extrs, None, downsample_voxel_size=0.001)
    pcd, plane_params, normal_vec = get_table_cluster(pcd, eps=0.01,
                            min_points=200, get_normal=self.summary["table_only"])
    pcd, transform = postprocess_pointcloud(pcd, plane_params, normal_vec,
                revert_normal=revert_normal, visualize=False,
                    keep_table=self.summary["table_only"])


    scene = Scene(args.scene_dir, scene_name)
    pcd, clustered_pcd, diff_objects_info, object_pcds = \
                get_pcd_cluster_info(args.scene_dir, get_object_pcds=True)
    object_centers = [np.mean(np.asarray(p.points), axis=0) for p in object_pcds]

    diff_colors = diff_objects_info['diff_colors'][:,:3]
    num_objs = len(object_pcds)
    assert num_objs == len(diff_colors)
    print("number of objects:", num_objs)

    data_dir = os.path.abspath(os.path.join(args.scene_dir, "labels"))
    os.makedirs(data_dir, exist_ok=True)
    flag = 'A'
    masks_dir = os.path.abspath(os.path.join(data_dir, 'A', 'masks'))
    cam_intr, height, width = get_intr_from_cam_params(scene.scene_dir, hw=True)

    tmp = os.path.join(args.scene_dir, 'transformed_configs.pkl')
    with open(tmp, 'rb') as f:
        configs = pickle.load(f)

    chosen_configs = random.sample(configs, k=NUM_VIEWS)
    cam_extrs = [config['extrinsics'] for config in chosen_configs]
    names = [config['fname'] for config in chosen_configs]

    masks_dir =  os.path.abspath(os.path.join(data_dir, 'A', 'masks'))
    os.makedirs(masks_dir, exist_ok=True)
        
    print('Projecting pcd to images...')
    data_lst = []
    for idx, cam_extr in enumerate(cam_extrs):            
        color_fname = names[idx]
        print(color_fname)
        background_val = 0
        mask = project_pcd_to_image(clustered_pcd, cam_intr, cam_extr, height, width)

        segments_info, mask_int = get_segmentation_data(mask, diff_colors, amodal_masks=None, 
                                    object_centers=object_centers,
                                    save_dir=None, max_num_pixels=10)

        d = get_dict(idx, image_dir, masks_dir, color_fname, height, width, cam_extr, 
                    segments_info, None, scene.object_category, flag='A')
        data_lst.append(d)

        mask = cv2.medianBlur(mask.astype(np.uint8), 5)
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
        mask_fname = os.path.join(masks_dir, color_fname)
        cv2.imwrite(mask_fname, mask)
        cv2.imwrite(os.path.join(masks_dir, 'labels_' + color_fname), mask_int)

    with open(os.path.join(args.scene_dir, f'data_lst_{NUM_VIEWS}.pkl'), 'wb') as f:
        pickle.dump(data_lst, f)
    
    # with open(os.path.join(args.scene_dir, f'data_lst_{NUM_VIEWS}.pkl'), 'rb') as f:
    #     data_lst = pickle.load(f)

    print('projecting masks complete')
    scene_dir_abs_path = str(os.path.abspath(args.scene_dir))

    # subprocess.run(["./run_detect_grasp_scripts.sh", "detect", scene_dir_abs_path, str(NUM_VIEWS)])

    results_fname = os.path.join(args.scene_dir, 'detic_results', 'predictions_summary.pkl')
    with open(results_fname, 'rb') as f:
        predictions_info = pickle.load(f)

    classes = predictions_info["names"]
    predictions = predictions_info["predictions"]
    print("length of classes", len(classes))

    count_distribution = {i: dict() for i in range(num_objs)}
    score_distribution = {i: dict() for i in range(num_objs)}
    
    for idx, d in enumerate(data_lst):
        mask = cv2.imread(os.path.join(masks_dir, 'labels_' + d["file_name"]),
                            cv2.IMREAD_UNCHANGED)
        instances = predictions[idx]["instances"]
        count_distribution, score_distribution = get_category_label(mask, 
                        instances.pred_masks, 
                        instances.pred_classes, 
                        instances.scores,
                        num_objs, classes, 
                        count_dict=count_distribution,
                        score_dict=score_distribution)

    object_info = process_score_distribution(score_distribution, 
                    args.scene_dir, plot=True, data_lst=data_lst)

    print('object detection complete')

    # subprocess.run(["./run_detect_grasp_scripts.sh", "grasp", scene_dir_abs_path, str(0.9)])

    with open(os.path.join(args.scene_dir, "contactnet_grasps.pkl"), 'rb') as f:
        contactnet_grasps = pickle.load(f)

    for k in object_info:
        object_info[k]["object_center"] = object_centers[k]
        object_info[k]["pred_grasps"] = contactnet_grasps[k]["pred_grasps"]
        object_info[k]["pred_grasp_scores"] = contactnet_grasps[k]["pred_success"]

    with open(os.path.join(args.scene_dir, "object_info_dict.pkl"), 'wb') as f:
        pickle.dump(object_info, f)

    visualize_labeled_scenes(pcd, object_info)