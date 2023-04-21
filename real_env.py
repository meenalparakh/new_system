import os
import numpy as np
import open3d
from robot_env import MyRobot
from visualize_pcd import VizServer

import pickle
import subprocess
import cv2
# from get_real_data import RealSenseCameras
import subprocess

DATA_DIR = "../data/scene_data/"

subprocess.run(['cp', 'demo_detic.py', 'Detic/'])
from collections import namedtuple
Args = namedtuple("Args", ["vocabulary", "custom_vocabulary"])

def get_detic_predictions(images, vocabulary="lvis", custom_vocabulary=""):

    # args = Args(vocabulary=vocabulary, custom_vocabulary=custom_vocabulary)
    # image_dir = "current_images"
    # os.makedirs(image_dir, exist_ok=True)

    # for idx, img in enumerate(images):
    #     cv2.imwrite(os.path.join(image_dir, f"{idx}.png"), img)

    # with open(os.path.join(image_dir, "args.pkl"), 'wb') as f:
    #     pickle.dump(args, f)
            
    # subprocess.run(["./run_detect_grasp_scripts.sh", "detect", os.path.abspath(image_dir)])

    with open("./Detic/predictions_summary.pkl", 'rb') as f:
        pred_lst, names = pickle.load(f)

    if vocabulary == "custom":
        names = custom_vocabulary.split(',')

    return pred_lst, names

def get_bb_labels(pred, names):
    '''
    if there are N images
    returned is N instances, each instance class 
    containing a list of bbs and a corresponding list of labels
    
    the bbs (and the labels) are filtered as long as the bb covers 
    non zero region in the pcd projection.

    For the filtered bbs (and labels), find the crop embedding using clip
    '''
    
    instances = pred["instances"]
    pred_masks = instances.pred_masks
    pred_boxes = instances.pred_boxes.tensor.numpy()
    pred_classes = instances.pred_classes 
    pred_scores = instances.scores
    pred_labels = [names[idx] for idx in pred_classes]

    return pred_boxes, pred_labels

def get_segmentation_mask(predictor, image, bbs):
    
    '''
    for each bb in the image, get the segmentation mask, using SAM
    
    returns a dictionary, that contains for each segment_id in the mask, 
    the corresponding label and the embedding

    this function will be used for each image to obtain a list of segs
    and image_embeddings (see robot_env) class
    '''
    # pass
    predictor.set_image(image, image_format="RGB")

    results = []
    for j in range(len(bbs)):
        b = bbs[j].numpy()
        masks, scores, logits = predictor.predict(
                box=bbs[j],
                multimask_output=False
        )

        results.append(masks[0])

    return results

def find_object(text_prompt, avg_obj_embeddings):
    
    '''
    returns the object index in the obj_embeddings that best match the 
    text_prompt
    '''
    pass





class Camera:
    def __init__(self, cam_extr=None, cam_intr=None, H=None, W=None):
        self.depth_scale = 1.0
        cam_int_mat_inv = np.linalg.inv(cam_intr)

        img_pixs = np.mgrid[0:H, 0:W].reshape(2, -1)
        img_pixs[[0, 1], :] = img_pixs[[1, 0], :]
        _uv_one = np.concatenate((img_pixs, np.ones((1, img_pixs.shape[1]))))
        self._uv_one_in_cam = np.dot(cam_int_mat_inv, _uv_one)
        self.cam_ext_mat = cam_extr
        self.cam_intr_mat = cam_intr

    def get_cam_ext(self):
        return self.cam_ext_mat

    def get_cam_intr(self):
        return self.cam_intr_mat


class RealRobot(MyRobot):
    def __init__(self, gui=False, scene_dir=None, realsense_cams=False, sam=False):
        super().__init__(gui)
        self.scene_dir = scene_dir
        self.table_bounds = np.array([[0.15, 0.50], [-0.5, 0.5], [0.03, 1.0]])

        self.realsense_cams = None
        if realsense_cams:
            from get_real_data import RealSenseCameras
            self.realsense_cams = RealSenseCameras([0, 1, 2, 3])

        self.sam_predictor = None
        if sam:
            self.set_sam()

    def set_sam(self):
        from segment_anything import sam_model_registry, SamPredictor
        sam_checkpoint = "sam_model/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        device = "gpu"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        self.sam_predictor = SamPredictor(sam)

    def get_obs(self, source="pkl"):
        ## source can be "scene_dir", "pkl", "realsense"

        if source == "scene_dir":
            from data_collection.utils.data_collection_utils import (
                read_all,
                get_scale,
                rescale_extrinsics,
                transform_configs,
            )
            colors, depths, configs, _ = read_all(self.scene_dir, skip_image=40)
            print("Number of configs", len(configs), len(depths), len(colors))
            scale = get_scale(self.scene_dir, depths, configs)
            configs = rescale_extrinsics(configs, scale)
            obs = {"colors": colors, "depths": depths, "configs": configs}

        elif source == "pkl":
            with open("obs.pkl", "rb") as f:
                obs = pickle.load(f)
        elif source == "realsense":
            obs = self.realsense_cams.get_rgb_depth()

        self.cams = []
        configs = obs["configs"]
        colors = obs["colors"]
        for idx, conf in enumerate(configs):
            H, W = colors[idx].shape[:2]
            cam = Camera(
                cam_extr=conf["extrinsics"], cam_intr=conf["intrinsics"], H=H, W=W
            )
            self.cams.append(cam)

        return obs

    def crop_pcd(self, pts, rgb=None, segs=None, bounds=None):
        if bounds is None:
            bounds = self.table_bounds
        # Filter out 3D points that are outside of the predefined bounds.
        ix = (pts[Ellipsis, 0] >= bounds[0, 0]) & (
            pts[Ellipsis, 0] < bounds[0, 1]
        )
        iy = (pts[Ellipsis, 1] >= bounds[1, 0]) & (
            pts[Ellipsis, 1] < bounds[1, 1]
        )
        iz = (pts[Ellipsis, 2] >= bounds[2, 0]) & (
            pts[Ellipsis, 2] < bounds[2, 1]
        )
        valid = ix & iy & iz

        pts = pts[valid]
        if rgb is not None:
            rgb = rgb[valid]
        if segs is not None:
            segs = segs[valid]

        return pts, rgb, segs
    


if __name__ == "__main__":
    # scene_dir = os.path.join(DATA_DIR, "20221010_1759")

    robot = RealRobot(gui=False, scene_dir=None)
    obs = robot.get_obs()

    pred_lst, names = get_detic_predictions(obs["colors"], vocabulary="custom", custom_vocabulary="mug,tray")

    for idx, rgb in enumerate(obs["colors"]):
        pred_boxes, pred_labels = get_bb_labels(pred_lst[idx], names)
        masks = get_segmentation_mask(robot.sam_predictor, rgb, pred_boxes)
        

    ###################### combined pcd visualization 
    # combined_pts, combined_rgb = robot.get_combined_pcd(obs["colors"], obs["depths"], idx=[0, 1, 3])
    # combined_pts, combined_rgb = robot.crop_pcd(combined_pts, combined_rgb)
    # viz = VizServer()
    # viz.view_pcd(combined_pts, combined_rgb)
