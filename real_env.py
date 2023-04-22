import os
import numpy as np
import open3d
from robot_env import MyRobot
from visualize_pcd import VizServer

import matplotlib.pyplot as plt

import pickle
import subprocess
import cv2
import shutil

# from get_real_data import RealSenseCameras

DATA_DIR = "../data/scene_data/"

import torch

subprocess.run(["cp", "demo_detic.py", "Detic/"])


def get_detic_predictions(images, vocabulary="lvis", custom_vocabulary=""):
    image_dir = "current_images"
    if os.path.exists(image_dir):
        shutil.rmtree(image_dir)

    os.makedirs(image_dir)

    image_fnames = []
    for idx, img in enumerate(images):
        fname = os.path.join(image_dir, f"{idx}.png")
        cv2.imwrite(fname, img[:,:,::-1])
        image_fnames.append(f"{idx}.png")

    print(image_fnames)

    subprocess.run(
        [
            "./run_detect_grasp_scripts.sh",
            "detect",
            os.path.abspath(image_dir),
            vocabulary,
            custom_vocabulary,
        ]
    )

    with open("./Detic/predictions_summary.pkl", "rb") as f:
        info_dict, names = pickle.load(f)

    pred_lst = [info_dict[fname] for fname in image_fnames]

    if vocabulary == "custom":
        names = custom_vocabulary.split(",")

    return pred_lst, names


def get_bb_labels(pred, names):
    """
    if there are N images
    returned is N instances, each instance class
    containing a list of bbs and a corresponding list of labels

    the bbs (and the labels) are filtered as long as the bb covers
    non zero region in the pcd projection.

    For the filtered bbs (and labels), find the crop embedding using clip
    """

    instances = pred["instances"]
    pred_masks = instances.pred_masks
    pred_boxes = instances.pred_boxes.tensor.cpu().numpy()
    pred_classes = instances.pred_classes
    pred_scores = instances.scores
    pred_labels = [names[idx] for idx in pred_classes]

    return pred_boxes, pred_labels


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )


def get_segmentation_mask(predictor, image, bbs, labels, prefix=0):
    """
    for each bb in the image, get the segmentation mask, using SAM

    returns a dictionary, that contains for each segment_id in the mask,
    the corresponding label and the embedding

    this function will be used for each image to obtain a list of segs
    and image_embeddings (see robot_env) class
    """
    # pass
    predictor.set_image(image, image_format="RGB")

    sam_predictions_dir = "sam_predictions"
    if os.path.exists(sam_predictions_dir):
        shutil.rmtree(sam_predictions_dir)
    os.makedirs(sam_predictions_dir)

    H, W = image.shape[:2]
    seg = np.zeros((H, W), dtype=int)
    embedding_dict = {}

    results = []
    for j in range(len(bbs)):
        masks, scores, logits = predictor.predict(box=bbs[j], multimask_output=False)
        mask = masks[0]

        r1, c1, r2, c2 = bbs[j]
        margin = 1
        r1 = max(int(r1 - margin), 0)
        r2 = min(int(r2 + margin), H - 1)
        c1 = max(int(c1 - margin), 0)
        c2 = min(int(c2 + margin), W - 1)

        print("crop dims", r1, c1, r2, c2)
        crop = image[r1 : r2 + 1, c1 : c2 + 1, :]

        seg[mask > 0.5] = j + 1
        embedding_dict[j + 1] = {"label": labels[j], "crop": crop}

        results.append(mask)

        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(masks[0], plt.gca())
        show_box(bbs[j], plt.gca())
        plt.axis("off")

        plt.savefig(sam_predictions_dir + f"/{prefix}_{j}.png")

    return seg, embedding_dict  # results


def get_clip_embeddings(d, clip_):
    keys = list(d.keys())

    crops = [d[k]["crop"] for k in keys]

    if len(crops) == 0:
        return d

    print("number of crops", len(crops))
    embeddings = clip_.get_image_embeddings(crops)

    for idx, k in enumerate(keys):
        d[k]["embedding"] = embeddings[idx]

    return d


def find_object(text_prompt, avg_obj_embeddings):
    """
    returns the object index in the obj_embeddings that best match the
    text_prompt
    """
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
    def __init__(
        self, gui=False, scene_dir=None, realsense_cams=False, sam=False, clip=False, cam_idx=[0, 1, 2, 3]
    ):
        super().__init__(gui)
        self.scene_dir = scene_dir
        self.table_bounds = np.array([[0.15, 1.0], [-0.5, 0.5], [-0.01, 1.0]])

        self.realsense_cams = None
        if realsense_cams:
            from get_real_data import RealSenseCameras
            self.realsense_cams = RealSenseCameras(cam_idx)

        self.sam_predictor = None
        if sam:
            self.set_sam()
        if clip:
            from clip_model import MyCLIP

            self.clip = MyCLIP()

    def set_sam(self):
        from segment_anything import sam_model_registry, SamPredictor

        sam_checkpoint = "sam_model/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        device = "cuda" if torch.cuda.is_available() else "cpu"

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

        elif "pkl" in source:
            with open(source, "rb") as f:
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

    def get_segment_labels_and_embeddings(
        self, colors, depths, clip_, vocabulary="lvis", custom_vocabulary=""
    ):
        if clip_ is None:
            clip_ = self.clip
        pred_lst, names = get_detic_predictions(
            colors, vocabulary=vocabulary, custom_vocabulary=custom_vocabulary
        )

        segs = []
        image_embeddings = []

        for idx, rgb in enumerate(colors):
            print("Obtaining segmentation mask")
            pred_boxes, pred_labels = get_bb_labels(pred_lst[idx], names)
            seg, embedding_dict = get_segmentation_mask(
                self.sam_predictor, rgb, pred_boxes, pred_labels, prefix=idx
            )
            embedding_dict = get_clip_embeddings(embedding_dict, clip_)

            segs.append(seg)
            image_embeddings.append(embedding_dict)

        return segs, image_embeddings

    def crop_pcd(self, pts, rgb=None, segs=None, bounds=None):
        if bounds is None:
            bounds = self.table_bounds
        # Filter out 3D points that are outside of the predefined bounds.
        ix = (pts[Ellipsis, 0] >= bounds[0, 0]) & (pts[Ellipsis, 0] < bounds[0, 1])
        iy = (pts[Ellipsis, 1] >= bounds[1, 0]) & (pts[Ellipsis, 1] < bounds[1, 1])
        iz = (pts[Ellipsis, 2] >= bounds[2, 0]) & (pts[Ellipsis, 2] < bounds[2, 1])
        valid = ix & iy & iz

        pts = pts[valid]
        if rgb is not None:
            rgb = rgb[valid]
        if segs is not None:
            segs = segs[valid]

        return pts, rgb, segs

    def find(self, object_name, object_description):
        print(f"finding object: {object_name}, description: {object_description}")

        obj_ids = list(self.object_dicts.keys())

        txts = []
        n = len(obj_ids)
        for oid in obj_ids:
            name = self.object_dicts[oid]["used_name"]
            descr = self.object_dicts[oid]["desc"]
            txts.append(name + " that " + descr)

        prompt = object_name + " described by " + object_description
        txts.append(prompt)

        embeddings = self.clip.get_text_embeddings(txts)

        possibilities = embeddings[:n]
        mat1 = embeddings[n:]

        prob = mat1 @ possibilities.T
        idx = np.argmax(prob[0])

        print("Chosen object: ", obj_ids[idx])
        return obj_ids[idx]

    def pick(self, obj_id):
        print(f"Picking object {obj_id}")

    def place(self, obj_id, position):
        print(f"Placing object {obj_id} at {position}")

    def get_location(self, obj_id):
        print(f"Getting location for object {obj_id}")
        return np.zeros(3)

    def learn_skill(self, skill_name, skill_inputs):
        def random():
            print(f"New skill: {skill_name}")

        return random

    def no_action(self):
        print("Ending ...")

    def get_primitives(self):
        self.primitives = {
            "find": {
                "fn": self.find,
                "description": """
find(object_name, object_description)
    Finds an object with the name `object_name`, and additional information about the object given by `object_description`.
    Arguments:
        object_name: str
            Name of the object to find
        object_description: str
            Description of the object to find

    Returns: int
        object_id , an integer representing the found object
""",
            },
            "pick": {
                "fn": self.pick,
                "description": """
pick(object_id)
    Picks up an object that `object_id` represents.
    Arguments:
        object_id: int
            Id of the object to pick
    Returns: None
""",
            },
            "place": {
                "fn": self.place,
                "description": """
place(object_id, position)
    Places the object that `object_id` represents, at the location given by `position`
    Arguments:
        object_id: int
            Id of the object to place
        position: 3D array
            the place location
    Returns: None
""",
            },
            "get_location": {
                "fn": self.get_location,
                "description": """
get_location(object_id)
    gives the location of the object `object_id`
    Arguments:
        object_id: int
            Id of the object
    Returns:
        position: 3D array
            the location of the object
""",
            },
            "no_action": {
                "fn": self.no_action,
                "description": """
no_action()
    The function marks the end of a program.
    Returns: None
""",
            },
        }

        fn_lst = list(self.primitives.keys())
        self.primitives_lst = ",".join([f"`{fn_name}`" for fn_name in fn_lst])
        self.primitives_description = "\n".join(
            [self.primitives[fn]["description"] for fn in fn_lst]
        )
        self.primitives_description = "```\n" + self.primitives_description + "\n```"

        return self.primitives


if __name__ == "__main__":
    # scene_dir = os.path.join(DATA_DIR, "20221010_1759")

    robot = RealRobot(gui=False, scene_dir=None, sam=True, clip=True, cam_idx=[0, 1, 2, 3])
    obs = robot.get_obs()

    pred_lst, names = get_detic_predictions(
        obs["colors"], vocabulary="custom", custom_vocabulary="mug,tray"
    )

    for idx, rgb in enumerate(obs["colors"]):
        print("Obtaining segmentation mask")
        pred_boxes, pred_labels = get_bb_labels(pred_lst[idx], names)
        masks = get_segmentation_mask(robot.sam_predictor, rgb, pred_boxes, prefix=idx)

    ###################### combined pcd visualization
    # combined_pts, combined_rgb = robot.get_combined_pcd(obs["colors"], obs["depths"], idx=[0, 1, 3])
    # combined_pts, combined_rgb = robot.crop_pcd(combined_pts, combined_rgb)
    # viz = VizServer()
    # viz.view_pcd(combined_pts, combined_rgb)
