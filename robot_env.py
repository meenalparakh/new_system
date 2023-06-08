from airobot import Robot
from airobot import log_warn
from airobot.utils.common import euler2quat
from airobot.sensor.camera.rgbdcam_pybullet import RGBDCameraPybullet
from yacs.config import CfgNode as CN
from copy import deepcopy
from tasks import task_lst as TASK_LST
import open3d
import numpy as np
import distinctipy as dp
from heuristics import get_relation
from scene_description_utils import (
    construct_graph,
    graph_traversal,
    text_description,
    get_place_description,
    get_direction_label,
)
from grasping.eval import initialize_net, cgn_infer
from magnetic_gripper import MagneticGripper
from scipy.spatial.transform import Rotation as R
from grasp import control_robot
from visualize_pcd import VizServer
import cv2
import random
from skill_learner import ask_for_skill
from matplotlib import pyplot as plt


"""
TODO:
1. copy over update pick function, and planner and execute functions, 
    and test them for simulation
2. create demos (may be possible now?)
3. fix the place function
4. fix the find function, also in primitives description
5. copy the templates
6. copy the API key
"""

OPP_RELATIONS = {"above": "below", "contained_in": "contains"}

np.random.seed(0)


def pix_to_xyz(pixel, height, bounds, pixel_size, skip_height=False):
    """Convert from pixel location on heightmap to 3D position."""
    u, v = pixel
    x = bounds[0, 0] + v * pixel_size
    y = bounds[1, 0] + u * pixel_size
    if not skip_height:
        z = bounds[2, 0] + height[u, v]
    else:
        z = 0.0
    return [x, y, z]


def xyz_to_pix(xyz, bounds, pixel_size):
    """Convert from pixel location on heightmap to 3D position."""
    # u, v = pixel
    # x = bounds[0, 0] + v * pixel_size
    # y = bounds[1, 0] + u * pixel_size

    x, y, z = xyz
    v = int((x - bounds[0, 0]) / pixel_size)
    u = int((y - bounds[1, 0]) / pixel_size)

    return u, v


def draw_gaussian(mat, pix):
    highest = 10
    dim = 15
    half = dim // 2
    sigma = 4

    x, y = np.meshgrid(range(dim), range(dim))
    distance = (np.square(x - half) + np.square(y - half)) / sigma**2
    distance = highest * np.exp(-distance)
    v, u = pix
    mat[u - half : u + half + 1, v - half : v + half + 1] = distance

    return mat


def clean_object_pcds(object_dicts):
    for obj_id in object_dicts:
        pcd = object_dicts[obj_id]["pcd"]
        pointcloud = open3d.geometry.PointCloud(open3d.utility.Vector3dVector(pcd))
        downsampled = pointcloud.voxel_down_sample(voxel_size=0.001)
        pointcloud, _ = downsampled.remove_statistical_outlier(
            nb_neighbors=20, std_ratio=1.0
        )

        object_dicts[obj_id]["pcd"] = np.asarray(pointcloud.points)

    return object_dicts


def get_bb(mask):
    row_projection = np.sum(mask, axis=1)
    col_projection = np.sum(mask, axis=0)
    row_idx = np.nonzero(row_projection)[0]
    col_idx = np.nonzero(col_projection)[0]
    min_row = np.min(row_idx)
    max_row = np.max(row_idx)
    min_col = np.min(col_idx)
    max_col = np.max(col_idx)
    return (min_row, min_col, max_row, max_col)


def combined_variance(xyz1, xyz2):
    std1 = np.std(xyz1, axis=0)
    std2 = np.std(xyz2, axis=0)
    std3 = np.std(np.vstack((xyz1, xyz2)), axis=0)
    combined = np.sqrt(
        (len(xyz1) * np.square(std1) + len(xyz2) * np.square(std2))
        / (len(xyz1) + len(xyz2))
    )
    # print("1, 2, avg(1, 2), 3", std1, std2, combined, std3)
    difference = std3 - combined
    difference[2] = 2 * difference[2]
    # print("new std", difference, np.linalg.norm(difference, ord=1))
    return np.linalg.norm(difference, ord=1)


def visualize_pcd(xyz, colors=None):
    pcd = open3d.geometry.PointCloud(open3d.utility.Vector3dVector(xyz))
    if colors is not None:
        if np.max(colors) > 10:
            colors = colors / 255.0
        pcd.colors = open3d.utility.Vector3dVector(colors)

    open3d.visualization.draw_geometries([pcd])

from airobot.cfgs.assets.robotiq2f140 import get_robotiq2f140_cfg

class MyRobot(Robot):
    def __init__(
        self,
        gui=False,
        grasper=False,
        magnetic_gripper=False,
        clip=False,
        meshcat_viz=True,
        device=None,
        skill_learner=False,
    ):
        super().__init__(
            "franka",
            pb_cfg={"gui": gui},
            # eetool_cfg=get_robotiq2f140_cfg()
            # use_arm=False,
            # use_eetool=False,
            # use_base=False,
        )
        self.table_bounds = np.array([[0.2, 1.0], [-0.4, 0.4], [-0.01, 0.5]])

        if grasper:
            self.grasper, _, self.grasper_config = initialize_net(
                config_file="./grasping/model/",
                load_model=True,
                save_path="./grasping/checkpoints/current.pth",
                args=None,
                device=device,
            )
            if magnetic_gripper:
                self.gripper = MagneticGripper(self)
            else:
                self.gripper = None

        if clip:
            from clip_model import MyCLIP

            self.clip = MyCLIP(device=device)

        if meshcat_viz:
            self.viz = VizServer()
        else:
            self.viz = None

        self.skill_learner = skill_learner

        # dict that contains object_ids that have moved and their approximate new locations
        # self.objects_moved = {}

        self.feedback_queue = []
        self.new_task = True
        self.load_primitives()

        self.home_ee_pose = np.array([[1, 0, 0, 0.20],
                                      [0, -1, 0, 0.0],
                                      [0, 0, -1, 1.35],
                                      [0, 0, 0, 1.]])
        self.action_ee_pose = np.array([[1, 0, 0, 0.40],
                                      [0, -1, 0, 0.0],
                                      [0, 0, -1, 1.40],
                                      [0, 0, 0, 1.]])
        
        success = self.arm.go_home()
        if not success:
            log_warn("Robot go_home failed!!!")

        self.object_dicts = {}
        self.sim_dict = {"object_dicts": {}}

        # setup table
        ori = euler2quat([0, 0, np.pi / 2])
        self.table_id = self.pb_client.load_urdf(
            "table/table.urdf", [0.6, 0, 0.4], ori, scaling=0.9
        )
        print("Table id", self.table_id)
        self.pb_client.changeDynamics(self.table_id, 0, mass=0, lateralFriction=2.0)
        self.table_bounds = np.array([[0.05, 0.95], [-0.5, 0.5], [0.85, 3.0]])

        # setup plane
        # self.plane_id = self.pb_client.load_urdf("plane.urdf")

        # setup camera
        self._setup_cameras()
        self.depth_scale = 1.0

    def start_task(self):
        self.new_task = False

    def end_task(self):
        self.new_task = True

    def get_feedback(self):
        # feedback = " ".join(self.feedback_queue)
        feedback = self.feedback_queue[-1]
        return feedback

        # new_scene_description = self.get_scene_description()
        # return feedback + " " + new_scene_description

    def empty_feedback_queue(self):
        self.feedback_queue = []

    def reset(self, task_name, *args):

        success = self.arm.go_home()
        if not success:
            log_warn("Robot go_home failed!!!")

        # focus_pt = [0, 0, 1]  # ([x, y, z])
        # self.cam.setup_camera(focus_pt=focus_pt, dist=3, yaw=90, pitch=0, roll=0)

        if isinstance(task_name, str):
            self.task = TASK_LST[task_name](self, *args)
        else:
            self.task = task_name(self, *args)  # assuming the class type has been passed
        
        self.task.reset()

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

    def _camera_cfgs(self):
        _C = CN()
        _C.ZNEAR = 0.01
        _C.ZFAR = 10
        _C.WIDTH = 640
        _C.HEIGHT = 480
        _C.FOV = 60
        _ROOT_C = CN()
        _ROOT_C.CAM = CN()
        _ROOT_C.CAM.SIM = _C
        return _ROOT_C.clone()

    def get_obs(self):
        """
        Returns:
            np.ndarray: observations from two cameras.
            The two images are concatenated together.
            The returned observation shape is [2H, W, 3].
        """
        self.arm.set_ee_pose(pos=self.home_ee_pose[:3, 3], ori=self.home_ee_pose[:3, :3])

        all_obj_ids = [v["mask_id"] for k, v in self.sim_dict["object_dicts"].items()]
        first_obj = min(all_obj_ids)
        last_obj = max(all_obj_ids)

        rgbs = []
        depths = []
        segs = []
        for idx, cam in enumerate(self.cams):
            rgb, depth, seg = cam.get_images(get_rgb=True, get_depth=True, get_seg=True)
            rgbs.append(rgb)
            depths.append(depth)

            seg = seg.astype(int)
            seg[seg < first_obj] = -1
            seg[seg > last_obj] = -1
            # print("unique ids", np.unique(seg))
            segs.append(seg)

        # ////////////////////////////////////////////////////////////////////////////
        self.sim_dict["segs"] = segs
        # ////////////////////////////////////////////////////////////////////////////

        # input()

        return {"colors": rgbs, "depths": depths}

    def _setup_cameras(self):
        self.cams = []

        for i in range(4):
            self.cams.append(
                RGBDCameraPybullet(cfgs=self._camera_cfgs(), pb_client=self.pb_client)
            )

        self.cams[0].setup_camera(
            focus_pt=[0.5, 0.0, 1.0], dist=1, yaw=0, pitch=-45, roll=0
        )
        self.cams[1].setup_camera(
            focus_pt=[0.5, 0.0, 1.0], dist=1, yaw=90, pitch=-45, roll=0
        )
        self.cams[2].setup_camera(
            focus_pt=[0.5, 0.0, 1.0], dist=1, yaw=0, pitch=-90, roll=0
        )
        self.cams[3].setup_camera(
            focus_pt=[0.5, 0.0, 1.0], dist=1, yaw=180, pitch=-45, roll=0
        )

    def get_combined_pcd(self, colors, depths, idx=None):
        pcd_pts = []
        pcd_rgb = []
        chosen_cams = self.cams

        if idx is None:
            idx = list(range(len(colors)))

        count = 0
        for color, depth, cam in zip(colors, depths, chosen_cams):
            if count in idx:
                cam_extr = cam.get_cam_ext()
                pts, rgb = self.get_pcd(
                    cam_ext_mat=cam_extr, rgb_image=color, depth_image=depth, cam=cam
                )
                pcd_pts.append(pts)
                pcd_rgb.append(rgb)
            count += 1

        return np.concatenate(pcd_pts, axis=0), np.concatenate(pcd_rgb, axis=0)

    def get_pcd(
        self,
        in_world=True,
        filter_depth=True,
        depth_min=0.20,
        depth_max=1.50,
        cam_ext_mat=None,
        rgb_image=None,
        depth_image=None,
        cam=None,
    ):
        """
        Get the point cloud from the entire depth image
        in the camera frame or in the world frame.
        Returns:
            2-element tuple containing

            - np.ndarray: point coordinates (shape: :math:`[N, 3]`).
            - np.ndarray: rgb values (shape: :math:`[N, 3]`).
        """

        rgb_im = rgb_image
        depth_im = depth_image
        # pcd in camera from depth
        depth = depth_im.reshape(-1) * cam.depth_scale

        rgb = None
        if rgb_im is not None:
            rgb = rgb_im.reshape(-1, 3)
        depth_min = depth_min
        depth_max = depth_max
        if filter_depth:
            valid = depth > depth_min
            valid = np.logical_and(valid, depth < depth_max)
            depth = depth[valid]
            if rgb is not None:
                rgb = rgb[valid]
            uv_one_in_cam = cam._uv_one_in_cam[:, valid]
        else:
            uv_one_in_cam = cam._uv_one_in_cam
        pts_in_cam = np.multiply(uv_one_in_cam, depth)
        if not in_world:
            pcd_pts = pts_in_cam.T
            pcd_rgb = rgb
            return pcd_pts, pcd_rgb
        else:
            if cam.cam_ext_mat is None and cam_ext_mat is None:
                raise ValueError(
                    "Please call set_cam_ext() first to set up"
                    " the camera extrinsic matrix"
                )
            cam_ext_mat = cam.cam_ext_mat if cam_ext_mat is None else cam_ext_mat
            pts_in_cam = np.concatenate(
                (pts_in_cam, np.ones((1, pts_in_cam.shape[1]))), axis=0
            )
            pts_in_world = np.dot(cam_ext_mat, pts_in_cam)
            pcd_pts = pts_in_world[:3, :].T
            pcd_rgb = rgb

            return pcd_pts, pcd_rgb

    def update_dict_util(self, original_dict, new_dict):
        """
        generate a new object dict and do the matching with the old objects
        mainly clip crop based matching
        run object detection, and segmentation and remaps the object ids
        with the new pcds and new scene descriptors
        """

        reset = False
        new_obj_lst = list(new_dict.keys())
        cur_obj_lst = list(original_dict.keys())

        if len(new_obj_lst) != len(cur_obj_lst):
            print("number of objects in the scene are different, resetting the dict")
            reset = True

        else:
            new_obj_centers = [new_dict[oid]["pcd"].mean(axis=0) for oid in new_obj_lst]
            new_obj_centers = np.array(new_obj_centers)

            possible_matchings = []
            matched_new = []
            matched_old = []
            not_matched = []
            for oid, info in original_dict.items():
                pcd_val = info["pcd"]
                try:
                    center = np.mean(pcd_val, axis=0)
                    idx = np.argmin(np.linalg.norm(new_obj_centers - center, axis=1))
                    nid = new_obj_lst[idx]
                    possible_matchings.append((oid, nid))
                    matched_new.append(nid)
                    matched_old.append(oid)
                except:
                    # continue
                    not_matched.append(oid)

            if len(matched_new) == len(np.unique(matched_new)):
                print("unique matching found")
                k = list(set(new_obj_lst).difference(set(matched_new)))
                possible_matchings.append((not_matched[0], k[0]))
                for cid, nid in possible_matchings:
                    original_dict[cid]["pcd"] = new_dict[nid]["pcd"]
                    # original_dict[cid]["relation"] = new_dict[nid]["relation"]

                description, original_dict = self.get_scene_description(original_dict, change_uname=False)

            else:
                print("Objects could not be uniquely matched. Resetting the dict")
                reset = True

        if reset:
            original_dict = new_dict
            description, original_dict = self.get_scene_description(original_dict)

        print(description)

        # return original_dict

        return description, original_dict
    
    def move_to_home(self):
        cur_pos, cur_quat, _, cur_euler = self.arm.get_ee_pose()
        delta_pos = np.array(self.action_ee_pose[:3, 3]) - np.array(cur_pos)
        self.arm.move_ee_xyz(delta_pos)
        self.arm.set_ee_pose()
        self.arm.set_ee_pose(pos=self.action_ee_pose[:3, 3], ori=self.action_ee_pose[:3, :3])


    def update_dicts(self):
        """
        generate a new object dict and do the matching with the old objects
        mainly clip crop based matching
        run object detection, and segmentation and remaps the object ids
        with the new pcds and new scene descriptors
        """
        new_dicts = self.get_object_dicts()
        current_dict = self.object_dicts

        description, final_dict = self.update_dict_util(current_dict, new_dicts)
        self.object_dicts = final_dict
        return description

    def visualize_object_dicts(self, object_dict, bg=False):
        print("Visualizing object_dicts:")
        # self.viz.mc_vis["scene"].delete()
        for obj_id in object_dict:
            pcd = object_dict[obj_id]["pcd"]
            rgb = object_dict[obj_id]["rgb"]
            name = object_dict[obj_id]["label"][0]
            print(f"\t{name}: pcd length: {len(pcd)}")
            self.viz.view_pcd(pcd, rgb, name=f"{obj_id}_{name}")

        if bg:
            self.viz.view_pcd(self.bg_pcd, name=f"background")

    def get_object_dicts(self):
        obs = self.get_obs()

        # //////////////////////////////////////////////////////////////////////////////
        # Labelling + segmentation + description
        # //////////////////////////////////////////////////////////////////////////////

        segs, info_dict = self.get_segment_labels_and_embeddings(
            obs["colors"], obs["depths"], self.clip
        )

        object_dicts = self.get_segmented_pcd(
            obs["colors"],
            obs["depths"],
            segs,
            remove_floor_ht=1.0,
            std_threshold=0.02,
            label_infos=info_dict,
            visualization=True,
            process_pcd_fn=self.crop_pcd,
        )

        return object_dicts

    def init_dicts(self, object_dicts):
        self.object_dicts = object_dicts

    def print_object_dicts(
        self, object_dict, ks=["label", "relation", "desc", "used_name"]
    ):
        for id, info in object_dict.items():
            print("Object id", id)
            for k, v in info.items():
                if k in ks:
                    print(f"    {k}: {v}")
            print()

    def get_segment_labels_and_embeddings(self, colors, depths, clip_):
        # for each rgb image, returns a segmented image, and
        # a dict containing the labels for each segment id
        # the segments are numbered taking into account different
        # instances of the category as well.

        # ////////////////////////////////////////////////////////////////
        # import cv2
        segs = self.sim_dict["segs"]

        image_embeddings = []

        # count = 0
        for s, c in zip(segs, colors):
            unique_ids = np.unique(s.astype(int))
            print("unique ids in segmentation mask", unique_ids)
            unique_ids = unique_ids[unique_ids >= 0]
            image_crops = []
            for uid in unique_ids:
                mask = s == uid
                bb = get_bb(mask)
                r1, c1, r2, c2 = bb
                # print(r1, c1, r2, c2)
                crop = c[r1 : r2 + 1, c1 : c2 + 1, :]

                # print("crop max", np.max(crop))
                # cv2.imwrite(f"crop{count}.png", crop.astype(np.uint8))
                # count += 1
                # cv2.imshow("crops", crop)
                # cv2.waitKey(0)

                image_crops.append(crop)

            embeddings = clip_.get_image_embeddings(image_crops)
            embedding_dict = {
                unique_ids[idx]: {"embedding": embeddings[idx]}
                for idx in range(len(unique_ids))
            }
            image_embeddings.append(embedding_dict)

        label_dict = {}
        for _, obj_info in self.sim_dict["object_dicts"].items():
            label_dict[obj_info["mask_id"]] = obj_info["name"]

        for d in image_embeddings:
            for mask_id in d:
                if mask_id in label_dict:
                    d[mask_id]["label"] = label_dict[mask_id]
                else:
                    d[mask_id]["label"] = "unknown"

        for d in image_embeddings:
            for mask_id in d:
                print(mask_id, ":", d[mask_id]["label"])
        # ////////////////////////////////////////////////////////////////
        return segs, image_embeddings

    def get_segmented_pcd(
        self,
        colors,
        depths,
        segs,
        remove_floor_ht=0.90,
        std_threshold=0.03,
        label_infos=None,
        visualization=False,
        process_pcd_fn=None,
        outlier_removal=False,
    ):
        pcd_pts = []
        pcd_rgb = []
        pcd_seg = []
        for color, depth, segment, cam in zip(colors, depths, segs, self.cams):
            cam_extr = cam.get_cam_ext()
            pts, rgb = self.get_pcd(
                cam_ext_mat=cam_extr,
                rgb_image=color,
                depth_image=depth,
                cam=cam,
            )

            _, seg = self.get_pcd(
                cam_ext_mat=cam_extr,
                rgb_image=np.repeat(segment[..., None], repeats=3, axis=2),
                depth_image=depth,
                cam=cam,
            )

            if process_pcd_fn is not None:
                pts, rgb, seg = process_pcd_fn(pts, rgb, seg)

            pcd_pts.append(pts)
            pcd_rgb.append(rgb)
            pcd_seg.append(seg)

            # valid = pts[:, 2] > remove_floor_ht
            # pcd_pts.append(pts[valid])
            # pcd_rgb.append(rgb[valid])
            # pcd_seg.append(seg[valid])

            # valid = pts[:, 2] > remove_floor_ht
            # pcd_pts.append(pts[valid])
            # pcd_rgb.append(rgb[valid])
            # pcd_seg.append(seg[valid])

        # //////////////////////////////////////////////////////////////////////////////
        # Collective visualization of segments under consideration
        # //////////////////////////////////////////////////////////////////////////////

        # cum_pcds = []
        # cum_rgbs = []

        # for idx in range(len(pcd_pts)):
        #     seg = pcd_seg[idx][:, 0]
        #     unique_ids = list(np.unique(seg.astype(int)))
        #     for uid in unique_ids:
        #         valid = (seg == uid)
        #         cum_pcds.append(pcd_pts[idx][valid])
        #         cum_rgbs.append(pcd_rgb[idx][valid])

        # open3d_pcds = []
        # uniform_colors = dp.get_colors(len(cum_pcds))
        # for idx, pcd in enumerate(cum_pcds):
        #     print(idx, pcd.shape, "printing pcd length")
        #     point_cloud = open3d.geometry.PointCloud(open3d.utility.Vector3dVector(pcd))
        #     point_cloud.paint_uniform_color(uniform_colors[idx])
        #     open3d_pcds.append(point_cloud)

        # open3d.visualization.draw_geometries(open3d_pcds)

        # //////////////////////////////////////////////////////////////////////////////

        objects = {}

        start_idx = 0
        print("\nProcessing Image index: 0")
        for j in range(len(pcd_pts)):
            seg = pcd_seg[j][:, 0]
            unique_ids = np.unique(seg.astype(int))
            # print("unique_ids in first image", unique_ids)

            unique_ids = unique_ids[unique_ids >= 0]

            # print("unique_ids in first image", unique_ids)

            if len(unique_ids) > 0:
                start_idx = j

                for uid in unique_ids:
                    valid = seg == uid
                    objects[uid] = {
                        "pcd": pcd_pts[j][valid],
                        "rgb": pcd_rgb[j][valid],
                        "label": [label_infos[j][uid]["label"]],
                        "embed": [label_infos[j][uid]["embedding"]],
                    }

                new_uids_count = np.max(unique_ids) + 1
                print(
                    "\tThis image contained the following categories and have been initialized:",
                    [i["label"][0] for _, i in objects.items()],
                )

                break

        print(
            "\tIn processing the next images, any particular category will either\n"
            f"\tbe updated if the inverse_similarity score is less than {std_threshold}, or will be\n"
            "\tadded as a new category if inverser_similarity score is greater than threshold"
        )

        for idx in range(start_idx + 1, len(pcd_pts)):
            seg = pcd_seg[idx][:, 0]
            unique_ids = np.array(np.unique(seg.astype(int)))
            # print("unique_ids in image", idx, unique_ids)

            unique_ids = unique_ids[unique_ids >= 0]

            # print("unique_ids in image", idx, unique_ids)
            obj_lst = list(objects.keys())
            new_dict = {}
            print(
                f"\nProcessing image index: {idx}. The image contains the categories:",
                [label_infos[idx][i]["label"] for i in unique_ids],
            )
            print("\tCurrent Objects", [objects[obj]["label"][0] for obj in obj_lst])
            for uid in unique_ids:
                l = label_infos[idx][uid]["label"]

                # print(uid, l)
                valid = seg == uid
                new_pcd = pcd_pts[idx][valid]
                new_rgb = pcd_rgb[idx][valid]

                diffs = []
                for obj in obj_lst:
                    original_pcd = objects[obj]["pcd"]
                    d = combined_variance(new_pcd, original_pcd)
                    diffs.append(d)

                print(
                    f"\t\t{l}'s inverse similarity score with current objects:", diffs
                )

                # print(diffs)

                min_diff = np.min(diffs)
                if min_diff < std_threshold:
                    print(
                        "\t\t\tMatch with current object found! Updating the current pcd."
                    )
                    l = label_infos[idx][uid]["label"]
                    obj_id = obj_lst[np.argmin(diffs)]
                    p = objects[obj_id]["pcd"]
                    r = objects[obj_id]["rgb"]
                    objects[obj_id]["pcd"] = np.vstack((p, new_pcd))
                    objects[obj_id]["rgb"] = np.vstack((r, new_rgb))

                    # print_object_dicts(objects)
                    objects[obj_id]["label"].append(l)
                    objects[obj_id]["embed"].append(label_infos[idx][uid]["embedding"])

                else:
                    print(
                        "\t\t\tNo match with current objects! Initializing a new category."
                    )
                    new_dict[new_uids_count] = {
                        "pcd": new_pcd,
                        "rgb": new_rgb,
                        "label": [l],
                        "embed": [label_infos[idx][uid]["embedding"]],
                    }
                    new_uids_count += 1
            objects.update(new_dict)

        extra_pcd = []
        for idx in range(len(pcd_pts)):
            seg_mask = pcd_seg[idx][:, 0] < 0
            extra_pcd.append(pcd_pts[idx][seg_mask])
        extra_pcd = np.concatenate(extra_pcd, axis=0)

        self.bg_pcd = extra_pcd

        if outlier_removal:
            objects = clean_object_pcds(objects)

        return objects

    def get_current_scene_description(self):
        description, _ = self.get_scene_description(
            self.object_dicts, change_uname=False
        )

        self.feedback_queue.append(description)
        return description

    def get_scene_description(self, object_dicts, change_uname=True):
        obj_ids_lst = list(object_dicts.keys())

        # ////////////////////////////////////////////////////////////////////////
        # finding object relations
        # ////////////////////////////////////////////////////////////////////////

        for obj_id in obj_ids_lst:
            object_dicts[obj_id]["relation"] = {
                "contains": [],
                "below": [],
                "above": [],
                "contained_in": [],
            }

        for obj1 in obj_ids_lst:
            for obj2 in obj_ids_lst:
                if obj1 == obj2:
                    continue
                print("Anchor:", obj1, "label:", object_dicts[obj1]["label"])
                print("Moving:", obj2, "label:", object_dicts[obj2]["label"])
                relation = get_relation(
                    object_dicts[obj1]["pcd"], object_dicts[obj2]["pcd"]
                )
                if relation in ["above", "contained_in"]:
                    object_dicts[obj1]["relation"][OPP_RELATIONS[relation]].append(obj2)
                    object_dicts[obj2]["relation"][relation].append(obj1)

        # return "", object_dicts

        object_dicts, new_obj_lst = construct_graph(object_dicts)
        side = "right"
        traversal_order, path = graph_traversal(object_dicts, new_obj_lst, side=side)
        description = text_description(
            object_dicts, traversal_order, path, side=side, change_uname=change_uname
        )

        return description, object_dicts

    def get_grasp(self, obj_id, threshold=0.85, add_floor=None, visualize=False):
        combined_pcds = []
        combined_masks = []

        for oid in self.object_dicts:
            # if oid in self.objects_moved:
            #     pcd = self.object_dicts[oid]["pcd"][self.objects_moved[oid]]
            # else:
            pcd = self.object_dicts[oid]["pcd"]
            mask = np.ones(len(pcd)) if oid == obj_id else np.zeros(len(pcd))

            combined_pcds.append(pcd)
            combined_masks.append(mask)

        final_pcd = np.concatenate(combined_pcds, axis=0)
        final_mask = np.concatenate(combined_masks, axis=0).reshape((-1, 1))

        if add_floor is not None:
            final_pcd = np.concatenate((final_pcd, add_floor), axis=0)
            final_mask = np.concatenate(
                (final_mask, np.zeros((len(add_floor), 1))), axis=0
            )

        # visualize_pcd(final_pcd)

        ht = np.min(final_pcd[:, 2])
        x, y = np.mean(final_pcd[:, :2], axis=0)

        translation = np.array([x, y, ht])
        final_pcd = final_pcd - translation

        assert len(final_mask) == len(final_pcd)

        pred_grasps, pred_success, _ = cgn_infer(
            self.grasper, final_pcd, final_mask, threshold=threshold
        )
        # print(len(pred_grasps), pred_grasps.shape, "<= these are the grasps")

        n = 0 if pred_success is None else pred_grasps.shape[0]
        print("model pass.", n, "grasps found.")
        if n == 0:
            return None, None

        pred_grasps[:, :3, 3] = pred_grasps[:, :3, 3] + translation

        if visualize and self.viz:
            grasp_idx = np.argmax(pred_success)
            grasp_pose = pred_grasps[grasp_idx : grasp_idx + 1]
            self.viz.view_grasps(grasp_pose)

        return pred_grasps, pred_success

    def pick(self, obj_id, visualize=False, grasp_pose=None):
        # self.obs_lst.append(self.get_obs())
        # self.arm.go_home()
        self.move_to_home()

        if grasp_pose is None:
            pred_grasps, pred_success = self.get_grasp(
                obj_id, threshold=0.8, add_floor=self.bg_pcd, visualize=visualize
            )

            if pred_grasps is None:
                print("Again, no grasps found. Need help with completing the pick.")
                pos, ori = self.pb_client.getBasePositionAndOrientation(obj_id)
                self.pb_client.resetBasePositionAndOrientation(
                    obj_id, [0.5, -0.6, 0.5], ori
                )
                for _ in range(100):
                    self.pb_client.stepSimulation()
                print("Teleported in simulator")
                return

            # grasp_idx = random.choice(range(len(grasps)))

            grasp_idx = np.argmax(pred_success)
            grasp_pose = pred_grasps[grasp_idx]

        self.predicted_pose = (obj_id, grasp_pose)
        pick_pose_mat = self.pick_given_pose(grasp_pose)
        # self.arm.go_home()


        pcd = self.object_dicts[obj_id]["pcd"]
        color = self.object_dicts[obj_id]["rgb"]
        object_pose = np.eye(4)
        object_pose[:3, 3] = np.mean(pcd, axis=0)
        self.picked_pose = np.linalg.inv(object_pose) @ pick_pose_mat
        

        name = self.object_dicts[obj_id]["used_name"]
        # self.object_dicts[obj_id]["pcd"] = "in_air"
        print(
            "Warning: the feedback is not actually checking the pick success. Make it conditioned"
        )

        self.feedback_queue.append(f"{name} was picked successfullly.")

    def pick_given_pose(self, pick_pose, translate=0.13):
        self.gripper.release()

        z_rot = np.eye(4)
        z_rot[2, 3] = translate
        z_rot[:3, :3] = R.from_euler("z", np.pi / 2).as_matrix()
        gripper_pose = np.matmul(pick_pose, z_rot)

        rotation = gripper_pose[:3, :3]
        direction_vector = gripper_pose[:3, 2]
        pick_position = gripper_pose[:3, 3]
        # pick_position[2] = pick_position[2] - 0.10
        pose = [pick_position, rotation, direction_vector]

        pose_mat = control_robot(
            self,
            pose,
            robot_category="franka",
            control_mode="linear",
            move_up=0.01,
            linear_offset=-0.01,
        )
        # self.obs_lst.append(self.get_obs())

        if self.gripper:
            self.gripper.activate()

        self.arm.move_ee_xyz([0, 0, 0.20])

        # self.obs_lst.append(self.get_obs())

        print("Pick completed")
        return pose_mat

    def move_arm(self, pos1, pos2=None):
        # move to pos1
        current_position = self.arm.get_ee_pose()[0]
        direction = np.array(pos1) - current_position
        direction[2] = 0
        self.arm.move_ee_xyz(direction)

        # # move to pos2
        # current_position = self.arm.get_ee_pose()[0]
        # direction = np.array(pos2) - current_position
        # direction[2] = 0
        # self.arm.move_ee_xyz(direction)

    def reorient_object(self, obj_id, angle):
        # move to pos1
        self.arm.rot_ee_xyz(angle, "x")

    def get_segmap(
        self, object_dicts=None, to_ignore=[], bounds=None, pixel_size=0.003125
    ):
        if object_dicts is None:
            object_dicts = self.object_dicts

        if bounds is None:
            bounds = self.table_bounds

        pcds, segs = [], []

        for oid in object_dicts:
            assert oid > 0, "error: the object ids start with zero?"

            if oid in to_ignore:
                continue
            p = object_dicts[oid]["pcd"]
            m = np.ones(len(p)) * oid

            pcds.append(p)
            segs.append(m)

        pcds = np.concatenate(pcds, axis=0)
        segs = np.concatenate(segs, axis=0)
        # segs = segs[..., None]

        width = int(np.round((bounds[0, 1] - bounds[0, 0]) / pixel_size))
        height = int(np.round((bounds[1, 1] - bounds[1, 0]) / pixel_size))

        heightmap = np.zeros((height, width), dtype=np.float32)
        segmap = np.zeros((height, width), dtype=int)

        # Filter out 3D points that are outside of the predefined bounds.
        ix = (pcds[Ellipsis, 0] >= bounds[0, 0]) & (pcds[Ellipsis, 0] < bounds[0, 1])
        iy = (pcds[Ellipsis, 1] >= bounds[1, 0]) & (pcds[Ellipsis, 1] < bounds[1, 1])
        iz = (pcds[Ellipsis, 2] >= bounds[2, 0]) & (pcds[Ellipsis, 2] < bounds[2, 1])

        valid = ix & iy & iz

        # checking that all the pcds lie inside table bounds
        if np.sum(1 - valid) != 0:
            print(
                "Warning: not all pcd boints lie inside table bounds!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            )

        pcds = pcds[valid]
        segs = segs[valid]

        # Sort 3D points by z-value, which works with array assignment to simulate
        # z-buffering for rendering the heightmap image.
        iz = np.argsort(pcds[:, -1])
        pcds, segs = pcds[iz], segs[iz]
        px = np.int32(np.floor((pcds[:, 0] - bounds[0, 0]) / pixel_size))
        py = np.int32(np.floor((pcds[:, 1] - bounds[1, 0]) / pixel_size))
        px = np.clip(px, 0, width - 1)
        py = np.clip(py, 0, height - 1)
        heightmap[py, px] = pcds[:, 2] - bounds[2, 0]
        segmap[py, px] = segs

        heightmap = heightmap.T
        segmap = segmap.T

        return heightmap, segmap

    def get_segmap(
        self, object_dicts=None, to_ignore=[], bounds=None, pixel_size=0.003125
    ):
        if object_dicts is None:
            object_dicts = self.object_dicts

        if bounds is None:
            bounds = self.table_bounds

        pcds, segs = [], []
        colors = []

        for oid in object_dicts:
            assert oid > 0, "error: the object ids start with zero?"

            if oid in to_ignore:
                continue
            p = object_dicts[oid]["pcd"]
            m = np.ones(len(p)) * oid

            pcds.append(p)
            segs.append(m)
            colors.append(object_dicts[oid]["rgb"])

        pcds = np.concatenate(pcds, axis=0)
        segs = np.concatenate(segs, axis=0)
        colors = np.concatenate(colors, axis=0)
        # segs = segs[..., None]

        width = int(np.round((bounds[0, 1] - bounds[0, 0]) / pixel_size))
        height = int(np.round((bounds[1, 1] - bounds[1, 0]) / pixel_size))

        heightmap = np.zeros((height, width), dtype=np.float32)
        segmap = np.zeros((height, width), dtype=int)

        colormap = np.zeros((height, width, 3))

        # Filter out 3D points that are outside of the predefined bounds.
        ix = (pcds[Ellipsis, 0] >= bounds[0, 0]) & (pcds[Ellipsis, 0] < bounds[0, 1])
        iy = (pcds[Ellipsis, 1] >= bounds[1, 0]) & (pcds[Ellipsis, 1] < bounds[1, 1])
        iz = (pcds[Ellipsis, 2] >= bounds[2, 0]) & (pcds[Ellipsis, 2] < bounds[2, 1])

        valid = ix & iy & iz

        # checking that all the pcds lie inside table bounds
        if np.sum(1 - valid) != 0:
            print(
                "Warning: not all pcd boints lie inside table bounds!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            )

        pcds = pcds[valid]
        segs = segs[valid]
        colors = colors[valid, :]

        # Sort 3D points by z-value, which works with array assignment to simulate
        # z-buffering for rendering the heightmap image.
        iz = np.argsort(pcds[:, -1])
        pcds, segs = pcds[iz], segs[iz]
        px = np.int32(np.floor((pcds[:, 0] - bounds[0, 0]) / pixel_size))
        py = np.int32(np.floor((pcds[:, 1] - bounds[1, 0]) / pixel_size))
        px = np.clip(px, 0, width - 1)
        py = np.clip(py, 0, height - 1)
        heightmap[py, px] = pcds[:, 2] - bounds[2, 0]
        segmap[py, px] = segs
        colormap[py, px, :] = colors

        heightmap = heightmap.T
        segmap = segmap.T
        colormap = colormap.T
        colormap = np.moveaxis(colormap, 0, -1)

        plt.imsave("colormap.png", colormap/255.0)

        return heightmap, segmap

    def get_place_position_new(
        self, object_id, reference_object_id, place_description, current_image=None
    ):
        info = self.object_dicts[reference_object_id]
        target_description = place_description + " of " + info["used_name"]

        place_locations = {}
        obj_id = object_id

        # get place positions lying above reference
        ht = np.max(info["pcd"][:, 2])
        place_pos = info["object_center"]
        desc = "above " + info["used_name"]
        place_locations[desc] = [[*(place_pos[:2]), ht]]

        # get top down segmap exclusing the obj_id
        hmap, segmap = self.get_segmap(self.object_dicts, to_ignore=[obj_id])

        # sample random points in open spacess
        empty_space = segmap == 0
        cv2_array = (empty_space * 255).astype(np.uint8)
        radius = 10
        kernel = np.ones((radius, radius), np.uint8)
        empty_space = cv2.erode(cv2_array, kernel, iterations=1)

        mask = np.zeros_like(empty_space)
        mask[radius:-radius, radius:-radius] = 255
        empty_space = empty_space & mask
        empty_space = np.where(empty_space > 100)

        k = min(len(empty_space[0]), 50)
        random_pts = random.sample(range(len(empty_space[0])), k)
        rows = empty_space[0][random_pts]
        cols = empty_space[1][random_pts]

        obj_lst = list(self.object_dicts.keys())
        obj_lst.remove(obj_id)
        # find their descriptions
        for r, c in zip(rows, cols):
            xyz = pix_to_xyz(
                (c, r), 0, self.table_bounds, pixel_size=0.003125, skip_height=True
            )
            # segmap[r, c] = 10
            ht = hmap[r, c]
            xyz[2] = ht

            obj_pixel_center = info["object_center"]
            direction_vector = xyz[:2] - obj_pixel_center[:2]
            labels = get_direction_label(direction_vector, info["used_name"])
            label = labels[0]
            if label in place_locations:
                place_locations[label].append(xyz)
            else:
                place_locations[label] = [xyz]

            # plt.imsave("segmap.png", segmap)

        descs = list(place_locations.keys())
        pixs = []
        if current_image is None:
            current_image = segmap

        for d in descs:
            location = place_locations[d][0]
            pix = xyz_to_pix(location, self.table_bounds, 0.003125)
            pixs.append(pix)
            current_image = draw_gaussian(current_image, pix)

        plt.imsave("segmap.png", current_image)
        current_image = cv2.imread("segmap.png")

        # count = 1
        # for pix, d in zip(pixs, descs):
        #     print("max val", np.max(current_image))
        #     current_image = cv2.putText(
        #         current_image,
        #         str(count),
        #         pix,
        #         cv2.FONT_HERSHEY_SIMPLEX,
        #         0.4,
        #         (240, 240, 240),
        #         1,
        #         cv2.LINE_AA
        #     )
        #     count += 1
        # cv2.imwrite("segmap.png", current_image)
        # print(descs)

        print(
            "possible place locations for",
            obj_id,
            self.object_dicts[obj_id]["used_name"],
            place_locations,
        )

        n = len(place_locations)
        txts = list(place_locations.keys())
        txts.append(target_description)

        print(txts)
        embeddings = self.clip.get_bert_embeddings(txts)

        possibilities = embeddings[:n]
        mat1 = embeddings[n:]

        prob = mat1 @ possibilities.T
        print(prob)
        idx = np.argmax(prob[0])

        print("Chosen location: ", txts[idx])
        location = random.choice(place_locations[txts[idx]])

        # return the location that is above certain height to account for place object's height
        # location[2] = location[2]
        print("coordinates:", location)
        return location

    def get_place_position(self, obj_id, place_description):
        place_locations = {}

        # get place positions lying above other objects
        for oid, info in self.object_dicts.items():
            if oid == obj_id:
                continue
            place_pos = info["object_center"]
            desc = "over " + info["used_name"]
            place_locations[desc] = [place_pos[:2]]

        # get top down segmap exclusing the obj_id
        _, segmap = self.get_segmap(self.object_dicts, to_ignore=[obj_id])

        # sample random points in open spacess
        empty_space = segmap == 0
        cv2_array = (empty_space * 255).astype(np.uint8)
        radius = 50
        kernel = np.ones((radius, radius), np.uint8)
        empty_space = cv2.erode(cv2_array, kernel, iterations=1)

        mask = np.zeros_like(empty_space)
        mask[radius:-radius, radius:-radius] = 255
        empty_space = empty_space & mask
        empty_space = np.where(empty_space > 100)

        k = min(len(empty_space[0]), 10)
        random_pts = random.sample(range(len(empty_space[0])), k)
        rows = empty_space[0][random_pts]
        cols = empty_space[1][random_pts]

        obj_lst = list(self.object_dicts.keys())
        obj_lst.remove(obj_id)
        # find their descriptions
        for r, c in zip(rows, cols):
            xyz = pix_to_xyz(
                (c, r), 0, self.table_bounds, pixel_size=0.003125, skip_height=True
            )
            # print("xyz point", xyz[:2])
            label = get_place_description(self.object_dicts, obj_lst, xyz[:2])
            if label in place_locations:
                place_locations[label].append(xyz[:2])
            else:
                place_locations[label] = [xyz[:2]]

        print(
            "possible place locations for",
            obj_id,
            self.object_dicts[obj_id]["used_name"],
            place_locations,
        )

        n = len(place_locations)
        txts = list(place_locations.keys())
        txts.append(place_description)

        print(txts)
        embeddings = self.clip.get_text_embeddings(txts)

        possibilities = embeddings[:n]
        mat1 = embeddings[n:]

        prob = mat1 @ possibilities.T
        print(prob)
        idx = np.argmax(prob[0])

        print("Chosen location: ", txts[idx])
        location = random.choice(place_locations[txts[idx]])
        print("location", location)
        return location

        # find the best match among them with place description given in argumnets

    def place(self, obj_id, position, skip_update=False):

        # self.arm.go_home()
        self.move_to_home()

        if self.picked_pose is None:
            print("It seems the pick failed, so aborting the place")
            self.update_dicts()
            return

        pcd_val = self.object_dicts[obj_id]["pcd"]

        # if isinstance(pcd_val, str) and pcd_val == "in_air":
        #     print("The object is in grasp")
        # else:
        #     print("The intended object needs to be grasped first. Returning")
        #     return

        place_pose_obj = np.eye(4)
        place_pose_obj[:3, 3] = position
        place_pose = place_pose_obj @ self.picked_pose

        position = place_pose[:3, 3]

        if len(position) == 2:
            position = [*position, 0.9]

        current_position = self.arm.get_ee_pose()[0]
        direction = np.array(position) - current_position
        direction[2] = 0
        self.arm.move_ee_xyz(direction)

        preplace_position = self.arm.get_ee_pose()[0]
        success = True
        while success:
            success = self.arm.move_ee_xyz([0, 0, -0.01])

        self.arm.eetool.open()
        if self.gripper:
            self.gripper.release()

        current_rotation = self.arm.get_ee_pose()[2]
        delta = current_rotation @ np.array([[0], [0], [-0.05]])
        self.arm.move_ee_xyz(delta[:, 0])

        self.arm.move_ee_xyz([0, 0, 0.15])
        self.arm.go_home()

        print("place completed")
        self.object_dicts[obj_id]["pcd"] = ("changed", position)
        name = self.object_dicts[obj_id]["used_name"]
        print("Warning: the feedback not linked to successful placement")
        self.feedback_queue.append(f"Placing {name} was successful.")

        if not skip_update:
            self.update_dicts()

        # self.update_dicts()

    def get_location(self, obj_id):
        obj_pcd = self.object_dicts[obj_id]["pcd"]

        position = np.mean(obj_pcd, axis=0)
        print(f"Getting location for object {obj_id}:", position)
        return position

    def get_all_object_ids(self):
        return list(self.object_dicts.keys())

    def find(
        self,
        object_label=None,
        visual_description=None,
        place_description=None,
        object_ids=None,
    ):
        obj_ids = object_ids
        if obj_ids is None:
            obj_ids = self.get_all_object_ids()

        # threshold = 0.6

        print(
            f"finding object: {object_label}, visual_description: {visual_description}, location_description: {place_description}"
        )

        specified_object = None
        if object_label is not None:
            specified_object = object_label
        elif visual_description is not None:
            specified_object = visual_description
        elif place_description is not None:
            specified_object = place_description

        # obj_ids = list(self.system_env.object_dicts.keys())
        n = len(obj_ids)
        txts = []
        image_embeddings = []

        if object_label is not None:
            template = "This is the {}."
            used_names = [
                template.format(self.object_dicts[oid]["used_name"]) for oid in obj_ids
            ]

            used_names.append(template.format(object_label))
            # print("Finding clip embeddings for:", used_names)
            print("Finding bert embeddings for:", used_names)
            # clip_embeddings = self.clip.get_text_embeddings(used_names)
            bert_embeddings = self.clip.get_bert_embeddings(used_names)
            # label_probs = (clip_embeddings[n:] @ clip_embeddings[:n].T)[0]
            label_probs = (bert_embeddings[n:] @ bert_embeddings[:n].T)[0]
            label_threshold = 0.7
            # todo - set separate threshjolds
        else:
            label_probs = np.ones(n, dtype=np.float32)
            label_threshold = 1.0

        if visual_description is not None:
            visual_embeddings = [self.object_dicts[oid]["embed"] for oid in obj_ids]
            visual_embeddings = np.array(
                [np.array(v).mean(axis=0) for v in visual_embeddings]
            )

            print("shape of visual template:", visual_embeddings.shape)
            text_template = "This is an image of {}"
            embedding = self.clip.get_text_embeddings(
                [text_template.format(visual_description)]
            )

            print("shape of visual text embedding", embedding.shape)
            visual_probs = (embedding @ visual_embeddings.T)[0]

            visual_threshold = 0.3
        else:
            visual_probs = np.ones(n, dtype=np.float32)
            visual_threshold = 1.0

        if place_description is not None:
            txts = []
            for oid in obj_ids:
                name = self.object_dicts[oid]["used_name"]
                descr = self.object_dicts[oid]["desc"]
                txts.append(name + " that " + descr)
                # txts.append(name)

            txts.append(place_description)

            print("finding bert embeddings for place location", txts)
            place_embeddings = self.clip.get_bert_embeddings(txts)
            location_probs = (place_embeddings[n:] @ place_embeddings[:n].T)[0]
            place_threshold = 0.6
        else:
            location_probs = np.ones(n, dtype=np.float32)
            place_threshold = 1.0

        print("obj ids", obj_ids)
        print("label probs:", label_probs)
        print("Visual probs:", visual_probs)
        print("Location probs:", location_probs)

        final_threshold = label_threshold * visual_threshold * place_threshold
        similarities = label_probs * visual_probs * location_probs
        best_score = np.max(similarities)
        print("Best similarity score and threshold:", best_score, final_threshold)

        if best_score < final_threshold:
            self.feedback_queue.append(f"{specified_object} not found.")
            return obj_ids[np.argmax(similarities)]

        idx = np.argmax(similarities)
        print("Object found!!")
        print("Chosen object: ", obj_ids[idx])

        # if visual_description is None and place_description is None:
        self.feedback_queue.append(f"{specified_object} found.")

        return obj_ids[idx]

    def no_action(self):
        print("Ending ...")

    def learn_skill(self, skill_name):
        if skill_name in self.primitives:
            print("Skill already exists. returning the existing one.")
            return self.primitives[skill_name]["fn"]

        print("Asking to learn the skill:", skill_name)
        fn = ask_for_skill(skill_name)

        def new_skill(obj_id):
            pcd = self.object_dicts[obj_id]["pcd"]
            fn(pcd)

        setattr(self, skill_name, new_skill)

        self.primitives[skill_name] = {
            "fn": self.new_skill,
            "description": f"""
{skill_name}(object_id_subject, object_id_receiver=None):
    performs the task of {skill_name.replace("_", " ")}
    Arguments:
        object_id_subject: int
            Id of the object to perform the task of {skill_name.replace("_", " ")}. 
            This object is the subject of the action.

        object_id_receiver: int (optional)
            Id of any other object that may be relevant to accomplishing the task.
    Returns: None
""",
        }

        self.primitives_lst = self.primitives_lst + f",`{skill_name}`"
        self.primitives_description = (
            self.primitives_description[:-4]
            + "\n"
            + self.primitives[skill_name]["description"]
            + "\n```"
        )
        self.primitives_running_lst.append(skill_name)

        return self.new_skill
    
    def get_objects_contained_and_over(self, object_id):
        object_ids_contained = self.object_dicts[object_id]["relation"]["contains"]
        object_ids_above = self.object_dicts[object_id]["relation"]["below"]
        
        name = self.object_dicts[object_id]["used_name"]

        result = object_ids_contained + object_ids_above
        if len(result) == 0:
            self.feedback_queue.append(f"No objects contained in {name}.")
        else:
            self.feedback_queue.append(f"some objects found in {name}.")
        return result 


    def get_container_id(self, object_id):
        container_id = self.object_dicts[object_id]["relation"]["contained_in"]
        name = self.object_dicts[object_id]["used_name"]
        if len(container_id) > 0:
            self.feedback_queue.append(f"container for {name} was found.")
            return container_id[0]
        else:
            self.feedback_queue.append(f"{name} is not contained in anything. no container found.")
            return None    
        
    def dummy(self, *args, **kwargs):
        print("called with arguments:", *args, **kwargs)

    def load_primitives(self):
        self.primitives = {
            "start_task": {
                "fn": self.dummy,
                "description": """
start_task()
    must be called at the start of any task. It starts the robot.                
""",
            },
            "end_task": {
                "fn": self.dummy,
                "description": """
end_task()
    must be called when a task completes. It stops the robot.                
""",
            },
            "get_all_object_ids": {
                "fn": self.dummy,
                "description": """
get_all_object_ids()
    returns a list of all integer object ids present in the scene; 
    Returns:
        ids: list(int)
""",
            },
            "get_container_id": {
                "fn": self.dummy,
                "description": """
get_container_id(object_id)
    gives the id of the object that contains `object_id`
    Arguments:
        object_id: int
            id of the object that is contained in some container
    Returns:
        container_id: int or None
            the id of the container that contains `object_id`
            None is returned when the object_id is not contained in any container.
""",
            },
            "get_objects_contained_and_over": {
                "fn": self.dummy,
                "description": """
get_objects_contained_and_over(object_id)
    gives the ids of all the objects that lie inside or over `object_id`
    Arguments:
        object_id: int
            id of an object that contains something or over which lie other objects
    Returns:
        ids: list(int)
            the ids of all the objects that lie either inside or over the `object_id`
            an empty list is returned when nothing lies over or inside.
                """,
            },
            "find": {
                "fn": self.dummy,
                "description": """
find(object_label=None, visual_description=None, place_description=None, object_ids=None)
Finds an object in the scene given atleast one of object_label, visual description or place description.
Arguments:
    object_label: str
        The name with which the object has been referred to earlier
        For example, "the second tray", "the third bowl" etc
        By default, this argument is None

    visual_description: str
        object with some visual description of what it is.
        For example, "the red mug", "the blue tray", "the checkered box"
        By default, this argument is None
        
    place_description: str
        a string that describes where the object is located. 
        For example, to find a bowl that is on the right of the tray, the function call will be 
        `find("bowl that lies to the right of the tray")`, or to get the mug that is contained in
        the second bowl, the call would be `find("mug that is inside the second bowl") and so on.
        By default, this argument is None

    object_ids: list(int)
        A list of `int` object ids in which the object should be found, when specified.
        By default when this argument is None, all the objects are considered for 
        finding the best matching 
    
    Atleast one of the first three arguments must be specified. Typically, the use of the first and 
    the second argument is enough but third can be used whenver needed.
        
Returns: int
    object_id, an integer representing the object that best matched with the description
""",
            },
            "get_location": {
                "fn": self.dummy,
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
            #             "move_arm": {
            #                 "fn": self.move_arm,
            #                 "description": """
            # move_arm(position)
            #     Moves to the robotic arm to a location given by position
            #     Arguments:
            #         position: 3D array
            #             the location to move the arm to
            #     Returns: None
            # """,
            #             },
            "pick": {
                "fn": self.dummy,
                "description": """
pick(object_id)
    Picks up an object that `object_id` represents. A `place` needs to occur before 
    another call to pick, i.e. two picks cannot occur one after the other
    Arguments:
        object_id: int
            Id of the object to pick
    Returns: None
""",
            },
            "get_place_position": {
                "fn": self.dummy,
                "description": """
get_place_position(object_id, reference_object_id, place_description)
    Finds the position to place the object `object_id` at a location described by `place_description`
    with respect to the `reference_object_id`. 
    Arguments:
        object_id: int
            Id of the object to place
        reference_object_id: int
            id of the object relevant for placing the object_id
        place_description: str
            a string that describes where with respect to the reference_object_id
            the object_id should be placed.

    Returns: 3D array
        the [x, y, z] value for the place location is returned.

    For example, 
    to place a mug to the left of a bowl, the following function call should be used
        get_place_position(mug_id, bowl_id, "to the left")
    to place a mug into a bowl:
        get_place_position(mug_id, bowl_id, "inside")
    to place a mug above a box:
        get_place_position(mug_id, box_id, "above")
                """,
            },
            "place": {
                "fn": self.dummy,
                "description": """
place(object_id, position)
    Moves to the position and places the object `object_id`, at the location given by `position`
    Arguments:
        object_id: int
            Id of the object to place
        position: 3D array
            the place location
    Returns: None
""",
            },
            #             "place": {
            #                 "fn": self.place,
            #                 "description": """
            # place(object_id_1, object_id_2)
            #     Moves 'object_id_1' over to 'object_id_2'
            #     Arguments:
            #         object_id_1: int
            #             Id of the object to place
            #         object_id_2: int
            #             Id of the object to place relative to
            #     Returns: None
            # """,
            #             },
        }

        if self.skill_learner:
            self.primitives["learn_skill"] = {
                "fn": self.dummy,
                "description": """
learn_skill(skill_name)
    adds a new skill to the current list of skills
	Arguments:
	    skill_name: str
            a short name for the skill to learn, must be a string that can 
            represent a function name (only alphabets and underscore can be used)
	Returns:
        skill_function: method
            a function that takes as input an object_id and 
            performs the skill on the object represented by the object_id
            another relevant object_id can be passed optionally.

    For example:
        # Example 1:
        drawer_id = 2
        open_drawer = learn_skill("open_drawer")
        open_drawer(drawer_id) # opens the drawer represented by drawer_id

        # Example 2:
        bowl_id = 3
        bin_id = 4
        tilt_bowl = learn_skill("tilt_bowl")
        tilt_bowl(bowl_id, bin_id) # tilts the bowl over the location of the bin
""",
            }

        fn_lst = list(self.primitives.keys())
        self.primitives_lst = ",".join([f"`{fn_name}`" for fn_name in fn_lst])
        self.primitives_description = "\n".join(
            [self.primitives[fn]["description"] for fn in fn_lst]
        )
        self.primitives_description = "```\n" + self.primitives_description + "\n```"
        self.primitives_running_lst = list(self.primitives.keys())

        return self.primitives

    def remove_objects(self):
        for idx, obj_info in self.sim_dict["object_dicts"].items():
            obj_id = obj_info["mask_id"]
            self.pb_client.removeBody(obj_id)
        self.sim_dict["object_dicts"] = {}
        self.object_dicts = {}

    # def check_update_dicts(self):

    #     input("press enter after setting the first scene")
    #     first_dict = self.get_object_dicts()
    #     first_desc, first_dict = self.get_scene_description(first_dict)

    #     self.object_dicts = first_dict

    #     print_object_dicts(first_dict)

    #     print("first description:", first_desc)

    #     obj_id_to_pick = int(input("enter object_id to pick"))
    #     self.pick(obj_id_to_pick)

    #     print("feedback queue", self.feedback_queue)

    #     self.place(obj_id_to_pick, np.zeros(2))

    #     print("feedback queue", self.feedback_queue)

    #     second_dict= self.get_object_dicts()

    #     print_object_dicts(second_dict)

    #     final_desc, final_dict = self.update_dict_util(first_dict, second_dict)
    #     print_object_dicts(final_dict)

    #     for obj_id in final_dict:
    #         obj_pcd = final_dict[obj_id]["pcd"]
    #         rgb = final_dict[obj_id]["rgb"]

    #         print(obj_pcd.shape, rgb.shape)
    #         name = f"final_{obj_id}"
    #         util.meshcat_pcd_show(self.mc_vis, obj_pcd, color=None, name=f"scene/second_{name}_{obj_id}")

    #     obj_id_to_pick = int(input("enter object_id to pick"))
    #     self.pick(obj_id_to_pick)

    #     print("feedback queue", self.feedback_queue)

    #     self.place(obj_id_to_pick, np.zeros(2))
    #     print("feedback queue", self.feedback_queue)

    #     second_dict= self.get_object_dicts()

    #     print_object_dicts(second_dict)

    #     final_desc, final_dict = self.update_dict_util(first_dict, second_dict)
    #     print_object_dicts(final_dict)

    #     for obj_id in final_dict:
    #         obj_pcd = final_dict[obj_id]["pcd"]
    #         rgb = final_dict[obj_id]["rgb"]

    #         print(obj_pcd.shape, rgb.shape)
    #         name = f"final_{obj_id}"
    #         util.meshcat_pcd_show(self.mc_vis, obj_pcd, color=None, name=f"scene/third_{name}_{obj_id}")
