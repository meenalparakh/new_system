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
from scene_description_utils import construct_graph, graph_traversal, text_description

OPP_RELATIONS = {"above": "below", "contained_in": "contains"}


def print_object_dicts(object_dict, ks=["label", "relation", "desc", "used_name"]):
    for id, info in object_dict.items():
        print("Object id", id)
        for k, v in info.items():
            if k in ks:
                print(f"    {k}: {v}")
        print()


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


class MyRobot(Robot):
    def __init__(self, gui=False):
        super().__init__(
            "franka",
            pb_cfg={"gui": gui},
            use_arm=False,
            use_eetool=False,
            use_base=False,
        )
        # success = self.arm.go_home()
        # if not success:
        #     log_warn("Robot go_home failed!!!")

        # setup table
        ori = euler2quat([0, 0, np.pi / 2])
        self.table_id = self.pb_client.load_urdf(
            "table/table.urdf", [0.6, 0, 0.4], ori, scaling=0.9
        )
        self.pb_client.changeDynamics(self.table_id, 0, mass=0, lateralFriction=2.0)

        # setup camera
        self.cams = []
        for i in range(2):
            self.cams.append(
                RGBDCameraPybullet(cfgs=self._camera_cfgs(), pb_client=self.pb_client)
            )
        self._setup_cameras()
        self.depth_scale = 1.0

        focus_pt = [0, 0, 1]  # ([x, y, z])
        self.cam.setup_camera(focus_pt=focus_pt, dist=3, yaw=90, pitch=0, roll=0)
        self.object_dicts = {}

        self.sim_dict = {"object_dicts": {}}

    def reset(self, task_name):
        self.task = TASK_LST[task_name](self)
        self.task.reset()
        for _ in range(100):
            self.pb_client.stepSimulation()

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
        rgbs = []
        depths = []
        segs = []
        for idx, cam in enumerate(self.cams):
            rgb, depth, seg = cam.get_images(get_rgb=True, get_depth=True, get_seg=True)
            rgbs.append(rgb)
            depths.append(depth)
            segs.append(seg)

        # ////////////////////////////////////////////////////////////////////////////
        self.sim_dict["segs"] = segs
        # ////////////////////////////////////////////////////////////////////////////

        return {"colors": rgbs, "depths": depths}

    def _setup_cameras(self):
        self.cams[0].setup_camera(
            focus_pt=[0.5, 0.0, 1.0], dist=1, yaw=-90, pitch=-45, roll=0
        )
        self.cams[1].setup_camera(
            focus_pt=[0.5, 0.0, 1.0], dist=1, yaw=90, pitch=-45, roll=0
        )

    def get_combined_pcd(self, colors, depths):
        pcd_pts = []
        pcd_rgb = []
        for color, depth, cam in zip(colors, depths, self.cams):
            cam_extr = cam.get_cam_ext()
            pts, rgb = self.get_pcd(
                cam_ext_mat=cam_extr, rgb_image=color, depth_image=depth, cam=cam
            )
            pcd_pts.append(pts)
            pcd_rgb.append(rgb)

        return np.concatenate(pcd_pts, axis=0), np.concatenate(pcd_rgb, axis=0)

    def get_pcd(
        self,
        in_world=True,
        filter_depth=True,
        depth_min=0.10,
        depth_max=2.50,
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

    def get_segment_labels_and_embeddings(self, colors, depths, clip_):
        # for each rgb image, returns a segmented image, and
        # a dict containing the labels for each segment id
        # the segments are numbered taking into account different
        # instances of the category as well.

        # ////////////////////////////////////////////////////////////////
        segs = self.sim_dict["segs"]

        image_embeddings = []

        for s, c in zip(segs, colors):
            unique_ids = np.unique(s.astype(int))
            image_crops = []
            for uid in unique_ids:
                mask = s == uid
                bb = get_bb(mask)
                r1, c1, r2, c2 = bb
                # print(r1, c1, r2, c2)
                crop = c[r1 : r2 + 1, c1 : c2 + 1, :]
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
        label_infos=None,
        visualization=False,
    ):
        pcd_pts = []
        pcd_rgb = []
        pcd_seg = []
        for color, depth, segment, cam in zip(colors, depths, segs, self.cams):
            cam_extr = cam.get_cam_ext()
            pts, rgb = self.get_pcd(
                cam_ext_mat=cam_extr, rgb_image=color, depth_image=depth, cam=cam
            )

            _, seg = self.get_pcd(
                cam_ext_mat=cam_extr,
                rgb_image=np.repeat(segment[..., None], repeats=3, axis=2),
                depth_image=depth,
                cam=cam,
            )

            valid = pts[:, 2] > remove_floor_ht

            pcd_pts.append(pts[valid])
            pcd_rgb.append(rgb[valid])
            pcd_seg.append(seg[valid])

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

        seg = pcd_seg[0][:, 0]
        unique_ids = list(np.unique(seg.astype(int)))

        for uid in unique_ids:
            valid = seg == uid
            objects[uid] = {
                "pcd": pcd_pts[0][valid],
                "rgb": pcd_rgb[0][valid],
                "label": [label_infos[0][uid]["label"]],
                "embed": [label_infos[0][uid]["embedding"]],
            }

        for idx in range(1, len(pcd_pts)):
            seg = pcd_seg[idx][:, 0]
            unique_ids = list(np.unique(seg.astype(int)))
            new_dict = {}
            for uid in unique_ids:
                print(uid)
                valid = seg == uid
                new_pcd = pcd_pts[idx][valid]
                new_rgb = pcd_rgb[idx][valid]

                diffs = []
                obj_lst = list(objects.keys())
                for obj in obj_lst:
                    original_pcd = objects[obj]["pcd"]
                    d = combined_variance(new_pcd, original_pcd)
                    diffs.append(d)

                min_diff = np.min(diffs)
                if min_diff < 0.02:
                    print("min_diff less than 2 cm, updating pcd")
                    obj_id = obj_lst[np.argmin(diffs)]
                    p = objects[obj_id]["pcd"]
                    r = objects[obj_id]["rgb"]
                    objects[obj_id]["pcd"] = np.vstack((p, new_pcd))
                    objects[obj_id]["rgb"] = np.vstack((r, new_rgb))

                    objects[obj_id]["label"].append(label_infos[idx][uid]["label"])
                    objects[obj_id]["embed"].append(label_infos[idx][uid]["embedding"])

                else:
                    new_dict[uid + np.random.randint(100, 200)] = {
                        "pcd": new_pcd,
                        "rgb": new_rgb,
                        "label": label_infos[idx][uid]["label"],
                        "embed": label_infos[idx][uid]["embedding"],
                    }
                    objects.update(new_dict)

        # //////////////////////////////////////////////////////////////////////////////
        # Visualization of point clouds
        # //////////////////////////////////////////////////////////////////////////////

        if visualization:
            print("Number of objects", len(objects))
            pcds = []
            rand_colors = dp.get_colors(len(objects))
            for idx, obj in enumerate(objects):
                print(objects[obj]["label"])
                pcd = objects[obj]["pcd"]
                color = objects[obj]["rgb"]
                p = open3d.geometry.PointCloud(open3d.utility.Vector3dVector(pcd))
                p.colors = open3d.utility.Vector3dVector(color / 255.0)
                open3d.visualization.draw_geometries([p])

                p.paint_uniform_color(rand_colors[idx])
                pcds.append(p)
            open3d.visualization.draw_geometries(pcds)

        # //////////////////////////////////////////////////////////////////////////////

        return objects

    def get_scene_description(self, object_dicts):
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
        description = text_description(object_dicts, traversal_order, path, side=side)

        return description, object_dicts
