import os
import numpy as np
import open3d
from robot_env import MyRobot
from visualize_pcd import VizServer 
from data_collection.utils.data_collection_utils import read_all, get_scale, rescale_extrinsics, transform_configs

import subprocess

DATA_DIR = "../data/scene_data/"

def get_image_labels_and_masks(images, depths):
    ################################################################################################

    labels "./image_labels"

    fpaths = []
    for i, rgb in enumerate(images):
        p = os.path.join(labels, "images", f'{i}.png')
        cv2.imwrite(p, rgb)
        fpaths.append(p)


    results_fname = os.path.join(labels, 'predictions_summary.pkl')
    data_lst_fname = "data_lst_real.pkl"
    data_lst = [{"file_name": config["fname"]} for config in configs]

    with open(os.path.join(labels, data_lst_fname), 'wb') as f:
        pickle.dump(data_lst, f)
            
    subprocess.run(["./run_detect_grasp_scripts.sh", "detect", os.path.abspath(scene_dir), str(3), 
                        data_lst_fname])



class Camera:
    def __init__(self, cam_extr=None, cam_intr=None, H=None, W=None):
        self.depth_scale = 1.0  
        cam_int_mat_inv = np.linalg.inv(cam_intr)

        img_pixs = np.mgrid[0: H,
                            0: W].reshape(2, -1)
        img_pixs[[0, 1], :] = img_pixs[[1, 0], :]
        _uv_one = np.concatenate((img_pixs,
                                np.ones((1, img_pixs.shape[1]))))
        self._uv_one_in_cam = np.dot(cam_int_mat_inv, _uv_one)
        self.cam_ext_mat = cam_extr
        self.cam_intr_mat = cam_intr

    def get_cam_ext(self):
        return self.cam_ext_mat

    def get_cam_intr(self):
        return self.cam_intr_mat


class RealRobot(MyRobot):

    def __init__(self, gui=False, scene_dir=None):
        super().__init__(gui)
        self.scene_dir = scene_dir

    def get_obs(self):
        colors, depths, configs, _ = read_all(self.scene_dir, skip_image=40)
        print("Number of configs", len(configs), len(depths), len(colors))
        scale = get_scale(self.scene_dir, depths, configs)
        configs = rescale_extrinsics(configs, scale)
        self.cams = []

        for idx, conf in enumerate(configs):
            H, W = colors[idx].shape[:2]
            cam = Camera(cam_extr=conf["extrinsics"], cam_intr=conf["intrinsics"], H=H, W=W)
            self.cams.append(cam)

        print("Number of images", len(colors))

        return {"colors": colors, "depths": depths, "configs": configs}

    def get_segment_labels_and_embeddings(self, colors, depths, configs, clip_):
        # for each rgb image, returns a segmented image, and
        # a dict containing the labels for each segment id
        # the segments are numbered taking into account different
        # instances of the category as well.

        # ////////////////////////////////////////////////////////////////
        # running object detection models 
        # ////////////////////////////////////////////////////////////////

        results_fname = os.path.join(scene_dir, 'detic_results', 'predictions_summary.pkl')
        if not os.path.exists(results_fname) or True:
            data_lst_fname = "data_lst_real.pkl"
            data_lst = [{"file_name": config["fname"]} for config in configs]

            with open(os.path.join(scene_dir, data_lst_fname), 'wb') as f:
                pickle.dump(data_lst, f)
                
            subprocess.run(["./run_detect_grasp_scripts.sh", "detect", os.path.abspath(scene_dir), str(3), 
                                data_lst_fname])
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
    
if __name__ == "__main__":

    scene_dir = os.path.join(DATA_DIR, "20221010_1759")
    robot = RealRobot(gui=False, scene_dir=scene_dir)
    obs = robot.get_obs()

    combined_pts, combined_rgb = robot.get_combined_pcd(obs["colors"], obs["depths"])

    viz = VizServer()
    viz.view_pcd(combined_pts, combined_rgb)


    ###################### open3d visualization

    # pcd = open3d.geometry.PointCloud(open3d.utility.Vector3dVector(combined_pts))
    # pcd.colors = open3d.utility.Vector3dVector(combined_rgb/255.0)

    # # open3d.visualization.draw_geometries([pcd])
    # open3d.io.write_point_cloud("combined_pcd.ply", pcd)

    
