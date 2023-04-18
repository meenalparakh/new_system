from __future__ import print_function

from copy import copy
import cv2
import numpy as np
import pickle
from scipy.spatial.transform import Rotation as R
# import matplotlib.pyplot as plt
import os
# import time
import open3d
from data_collection.utils.pose_utils import gen_poses
import sys
# from airobot.utils.common import to_rot_mat, rot2quat

# from PIL import Image, ImageEnhance

import json
# import argparse
from airobot import Robot
# import shutil
import json, random
from datetime import datetime
import argparse

from data_collection.utils.pointcloud_utils import dbscan_clustering
from data_collection.utils.object_pointclouds import create_object_pcd

from data_collection.utils.data_collection_utils import collect_data, read_all, get_scale, transform_configs
from data_collection.utils.data_collection_utils import rescale_extrinsics, clear_colmap_dirs
from data_collection.utils.pointcloud_utils import get_pcds, save_pcd, plane_segmentation, postprocess_pointcloud

from data_collection.utils.pointcloud_utils import project_pcd_to_image, get_table_cluster
from data_collection.utils.data_post_processing_utils import get_dict, get_segmentation_data, divide_test_data
from data_collection.utils.data_collection_utils import get_intr_from_cam_params, read_updated_configs
from data_collection.utils.utils import generate_random_camera_matrices

# import albumentations as A

class Scene:
    def __init__(self, scene_dir, scene_name, object_category='unknown', table_only=False):
        self.scene_name = scene_name
        self.scene_dir = os.path.abspath(scene_dir)

        summary_file = os.path.join(self.scene_dir, 'summary.json')
        self.summary_fname = summary_file
        self.object_category = object_category

        if os.path.exists(summary_file):
            with open(summary_file, 'r') as f:
                self.summary = json.load(f)
            if not ('cleaned_pcd_fname' in self.summary):
                self.summary['cleaned_pcd_fname'] = None
            else:
                self.summary['cleaned_pcd_fname'] = os.path.join(scene_dir, "cleaned_pcd.ply")
            
            if not ('clustered_pcd_fname' in self.summary):
                self.summary['clustered_pcd_fname'] = None
            else:
                self.summary['clustered_pcd_fname'] = os.path.join(scene_dir, "clustered_pcd.ply")
            
            if not ('transformed_configs' in self.summary):
                self.summary['transformed_configs'] = None
            else:
                self.summary['transformed_configs'] = os.path.join(scene_dir, "transformed_configs.pkl")
            
            if not ('transformed_far_configs' in self.summary):
                self.summary['transformed_far_configs'] = None
            else:
                self.summary['transformed_far_configs'] = os.path.join(scene_dir, "transformed_far_configs.pkl")

            if not ('shared_intrinsics' in self.summary):
                self.summary['shared_intrinsics'] = None
            else:
                self.summary['shared_intrinsics'] = os.path.join(scene_dir, "shared_intrinsic.pkl")
                
            if not ('cluster_info' in self.summary):
                self.summary['cluster_info'] = None
            else:
                self.summary['cluster_info'] = os.path.join(scene_dir, "clustered_pcd_color_info.pkl")
                
            if not ('object_pcds_dir' in self.summary):
                self.summary['object_pcds_dir'] = None
            else:
                self.summary['object_pcds_dir'] = os.path.join(scene_dir, "object_pcds")                

            if not ('table_only' in self.summary):
                self.summary['table_only'] = False
            if not ('config_lst' in self.summary):
                self.summary['config_lst'] = {}

        else:
            self.summary = {
                'table_only': table_only,
                'data_collected': False,
                'n_far': 0,
                'n_close': 0,
                'poses_registered': False,
                'shared_intrinsics': None,
                'merged_pcd_fname': None,
                'cleaned_pcd_fname': None,
                'clustered_pcd_fname': None,
                'cluster_info': None,
                'transformed_configs': None,
                'transformed_far_configs': None,
                'scene_name': self.scene_name,
                'object_category': self.object_category,
                'revert_normal': False,
                'comment': "",
                'config_lst': {}
            }

            self.save_summary()

    def save_summary(self):
        if not os.path.exists(self.scene_dir):
            os.makedirs(self.scene_dir)

        with open(self.summary_fname, 'w') as f:
            json.dump(self.summary, f)
        

    def collect_data(self, n, data_type='close', write_=True, on_click=False):
        if n <= 0:
            return
        robot = Robot('ur5e_2f140', pb=False, use_arm=False, use_cam=True)
        collect_data(robot, n, self.scene_dir, data_type=data_type, 
                    depth_scale=1000, on_click=on_click)
        t = 'n_close' if data_type == 'close' else 'n_far'
        self.summary[t] = self.summary[t] + n
        self.summary['data_collected'] = True
        self.summary['shared_intrinsics'] = os.path.join(self.scene_dir, 'shared_intrinsic.pkl')

        self.summary['poses_registered'] = False
        self.summary['merged_pcd_fname'] = None
        self.summary['cleaned_pcd_fname'] = None
        self.summary['clustered_pcd_fname'] = None
        self.summary['object_category'] = self.object_category

        if write_:
            self.save_summary()

    def run_colmap(self, skip_if_done=True):
        if skip_if_done:
            if self.summary['poses_registered']:
                print('Poses already registered. Skipping COLMAP')
                return

        clear_colmap_dirs(self.scene_dir)
        with open(self.summary['shared_intrinsics'], 'rb') as f:
            cam_params_shared = pickle.load(f)  
        success = gen_poses(self.scene_dir, 'exhaustive_matcher', cam_model='PINHOLE', 
                    cam_params=cam_params_shared,
                    call_colmap='colmap', dense_reconstruct=False)
        if success:
            print('COLMAP ran successfully!')
            self.summary['poses_registered'] = True
        self.save_summary()

    def get_scene_pcd(self, visualize=False, voxel_size=0.002):
        colors, depths, configs, _ = read_all(self.scene_dir, skip_image=5)
        print("Number of configs", len(configs), len(depths), len(colors))
        scale = get_scale(self.scene_dir, depths, configs)
        configs = rescale_extrinsics(configs, scale)

        print('Combining PCDs...')
        xyzs, colors, cam_extrs = get_pcds(colors, depths, configs)

        # print('Saving PCD...')
        pcd = save_pcd(xyzs, colors, cam_extrs, 
                None, 
                save_pcd=False,
                visualize_pcd=visualize,
                downsample_voxel_size=voxel_size,
                visualize_camera=True)

        cleaned_pcd_fname = os.path.join(self.scene_dir, 'cleaned_pcd.ply')

        pcd, plane_params, normal_vec = get_table_cluster(pcd, eps=0.01,
                             min_points=200, get_normal=self.summary["table_only"])
        pcd, transform = postprocess_pointcloud(pcd, plane_params, normal_vec,
                    revert_normal=False, visualize=False,
                    keep_table=self.summary["table_only"])

        print('Saving new configs and the point cloud...')

        configs = transform_configs(configs, transform)
        new_configs_file = os.path.join(self.scene_dir, 'transformed_configs.pkl')
        with open(new_configs_file, 'wb') as f:
            pickle.dump(configs, f)
        self.summary['transformed_configs'] = new_configs_file

        # self.save_summary()
        margin = 3.0
        min_bound = np.array([-margin, -margin, -0.01])
        max_bound = np.array([margin, margin, np.inf])
        bb = open3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)

        pcd = pcd.crop(bb)

        if visualize:
            open3d.visualization.draw_geometries([pcd])

        # pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=0.01)
        # if visualize:
        #     open3d.visualization.draw_geometries([pcd])

        open3d.io.write_point_cloud(cleaned_pcd_fname, pcd)

        self.summary['cleaned_pcd_fname'] = cleaned_pcd_fname
        self.save_summary()
        print('Cleaning... done.')

        return pcd

    def merge_and_clean_pcd(self, skip_if_done=True, visualize=False, revert_normal=False,
                    voxel_size=0.002, outlier_std_ratio=0.005):
        if skip_if_done:
            if self.summary['cleaned_pcd_fname'] is not None:
                print('Cleaned PCD exists, skipping processing')
                return

        print('Reading COLMAP poses...')

        if not self.summary['poses_registered']:
            raise ValueError("Poses not registered! Run COLMAP to obtain poses.")

        colors, depths, configs, far_configs = read_all(self.scene_dir)
        print(f'Total registered far images: {len(far_configs)}')
        scale = get_scale(self.scene_dir, depths, configs)
        configs = rescale_extrinsics(configs, scale)
        far_configs = rescale_extrinsics(far_configs, scale)

        print('Combining PCDs...')
        xyzs, colors, cam_extrs = get_pcds(colors, depths, configs)

        # print('Saving PCD...')
        pcd = save_pcd(xyzs, colors, cam_extrs, 
                None, 
                save_pcd=False,
                visualize_pcd=visualize,
                # downsample_voxel_size=0.0005,
                # downsample_voxel_size=0.001,
                downsample_voxel_size=voxel_size,
                visualize_camera=True)

        print('Cleaning:')

        cleaned_pcd_fname = os.path.join(self.scene_dir, 'cleaned_pcd.ply')

        print('Removing outliers before obtaining scene components')
# 0.005
        pcd, plane_params, normal_vec = get_table_cluster(pcd, eps=0.01,
                             min_points=200, get_normal=self.summary["table_only"])
        pcd, transform = postprocess_pointcloud(pcd, plane_params, normal_vec,
                    revert_normal=revert_normal, visualize=False,
                    keep_table=self.summary["table_only"])

        print('Saving new configs and the point cloud...')

        configs = transform_configs(configs, transform)
        new_configs_file = os.path.join(self.scene_dir, 'transformed_configs.pkl')
        with open(new_configs_file, 'wb') as f:
            pickle.dump(configs, f)
        self.summary['transformed_configs'] = new_configs_file

        far_configs = transform_configs(far_configs, transform)
        new_configs_file = os.path.join(self.scene_dir, 'transformed_far_configs.pkl')
        with open(new_configs_file, 'wb') as f:
            pickle.dump(far_configs, f)
        self.summary['transformed_far_configs'] = new_configs_file

        # self.save_summary()
        # if visualize:
        #     open3d.visualization.draw_geometries([pcd])

        margin = 3.0
        min_bound = np.array([-margin, -margin, -0.01])
        max_bound = np.array([margin, margin, np.inf])
        bb = open3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)

        pcd = pcd.crop(bb)
        pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=0.01)


        open3d.io.write_point_cloud(cleaned_pcd_fname, pcd)

        self.summary['cleaned_pcd_fname'] = cleaned_pcd_fname
        self.save_summary()
        print('Cleaning... done.')


    def cluster_pcd(self, skip_if_done=True, save_object_pcds=True, visualize=False):

        if skip_if_done:
            if self.summary['clustered_pcd_fname'] is not None:
                print('Clustered PCD exists, skipping clustering')
                return

        if self.summary['cleaned_pcd_fname'] is None:
            raise ValueError("No cleaned pointcloud exists.")

        clustered_pcd_fname = os.path.join(self.scene_dir, 'clustered_pcd.ply')
        clustered_pcd_info = os.path.join(self.scene_dir, 'clustered_pcd_color_info.pkl')
        object_pcds_dir = os.path.join(self.scene_dir, 'object_pcds')

        pcd = open3d.io.read_point_cloud(self.summary['cleaned_pcd_fname'])
        # pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=0.001)

        if visualize:
            open3d.visualization.draw_geometries([pcd])

        # cluster and find mask
        margin = 3.0
        min_bound = np.array([-margin, -margin, 0.02])
        max_bound = np.array([margin, margin, 0.50])
        bb = open3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        pcd_mask = pcd.crop(bb)

        # pcd_mask = pcd

        # pcd_mask = pcd_mask.voxel_down_sample(voxel_size=0.0008)
        # pcd_mask = pcd_mask.voxel_down_sample(voxel_size=0.0005)

        pcd_mask, diff_objects_info = dbscan_clustering(pcd_mask, None, 
                                    clustered_pcd_fname, 
                                    save_pcd=True, 
                                    eps=0.01, min_points=200,  # eps=0.005, min_points=50,
                                    get_object_pcds=save_object_pcds,
                                    object_pcds_dir=object_pcds_dir,
                                    visualize=visualize)
        # open3d.visualization.draw_geometries([pcd_mask])

        with open(clustered_pcd_info, 'wb') as f:
            pickle.dump(diff_objects_info, f)

        self.summary['clustered_pcd_fname'] = clustered_pcd_fname
        self.summary['cluster_info'] = clustered_pcd_info
        self.summary['object_pcds_dir'] = object_pcds_dir


        self.save_summary()
        print('Clustering... Done.')

    def get_pcd_cluster_info(self, get_object_pcds=False):

        print('Loading point clouds...')

        if self.summary['cleaned_pcd_fname'] is None:
            raise ValueError("Cleaned point cloud does not exist.")

        if self.summary['clustered_pcd_fname'] is None:
            raise ValueError("Clustered point cloud does not exist.")

        pcd = open3d.io.read_point_cloud(self.summary['cleaned_pcd_fname'])

        margin = 3.0
        min_bound = np.array([-margin, -margin, -0.01])
        max_bound = np.array([margin, margin, np.inf])

        # min_bound = np.array([-0.5, -0.5, -0.01])
        # max_bound = np.array([0.5, 0.5, np.inf])
        bb = open3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        pcd = pcd.crop(bb)
        pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=0.05)

        clustered_pcd = open3d.io.read_point_cloud(self.summary['clustered_pcd_fname'])
        with open(self.summary['cluster_info'], 'rb') as f:
            diff_objects_info = pickle.load(f)

        if get_object_pcds:
            object_pcds = []
            for l in diff_objects_info['labels']:
                fname = os.path.join(self.summary['object_pcds_dir'], f'{l}.ply')
                object_pcds.append(open3d.io.read_point_cloud(fname))
            return pcd, clustered_pcd, diff_objects_info, object_pcds

        return pcd, clustered_pcd, diff_objects_info

    def get_labeled_data_fn(self, pcd, clustered_pcd, diff_objects_info, object_pcds,
                data_dir, image_type, n, p=1.0):
        if n == 0:
            print(f'No image of type {image_type} generated.')
            return 
        # get camera extrinsics and intrinsics
        diff_colors = diff_objects_info['diff_colors']
        labels = diff_objects_info['labels']

        # make dirs for storing data
        data_dir = os.path.abspath(data_dir)
        
        print('Obtaining camera extrinsics...')

        if image_type == 'close':
            flag = 'A'
            tmp = self.summary['transformed_configs']

        if image_type == 'far':
            flag = 'B'
            tmp = self.summary['transformed_far_configs']

        if image_type in ["close", "far"]:
            with open(tmp, 'rb') as f:
                configs = pickle.load(f)

            cam_extrs = [config['extrinsics'] for config in configs]
            names = [config['fname'] for config in configs]
            geoms = []

        elif image_type == 'random-near':
            cam_extrs, geoms = generate_random_camera_matrices(n, 'near')
            names = [f'{idx}.png' for idx in range(len(cam_extrs))]
            flag = 'C'

        elif image_type == 'random-far':
            cam_extrs, geoms = generate_random_camera_matrices(n, 'far')
            names = [f'{idx}.png' for idx in range(len(cam_extrs))]
            flag = 'D'

        else:
            print('Warning: Unkown image type, returning')
            return
        

        projected_images_dir = os.path.abspath(os.path.join(data_dir, flag, 'images'))
        # projected_depths_dir = os.path.abspath(os.path.join(data_dir, flag, 'depths'))

        masks_dir =  os.path.abspath(os.path.join(data_dir, flag, 'masks'))

        if not os.path.exists(projected_images_dir):
            os.makedirs(projected_images_dir)
        if not os.path.exists(masks_dir):
            os.makedirs(masks_dir)

        if flag == 'C' or flag == 'D':
            with open(os.path.join(data_dir, flag, 'extrs.pkl'), 'wb') as f:
                pickle.dump(cam_extrs, f)

            
        # height, width = 480, 640
        cam_intr, height, width = get_intr_from_cam_params(self.scene_dir, hw=True)
        dilation_kernel = np.ones((2,2), np.uint8)
        # transform = A.Compose([
        #     A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
        # ])


        print('generating images', cam_intr, height, width)
        print('Projecting pcd to images...')
        data_lst = []
        for idx, cam_extr in enumerate(cam_extrs):
            epsilon = np.random.rand()
            if epsilon > p:
                continue
                
            color_fname = names[idx]
            print(color_fname)
            background_val = -1 if self.summary["table_only"] else 0
            colormap = project_pcd_to_image(pcd, cam_intr, cam_extr, 
                                            height, width, depth=False,
                                            background_val=background_val)
            # colormap = cv2.dilate(colormap, kernel, iterations=1)
            # tmp = np.where(colormap == -1, 0, colormap)
            # tmp = cv2.dilate(colormap, )

            # colormap = cv2.dilate(colormap, dilation_kernel, iterations=1)
            
            if self.summary["table_only"]:
                print("shape:", colormap.shape, colormap.max(), colormap.dtype)
                _colormap = np.where(colormap == -1, np.zeros_like(colormap), colormap).astype(np.uint8)
                # _colormap = Image.fromarray(_colormap)
                # enhancer = ImageEnhance.Brightness(_colormap)
                # factor = 1.1
                # _colormap = np.array(enhancer.enhance(factor))
                _colormap = transform(image=_colormap)['image']

                image_location = os.path.join(self.scene_dir, "images", color_fname)
                background_img = cv2.imread(image_location)
                _colormap = _colormap[:, :, ::-1]
                colormap = colormap[:, :, ::-1]

                # _colormap = cv2.dilate(_colormap, dilation_kernel)
                colormap = np.where(colormap == -1, background_img, _colormap).astype(np.uint8)
                kernel = np.ones((3,3),np.float32)/9
                colormap = cv2.filter2D(colormap, -1, kernel)
            else:
                colormap = colormap[:,:,::-1].astype(np.uint8)


            colormap_fname = os.path.join(projected_images_dir, color_fname)
            cv2.imwrite(colormap_fname, colormap)

            mask = project_pcd_to_image(clustered_pcd, cam_intr, cam_extr, 
                                                    height, width)

            amodal_masks, object_centers = None, np.zeros((len(diff_colors), 3))
            # amodal_masks, object_centers = self.get_object_amodal_masks_and_centers(labels, 
            #                 object_pcds, cam_intr, cam_extr, height, width)

            binary_masks_dir = os.path.join(masks_dir, color_fname[:-4])


#            max_num_pixels = 250 if ((flag == 'B') or (flag == 'D')) else 1000
            max_num_pixels = 10
            segments_info, mask_int = get_segmentation_data(mask, diff_colors, amodal_masks, 
                                        object_centers,
                                        save_dir=binary_masks_dir,
                                        max_num_pixels=max_num_pixels)

            _flag = flag
            if self.summary["table_only"]:
                _flag = "C" if flag=='A' else "D"

            d = get_dict(idx, self.scene_dir, 
                            projected_images_dir, masks_dir,
                            color_fname, 
                            height, width, 
                            cam_extr,
                            segments_info, 
                            binary_masks_dir, 
                            self.object_category,
                            _flag)

            mask = cv2.medianBlur(mask.astype(np.uint8), 5)
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
            mask_fname = os.path.join(masks_dir, color_fname)
            cv2.imwrite(mask_fname, mask)
            cv2.imwrite(os.path.join(masks_dir, 'labels_' + color_fname), mask_int)

            data_lst.append(d)

        with open(os.path.join(data_dir, f'summary_{flag}.pkl'), 'wb') as f:
            pickle.dump(data_lst, f)
        print('Done')

    def get_object_amodal_masks_and_centers(self, labels, object_pcds, cam_intr, cam_extr, 
                            height, width):

        amodal_masks = []
        object_centers = []
        print("labels", labels)
        for l in labels:
            object_pcd = object_pcds[l]
            amodal_mask = project_pcd_to_image(object_pcd, cam_intr, cam_extr, height, width)
            amodal_masks.append(amodal_mask)
            object_centers.append(np.median(np.asarray(object_pcd.points), axis=0))

        return amodal_masks, object_centers


    def train_val_test_split(self, data_dir, splits):
        # if self.summary["table_only"]:
        #     for config_name, val in self.summary['config_lst'].items():
        #         scene_data_dir = os.path.join(data_dir, config_name)
        divide_test_data(data_dir, split_ratio=splits)


    def create_object_pcd_with_cluster(self, visualize=False, name=None):
        pcd = open3d.io.read_point_cloud(self.summary['cleaned_pcd_fname'])
        if visualize:
            open3d.visualization.draw_geometries([pcd])

        if name is None:
            name = datetime.now().strftime('%Y%m%d_%H%M%S')
        merged, clustered, diff_objects_info = create_object_pcd(pcd, visualize=visualize)

        if visualize:
            open3d.visualization.draw_geometries([merged])
            open3d.visualization.draw_geometries([clustered])

        merged_pcd_fname = os.path.join(self.scene_dir, "merged_" + name + ".ply")
        cluster_pcd_fname = os.path.join(self.scene_dir, "cluster_" + name + ".ply")
        cluster_info_fname = os.path.join(self.scene_dir, "cluster_info_" + name + ".pkl")

        open3d.io.write_point_cloud(merged_pcd_fname, merged)
        open3d.io.write_point_cloud(cluster_pcd_fname, clustered)
        with open(cluster_info_fname, 'wb') as f:
            pickle.dump(diff_objects_info, f)

        self.summary['config_lst'][name] = {
                            'pcd': merged_pcd_fname,
                            'clustered': cluster_pcd_fname,
                            'cluster_info': cluster_info_fname}
        self.save_summary()

    def delete_configs(self):
        self.clear_old_configs()
        configs = self.summary['config_lst']
        for config_name, val in self.summary['config_lst'].items():
            print('Deleting', val)
            os.remove(val['pcd'])
            os.remove(val['clustered'])
            os.remove(val['cluster_info'])

        self.summary['config_lst'] = {}
        self.save_summary()

    def clear_old_configs(self):
        configs = self.summary['config_lst']
        to_remove = []
        clean_dict = copy(self.summary['config_lst'])
        for config_name, val in self.summary['config_lst'].items():
            if not os.path.exists(val['pcd']):
                print(f"No {config_name} exists. Removing")
                clean_dict.pop(config_name)

        self.summary['config_lst'] = clean_dict
        self.save_summary()


    def get_data_for_all_configurations(self, data_dir, p_far=1.0, p_close=1.0):
        self.clear_old_configs()
        num_configs = len(self.summary["config_lst"])
        print(f"Number of configs for this scene: {num_configs}")
        for config_name, val in self.summary['config_lst'].items():
            scene_data_dir = os.path.join(data_dir, config_name)
            if os.path.exists(scene_data_dir):
                print("f{config_name} exists. Skipping")
                continue
            pcd = open3d.io.read_point_cloud(val['pcd'])
            clustered_pcd = open3d.io.read_point_cloud(val['clustered'])
            with open(val['cluster_info'], 'rb') as f:
                diff_objects_info = pickle.load(f)

            # scene_data_dir = os.path.join(data_dir, config_name)
            # if not os.path.exists(scene_data_dir):
            self.get_labeled_data_fn(pcd, clustered_pcd, diff_objects_info, None,
                    scene_data_dir, 'far', 1, p=p_far)
            self.get_labeled_data_fn(pcd, clustered_pcd, diff_objects_info, None,
                    scene_data_dir, 'close', 1, p=p_close)

