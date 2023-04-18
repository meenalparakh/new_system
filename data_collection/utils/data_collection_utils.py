from __future__ import print_function

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

import os
import matplotlib.pyplot as plt
# from detectron2.structures import BoxMode
# from sklearn.cluster import DBSCAN, KMeans
# import open3d
# import pycocotools
from data_collection.utils.colmap_read_model import read_cameras_text, read_images_text, read_points3D_text
from data_collection.utils.pose_utils import load_data
import pickle
import shutil
import time
from datetime import datetime
############################################################################
##                 data collection                                        ##
############################################################################

def collect_data(robot, n, colmap_dir, data_type='close', depth_scale=1000, on_click=False):

    if n == 0:
        return

    idx = 0
    input(f'Press any key to start collection {data_type} data.')
    colors = []
    depths = []
    intrinsics = []
    fnames = []

    for idx in range(n):

        color, depth = robot.cam.get_images(get_rgb=True, get_depth=True)
        depth = depth.astype(np.float32)
        print(f'Min max depth, {np.max(depth), np.min(depth), depth.dtype}')
        intrinsic = robot.cam.get_cam_int()
        fname = datetime.now().strftime('%Y%m%d_%H%M%S')
        print(idx,':', fname)
        colors.append(color)
        depths.append(depth)
        fnames.append(fname)
        intrinsics.append(intrinsic)
        if on_click:
            input("Press next key")
        else:
            time.sleep(1.5)

    print('Data collection finished! Saving data...')
    prefix = "" if data_type == 'close' else "far_"
    save_rgbd(colmap_dir, colors, depths, intrinsics, fnames, prefix)

def read_all(basedir, depth_scale=1000, far_prefix='far', skip_image=1):
    fwh = [None, None, None]
    poses, imgfiles = load_data(basedir, *fwh)  
    cam_intrinsics = get_intr_from_cam_params(basedir)

    print(f"Cam intrinsic: {cam_intrinsics}")
    print("Poses length:", len(poses))

    if len(poses) < 30:
        raise ValueError("less than 30 poses registered.")

    colors = []
    depths = []
    configs = []

    far_configs = []

    indices = list(range(len(imgfiles)))[::skip_image]

    for idx in indices:
        # print(f'Image: {imgfiles[idx]}')
        color_fname = os.path.join(basedir, 'images', imgfiles[idx])
        # depth_fname = os.path.join(basedir, 'depths', imgfiles[idx].replace('.png', '.pkl'))
        # depth_fname = os.path.join(basedir, 'depths', imgfiles[idx])
        depth_fname = os.path.join(basedir, 'depths', imgfiles[idx].replace('.png', '_new.png'))

        config_fname = os.path.join(basedir, 'configs', imgfiles[idx].replace('.png', '.pkl'))

        color = cv2.imread(color_fname)

        with open(config_fname, 'rb') as f:
            config = pickle.load(f)
            pos, ori = poses[idx]

            # print(f'Config keys: {config.keys()}')

            config['intrinsics'] = cam_intrinsics
            config['fname'] = imgfiles[idx]
            config['height'], config['width'] = color.shape[:2]

            position = np.array(pos).reshape(3, 1)
            w, x, y, z = ori
            rotation = R.from_quat([x, y, z, w]).as_matrix()
            cam_extr = np.eye(4)
            cam_extr[:3,:3] = rotation.T
            cam_extr[:3,3:] = -rotation.T @ position
            config['extrinsics'] = cam_extr

        if far_prefix in imgfiles[idx]:
            far_configs.append(config)

        else:
            color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            depth = 0.001 * cv2.imread(depth_fname, cv2.IMREAD_UNCHANGED)

            # with open(depth_fname, 'rb') as f:
            #     depth = 0.001*pickle.load(f)

            colors.append(color)
            depths.append(depth)
            configs.append(config)

    # last = len(colors)
    return colors, depths, configs, far_configs

def get_config_and_depth(fname, configs, depths):
    for idx in range(len(configs)):
        if configs[idx]['fname'] == fname:
            return depths[idx], configs[idx]
    print('Config not found, returning none')
    return None, None

    
def rescale_extrinsics(configs, scale):
    for i in range(len(configs)):
        cam_extr = configs[i]['extrinsics']
        cam_extr[:3,3] = cam_extr[:3,3]*scale
        configs[i]['extrinsics'] = cam_extr

    return configs

def save_rgbd(colmap_dir, colors, depths, intrinsics, fnames, prefix):
    cam_params_shared = None
    color_dir = os.path.join(colmap_dir, 'images')
    depth_dir = os.path.join(colmap_dir, 'depths')
    config_dir = os.path.join(colmap_dir, 'configs')

    if not os.path.exists(color_dir):
        os.makedirs(color_dir)
        os.makedirs(depth_dir)
        os.makedirs(config_dir)

    for color, depth, intrinsic, fname in zip(colors, depths, intrinsics, fnames):

        fname = prefix + fname
        config = {'intrinsics': intrinsic}
        config['height'], config['width'] = color.shape[:2]

        cam_params = get_cam_params_from_intr(intrinsic)
        if cam_params_shared is None:
            cam_params_shared = cam_params
        else:
            assert (cam_params_shared == cam_params)

        color_fname = fname + '.png'
        color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(color_dir, color_fname), color)


        # scale = 1000.0
        # sdepth = depth * scale
        # by converting the image data to CV_16U, the resolution of the
        # depth image data becomes 1 / scale m (0.001m in this case).

        # depth_fname = fname + '.png'
        depth_fname = fname + '_new.png'
        cv2.imwrite(os.path.join(depth_dir, depth_fname), depth.astype(np.uint16))
        # cv2.imwrite(os.path.join(depth_dir, fname + '_scaled.png'), sdepth.astype(np.uint16))

        # re_depth = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)

        # plt.imsave(os.path.join(depth_dir, fname + '.png'), depth/np.max(depth))
        # depth_fname = fname + '.pkl'
        # with open(os.path.join(depth_dir, depth_fname), 'wb') as f:
        #     pickle.dump(depth, f)

        config_fname = fname + '.pkl'
        with open(os.path.join(config_dir, config_fname), 'wb') as f:
            pickle.dump(config, f)
        
    with open(os.path.join(colmap_dir, 'shared_intrinsic.pkl'), 'wb') as f:
        pickle.dump(cam_params_shared, f)

    print('Saving finished.')

def clear_colmap_dirs(colmap_dir):
    sparse = os.path.join(colmap_dir, 'sparse')
    database = os.path.join(colmap_dir, 'database.db')
    out = os.path.join(colmap_dir, 'colmap_output.txt')

    if os.path.exists(sparse):
        shutil.rmtree(sparse)

    if os.path.exists(database):
        os.remove(database)
        os.remove(out)

def get_cam_params_from_intr(intr):
    fx = intr[0,0]
    fy = intr[1,1]
    cx = intr[0,2]
    cy = intr[1,2]
    params = [fx, fy, cx, cy]
    return ','.join([str(p) for p in params])

def get_intr_from_cam_params(basedir, hw=False):
    cameras = read_cameras_text(os.path.join(basedir, 'sparse/0/cameras.txt'))
    assert (len(cameras)==1)
    cam = cameras[list(cameras.keys())[0]]
    fx, fy, cx, cy = cam.params
    h, w = cam.height, cam.width
    intr = np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0,  0,  1]])
    if hw:
        return intr, h, w
    return intr

def get_scale(basedir, depths, configs, max_image=5, far_prefix='far'):
    images = read_images_text(os.path.join(basedir, 'sparse/0/images.txt'), 
                              max_image=None, exclude_prefix='far')

                            #   max_image=3+max_image, exclude_prefix='far')
    pts = read_points3D_text(os.path.join(basedir, 'sparse/0/points3D.txt'))          

    actual_depths = []
    colmap_depths = []

    print(len(images), "inside get scale", "printing length")

    found = 0
    max_images = len(depths)

    img_count = 0
    for key in images:

        if found == max_images:
            break
        
        im = images[key]
        # if test_prefix in im.name:
        #     continue

        pt_ids = im.point3D_ids[im.point3D_ids >= 0]
        pcd = [pts[pt3D_idx].xyz for pt3D_idx in pt_ids]
        color = [pts[pt3D_idx].rgb for pt3D_idx in pt_ids]
        pcd = np.array(pcd)
        color = np.array(color)
        # print('Feature points shape', pcd.shape)
        if pcd.shape[0] == 0:
            continue

        print("inside get scale loop", key, im.name, len(configs), len(depths))

        depth, c = get_config_and_depth(im.name, configs, depths)

        if depth is None:
            continue
        
        found += 1

        print("inside scale")
        cam_intr, cam_extr = c['intrinsics'], c['extrinsics']
        height, width = c['height'], c['width']

        N = pcd.shape[0]
        X_WC = cam_extr
        X_CW = np.linalg.inv(X_WC)
        pcd_cam = X_CW @ np.concatenate((pcd, np.ones((N, 1))), axis=1).T
        pcd_cam = pcd_cam[:3,:].T
        iz = np.argsort(pcd_cam[:, -1])[::-1]
        pcd_cam, color = pcd_cam[iz], color[iz]

        px = (pcd_cam[:, 0] * cam_intr[0,0]/pcd_cam[:, 2]) + cam_intr[0,2] 
        py = (pcd_cam[:, 1] * cam_intr[1,1]/pcd_cam[:, 2]) + cam_intr[1,2] 
        px = np.clip(np.int32(px), 0, width-1)
        py = np.clip(np.int32(py), 0, height-1)

        for pt_id in range(len(px)):
            col, row = px[pt_id], py[pt_id]
            depth_val = depth[row, col]

            colmap_depth = pcd_cam[pt_id][2]
            if (np.abs(depth_val) > 1e-4) and (np.abs(colmap_depth) > 1e-4):
                actual_depths.append(depth_val)
                colmap_depths.append(colmap_depth)

        img_count += 1
        if img_count >= max_image:
            break

    actual_depths = np.array(actual_depths)
    colmap_depths = np.array(colmap_depths)

    scales = actual_depths/colmap_depths
    scale = compute_best_fit(scales)
    return scale
        
def compute_best_fit(scales):
    return np.median(scales)
    q = 0.25
    v1, v2 = np.quantile(scales, [q, 1-q])

    scale = np.average(scales, weights=np.logical_and((scales>v1), (scales<v2)))
    return scale


def read_updated_configs(basedir):
    new_configs_fname = os.path.join(basedir, 'transformed_configs.pkl')

    with open(new_configs_fname, 'rb') as f:
        configs = pickle.load(f)

    return configs

# def copy_test_data(test_dir, destination_dir, prefix='far', ext='.png'):
#     file_lst = []
#     for fname in os.listdir(test_dir):
#         if fname.endswith(ext):
#             new_path = os.path.join(destination_dir, prefix + '_' + fname) 
#             shutil.copy(os.path.join(test_dir, fname), new_path)
#             file_lst.append(new_path)
    
#     print(f'Test data ({ext}) copied from {test_dir} to {destination_dir}: {len(file_lst)} files.')
#     return file_lst

def transform_configs(configs, transform):
    for idx in range(len(configs)):
        extr = configs[idx]['extrinsics']
        configs[idx]['extrinsics'] = transform @ extr
    return configs
