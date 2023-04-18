from __future__ import print_function

import distinctipy
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import sys
import matplotlib.pyplot as plt
# from detectron2.structures import BoxMode
from sklearn.cluster import DBSCAN, KMeans
import open3d
# import pycocotools
import os, shutil
from data_collection.utils.utils import plot_ref_frame, get_frame_rotation, get_positive_normal, get_arrow
############################################################################
##                 pcd to image and vice versa                            ##
############################################################################

# part of the code taken from AIRobot
def get_pcd_from_depth(cam_intr, cam_extr, rgb_image, depth_image, 
                       filter_depth=True, depth_min=0.40, depth_max=1.3, valid_mask=False):

    cam_int_mat_inv = np.linalg.inv(cam_intr)
    H, W = rgb_image.shape[:2]

    img_pixs = np.mgrid[0: H,
                        0: W].reshape(2, -1)
    img_pixs[[0, 1], :] = img_pixs[[1, 0], :]
    _uv_one = np.concatenate((img_pixs,
                              np.ones((1, img_pixs.shape[1]))))
    uv_one_in_cam = np.dot(cam_int_mat_inv, _uv_one)


    rgb_im = rgb_image
    depth_im = depth_image
    rgb = rgb_im.reshape(-1, 3)
    depth = depth_im.reshape(-1)
    
    if filter_depth:
        valid = depth > depth_min
        valid = np.logical_and(valid,
                                depth < depth_max)
        depth = depth[valid]
        if rgb is not None:
            rgb = rgb[valid]
        uv_one_in_cam = uv_one_in_cam[:, valid]

    pts_in_cam = np.multiply(uv_one_in_cam, depth)
    cam_ext_mat = cam_extr
    pts_in_cam = np.concatenate((pts_in_cam,
                                np.ones((1, pts_in_cam.shape[1]))),
                                axis=0)
    pts_in_world = np.dot(cam_ext_mat, pts_in_cam)
    pcd_pts = pts_in_world[:3, :].T
    pcd_rgb = rgb
    if valid_mask:
        return pcd_pts, pcd_rgb, valid
    return pcd_pts, pcd_rgb

def get_pcds(colors, depths, configs, remove_outliers=True):
    """Reconstruct top-down heightmap views from multiple 3D pointclouds."""
    xyzs_list = []
    colors_list = []

    cam_extrs = []

    print('inside get pcd', len(colors))
    for color, depth, config in zip(colors, depths, configs):
        cam_intr = config['intrinsics']
        cam_extr = config['extrinsics']
        xyz, color = get_pcd_from_depth(cam_intr, cam_extr, color, depth)

        if remove_outliers:
            point_cloud = open3d.geometry.PointCloud()
            point_cloud.points = open3d.utility.Vector3dVector(xyz)
            point_cloud.colors = open3d.utility.Vector3dVector(np.array(color)/255.0)
            point_cloud, ind = point_cloud.remove_statistical_outlier(nb_neighbors=10, std_ratio=0.1)
            xyz = np.asarray(point_cloud.points)
            color = np.uint8(255*np.asarray(point_cloud.colors))

        xyzs_list.append(xyz)
        colors_list.append(color)
        cam_extrs.append(cam_extr)

    xyzs = np.concatenate(xyzs_list, axis=0)
    colors = np.concatenate(colors_list, axis=0)
    print('point cloud size:', xyzs.shape)
    
    assert (xyzs.shape[1]==3)
    assert (colors.shape[1]==3)

    return xyzs, colors, cam_extrs


def project_pcd_to_image(pcd, cam_intr, cam_extr, height, width,
                labels=None, depth=False, background_val=0):
    '''
    pcd: Nx3
    cam_intr: 3x3
    c2w: rotation, position
    labels = if given, returns the projected labels, bounding box, and colormap
    '''

    xyzs = np.asarray(pcd.points)
    colors = np.uint8(255*np.asarray(pcd.colors))

    N = xyzs.shape[0]
    X_WC = cam_extr
    X_CW = np.linalg.inv(X_WC)
    pcd_cam = X_CW @ np.concatenate((xyzs, np.ones((N, 1))), axis=1).T
    pcd_cam = pcd_cam[:3,:].T

    iz = np.argsort(pcd_cam[:, -1])[::-1]
    pcd_cam, colors = pcd_cam[iz], colors[iz]

    fx = cam_intr[0,0]
    fy = cam_intr[1,1]
    cx = cam_intr[0,2]
    cy = cam_intr[1,2]

    px = (pcd_cam[:, 0] * fx/pcd_cam[:, 2]) + cx 
    py = (pcd_cam[:, 1] * fy/pcd_cam[:, 2]) + cy 

    px = np.clip(np.int32(px), 0, width-1)
    py = np.clip(np.int32(py), 0, height-1)

#    colormap = np.zeros((height, width, colors.shape[-1]), dtype=np.uint8)
    colormap = background_val*np.ones((height, width, colors.shape[-1]))
    colormap[py, px, :] = colors

    if depth:
        # return colormap, None
        depthmap = np.zeros((height, width), dtype=np.float32)
        depthmap[py, px] = pcd_cam[:, 2] 
        return colormap, depthmap

    if labels is not None:
        raise NotImplementedError
        max_label = labels.max()
        c = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        c[labels < 0] = 0

        labelmap = np.zeros((height, width, 3))
        labelmap[py, px, :] = c[:,:3]
        # np.savetxt('/workspace/segmentation/labels.txt', 
        #         labelmap, fmt='%.2e')        
        # assert False
        return colormap, labelmap
    
    return colormap

############################################################################
##                 point cloud processing                                 ##
############################################################################

def save_pcd(xyzs, colors, cam_extrs, fname, save_pcd=True, 
                visualize_pcd=False, 
                downsample_voxel_size=0.001,
                visualize_camera=False):
    xyzs = np.array(xyzs)
    colors = np.array(colors)/255.0

    point_cloud = open3d.geometry.PointCloud()
    point_cloud.points = open3d.utility.Vector3dVector(xyzs)
    point_cloud.colors = open3d.utility.Vector3dVector(colors)

    print('Downsamlpling point cloud.')
    if downsample_voxel_size is not None:
        point_cloud = point_cloud.voxel_down_sample(voxel_size=downsample_voxel_size)
        print('Downsampled size:', len(point_cloud.points))

    if save_pcd:
        open3d.io.write_point_cloud(fname, point_cloud)
        print('Point cloud saved.')

    geoms_lst = [point_cloud]
    if visualize_camera:

        for cam_idx in range(len(cam_extrs)):
            cam_extr = cam_extrs[cam_idx]
            ref_frame = plot_ref_frame(cam_extr)
            geoms_lst.extend(ref_frame)

    if visualize_pcd:
        open3d.visualization.draw_geometries(geoms_lst)

    print('Open3D pcd created.')
    return point_cloud

def get_table_cluster(pcd, eps=0.005, min_points=200, get_normal=False):

    with open3d.utility.VerbosityContextManager(
            open3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(
            pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))

    max_label = labels.max()
    
    clusters = []
    cluster_plane_params = []
    planar_pts_count = []
    normal_vecs = []

    print(f'Num components found: {max_label+1}')
    for l in range(max_label + 1):
        indices = np.where(labels == l)[0]
        cluster_pcd = pcd.select_by_index(indices)
        plane_params, num_inliers = plane_segmentation(cluster_pcd, None, visualize=False, 
                            return_count=True)
        # print('Post processing PCD...')
        # normal_vec = get_positive_normal(plane_params, pcd)
        normal_vec = get_positive_normal(plane_params, cluster_pcd)

        clusters.append(cluster_pcd)
        cluster_plane_params.append(plane_params)
        normal_vecs.append(normal_vec)
        planar_pts_count.append(num_inliers)

    print(f'Counts max, avg: {max(planar_pts_count), np.mean(planar_pts_count)}')
    cluster_idx = np.argmax(planar_pts_count)
    cluster = clusters[cluster_idx]
    plane_params = cluster_plane_params[cluster_idx]
    normal_vec = normal_vecs[cluster_idx]

    if get_normal:
        arrow_mesh = get_arrow(vec=-normal_vec)
        open3d.visualization.draw_geometries([pcd, arrow_mesh])

        return cluster, plane_params, -normal_vec
    else:
        return cluster, plane_params, None
    


def dbscan_clustering(pcd, pcd_file, save_fname, eps=0.01, min_points=20, 
            save_pcd=True, get_object_pcds=False, object_pcds_dir=None, visualize=True):
    if pcd is None:
        pcd = open3d.io.read_point_cloud(pcd_file)

    with open3d.utility.VerbosityContextManager(
            open3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(
            pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))

    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")

    N = 50
    unique_colors = np.array(distinctipy.get_colors(N))
    # colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors = np.take(unique_colors, labels, axis=0)

    colors[labels < 0] = 0
    pcd.colors = open3d.utility.Vector3dVector(colors[:, :3])

    if visualize:
        open3d.visualization.draw_geometries([pcd])

    if save_pcd:
        open3d.io.write_point_cloud(save_fname, pcd)

    if max_label < 0:
        raise ValueError("Number of clusters is 0 in the point cloud.")

    label_array = np.arange(max_label+1) / (max_label if max_label > 0 else 1)
    diff_colors = unique_colors[:max_label+1, :3]
    diff_objects_info = {'diff_colors': diff_colors, 'labels': np.arange(max_label+1)}

    if get_object_pcds:
        ## saving the object point clouds
        if object_pcds_dir is not None:
            if os.path.exists(object_pcds_dir):
                shutil.rmtree(object_pcds_dir)
            os.makedirs(object_pcds_dir)

        for l in range(max_label + 1):
            indices = np.where(labels == l)[0]
            object_pcd = pcd.select_by_index(indices)
            object_pcd = object_pcd.paint_uniform_color(np.array([1.0, 1.0, 1.0]))
            object_fname = os.path.join(object_pcds_dir, f'{l}.ply')
            open3d.io.write_point_cloud(object_fname, object_pcd)

    return pcd, diff_objects_info

def kmeans_clustering(pcd, pcd_file, save_fname, save_pcd=True):
    if pcd is None:
        pcd = open3d.io.read_point_cloud(pcd_file)

    pts = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    X = np.concatenate((pts, colors), axis=1)
    print('feature shape', X.shape)

    kmeans = KMeans(n_clusters=4, random_state=0).fit(X)

    point_cloud = open3d.geometry.PointCloud()
    point_cloud.points = open3d.utility.Vector3dVector(pts)
    labels = np.zeros_like(colors)
    print(kmeans.labels_.shape)
    labels[:,0] = kmeans.labels_[:]*0.5
    labels[:,1] = kmeans.labels_[:]
    labels[:,2] = kmeans.labels_[:]
    labels = labels/np.max(labels)
    point_cloud.colors = open3d.utility.Vector3dVector(labels)

    open3d.visualization.draw_geometries([point_cloud])

    if save_pcd:
        open3d.io.write_point_cloud(save_fname, point_cloud)

def plane_segmentation(pcd, pcd_file, visualize=False, return_count=False):
    if pcd is None:
        pcd = open3d.io.read_point_cloud(pcd_file)
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                                ransac_n=3,
                                                num_iterations=100)
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    if visualize:
        inlier_cloud = pcd.select_by_index(inliers)
        inlier_cloud.paint_uniform_color([1.0, 0, 0])
        outlier_cloud = pcd.select_by_index(inliers, invert=True)
        open3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

    if return_count:
        return [a, b, c, d], len(inliers)
    return [a, b, c, d]

def postprocess_pointcloud(pcd, plane_params, normal_vec,
            revert_normal=False, visualize=True, keep_table=False):
    if normal_vec is None:
        unit_vec = get_positive_normal(plane_params, pcd)
        if revert_normal:
            unit_vec = -unit_vec
    else:
        unit_vec = normal_vec

    R_WCurrent = get_frame_rotation(unit_vec)

    a, b, c, d = plane_params
    pt_on_plane = -d*np.array([a, b, c])/(a**2 + b**2 + c**2)

    pts = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    pts = pts - pt_on_plane
    T1 = np.eye(4); T1[:3,3] = -pt_on_plane

    R_CurrentW = np.linalg.inv(R_WCurrent)
    pts = R_CurrentW @ (pts.T)
    pts = pts.T
    T2 = np.eye(4); T2[:3,:3] = R_CurrentW

    if keep_table:
        selected_pts = pts[pts[:, 2] > -0.01]
    else:
        selected_pts = pts[pts[:, 2] > 0.02]

    median_x = np.median(selected_pts[:,0])
    median_y = np.median(selected_pts[:,1])
    pts[:, :2] = pts[:, :2] - [median_x, median_y]
    T3 = np.eye(4); T3[:3,3] = -np.array([median_x, median_y, 0])

    transform = T3 @ (T2 @ T1)

    pcd.points = open3d.utility.Vector3dVector(pts)

    margin = 3.0
    min_bound = np.array([-margin, -margin, -0.01])
    max_bound = np.array([margin, margin, np.inf])
    bb = open3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    pcd = pcd.crop(bb)

    # print('removing outlier...')
    # pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=0.05)
    # print('outlier removal done.')

    if visualize:
        geoms_lst = [pcd]
        ref_frame = plot_ref_frame(np.eye(4))
        geoms_lst.extend(ref_frame)
        open3d.visualization.draw_geometries(geoms_lst)
 
    return pcd, transform


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    outlier_cloud.paint_uniform_color([1,0,0])
    inlier_cloud.paint_uniform_color([0.8,0.8,0.8])
    open3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
