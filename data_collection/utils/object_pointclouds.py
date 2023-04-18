import numpy as np
import open3d
import os
from glob import glob
import copy
import pickle
import cv2
from matplotlib import pyplot as plt
import random
import os
from scipy.spatial.transform import Rotation as R
import distinctipy
import trimesh
from  data_collection.utils.pointcloud_utils import project_pcd_to_image, get_pcd_from_depth

OBJECT_PCD_DIR = '/workspace/segmentation/supporting_data/object_pcds'
MAX_AREA_PER_OBJECT = (20/100*10/100)*6/7 # (25cm*25cm)
# DIST_BW_OBJS = 0.05
# MAX_SCENES_PER_TABLE = 5
CLOSEST_DIST = 0.02
TABLE_DOWNSAMMPLE_SIZE = 0.01
OBJECT_DOWNSAMPLE_SIZE = 0.01
OBJECT_SCALE = 0.9

overhead_cam_extr = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0,-1, 5],
                              [0, 0, 0, 1]])

overhead_cam_intr = np.array([[920.15307617,   0. ,        635.46258545],
                              [  0.,         917.98101807, 357.48895264],
                              [  0.  ,         0.   ,        1.        ]])
height, width = 720, 1280


def get_truncated_table_pcd(table):
    kernel = np.ones((15, 15), np.uint8)
    colormap, depthmap = project_pcd_to_image(table, overhead_cam_intr,
                overhead_cam_extr, height, width, depth=True)
    colormap = cv2.erode(colormap, kernel) 
    kept = colormap.sum(axis=2) > 50
    # plt.imsave("/workspace/kept_table.png", kept)
    # new_color = np.where(kept, colormap, np.zeros_like(colormap))
    new_depth = np.where(kept, depthmap, np.zeros_like(depthmap))
    pts, _ = get_pcd_from_depth(overhead_cam_intr, overhead_cam_extr, colormap, 
                    new_depth, filter_depth=False)

    n = len(pts)
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = open3d.utility.Vector3dVector(0.5*np.ones((n, 3), dtype=np.float64))
    # open3d.visualization.draw_geometries([pcd])
    return pcd

    # assert False


def toOpen3dCloud(points, colors=None, normals=None):

    cloud = open3d.geometry.PointCloud()
    cloud.points = open3d.utility.Vector3dVector(points.astype(np.float64))
    if colors is not None:
      if colors.max()>1:
        colors = colors/255.0
      cloud.colors = open3d.utility.Vector3dVector(colors.astype(np.float64))
    if normals is not None:
      cloud.normals = open3d.utility.Vector3dVector(normals.astype(np.float64))
    return cloud


def get_object_pcd_lst(object_pcd_dir):
    possible_meshes = glob(object_pcd_dir+"/*/*/*.obj", recursive = False)
    pcd_fnames = []
    for fname in possible_meshes:
        pcd_name = fname[:-3] + "ply"
        print(pcd_name)
        if os.path.exists(pcd_name):
            pcd_fnames.append(pcd_name)
            continue

        mesh = trimesh.load(fname)
        pts, face_ids, colors = trimesh.sample.sample_surface(mesh, 2000000, sample_color=True)

        x, y, z = pts.mean(axis=0)
        # print(x, y, z)
        # print("colors", colors.shape)
        pcd = toOpen3dCloud(pts, colors=colors[:, :3])
        open3d.io.write_point_cloud(pcd_name, pcd)
        pcd_fnames.append(pcd_name)

        # open3d.visualization.draw_geometries([pcd, mesh_frame])

    return pcd_fnames
    # names = []
    # return possible_clouds
    
# MARGIN = 0.1
# nn = 3
# MAX_DIST_OBJ = 0.5 #for 1x1 table

def create_object_pcd(_table, visualize=False):
    table = get_truncated_table_pcd(_table)

    object_pcds = get_object_pcd_lst(OBJECT_PCD_DIR)

    table_points = np.asarray(table.points)
    table_points[:, 2] = 0
    table.points = open3d.utility.Vector3dVector(table_points)
    table = table.voxel_down_sample(TABLE_DOWNSAMMPLE_SIZE)
    table_points = np.asarray(table.points)
    num_points = len(table_points)

    area = num_points*TABLE_DOWNSAMMPLE_SIZE**2
    print("table area:", area)
    max_objects = int(area/MAX_AREA_PER_OBJECT)
    print("max obj:", max_objects)

    num_objs = np.random.randint(low=1, high=max_objects)
    N = 50
    num_objs = min(N, num_objs)

    object_fnames = random.choices(object_pcds, k=num_objs)

    print("Object pcds chosen are:", object_fnames)

    combined_tmp = open3d.geometry.PointCloud()

    obj_pcd_1 = open3d.io.read_point_cloud(object_fnames[0]).scale(OBJECT_SCALE, 
                            center=np.zeros(3))

    obj_table_pts = [table_points[random.choice(range(num_points)), :]]
    point_clouds = [obj_pcd_1]
    obj_ids = [0]

    _pcd = copy.deepcopy(obj_pcd_1).translate(obj_table_pts[-1])
    combined_tmp = combined_tmp + _pcd.voxel_down_sample(OBJECT_DOWNSAMPLE_SIZE)

    for idx, object_fname in enumerate(object_fnames[1:]):
        object_pcd = open3d.io.read_point_cloud(object_fname).scale(OBJECT_SCALE, 
                                    center=np.zeros(3))

        for it in range(100):
            point = table_points[random.choice(range(num_points)),:]
            _pcd = copy.deepcopy(object_pcd).translate(point)
            _pcd = _pcd.voxel_down_sample(OBJECT_DOWNSAMPLE_SIZE)
            
            dist = np.asarray(combined_tmp.compute_point_cloud_distance(_pcd), dtype=np.float32)
            if np.all(dist > CLOSEST_DIST):
                combined_tmp = combined_tmp + _pcd
                obj_table_pts.append(point)
                point_clouds.append(object_pcd)
                obj_ids.append(idx)
                break

    num_objs = len(obj_ids)
    print(f"{num_objs} out of {len(object_fnames)} objects laid")    

    unique_colors = np.array(distinctipy.get_colors(N))

    rotations = np.random.uniform(0, 360, num_objs)
    combined = open3d.geometry.PointCloud()
    clustered = open3d.geometry.PointCloud()

    for idx, obj in enumerate(point_clouds):
        print("Object:", object_fnames[obj_ids[idx]], "placed at", 
                    obj_table_pts[idx], "rotated by", rotations[idx], "deg")
        r = R.from_euler('z', rotations[idx], degrees=True)
        obj.translate(obj_table_pts[idx])
        obj.rotate(r.as_matrix(), np.array(obj.get_center()))

        combined = combined + obj

        cluster_color = unique_colors[idx]
        obj.paint_uniform_color(cluster_color)
        clustered = clustered + obj

    diff_objects_info = {'diff_colors': unique_colors[:num_objs, :], 
                'labels': np.arange(num_objs)}

    if visualize:
        open3d.visualization.draw_geometries([table + combined])

    return combined, clustered, diff_objects_info



        # total_iter = 0
        # for key in extrs_close:
        #     colormap = project_pcd_to_image(combined, cam_intr, extrs_close[key], height, width)
        #     kernel = np.ones((nn,nn),np.float32)/nn**2
        #     colormap1 = cv2.filter2D(colormap,-1,kernel)
        #     colormap = np.where(colormap == -1, colormap, colormap1)

        #     background_img_dir = os.path.join(table_img_dir,key)

        #     background_img = cv2.imread(background_img_dir)#[::-1]
        #     background_img = cv2.cvtColor(background_img, cv2.COLOR_BGR2RGB)

        #     new_img = np.where(colormap == -1, background_img, colormap)
        #     new_img = new_img.astype(np.uint8)
        #     im_bgr = cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR)
        #     cv2.imwrite(os.path.join(modified_img_folder,str(num_imgs)+key), im_bgr)
        #     total_iter += 1
        #     if total_iter>30:
        #         break

        # print("close complete")

        # total_iter = 0
        # for key in extrs_far:

        #     colormap = project_pcd_to_image(combined, cam_intr, extrs_far[key], height, width)
        #     kernel = np.ones((nn,nn),np.float32)/nn**2
        #     colormap1 = cv2.filter2D(colormap,-1,kernel)
        #     colormap = np.where(colormap == -1, colormap, colormap1)

        #     background_img_dir = os.path.join(table_img_dir,key)
        #     background_img = cv2.imread(background_img_dir)#[::-1]
        #     background_img = cv2.cvtColor(background_img, cv2.COLOR_BGR2RGB)

        #     new_img = np.where(colormap == -1, background_img, colormap)
        #     new_img = new_img.astype(np.uint8)
        #     im_bgr = cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR)
        #     cv2.imwrite(os.path.join(modified_img_folder,str(num_imgs)+key), im_bgr)

        #     total_iter += 1
        #     if total_iter>30:
        #         break

        # print("far complete")

        # print(num_imgs)
