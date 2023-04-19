import open3d
import os
import numpy as np
import pyrealsense2 as rs
from realsense_utils.realsense import RealsenseLocal, enable_devices
from realsense_utils.simple_multicam import MultiRealsenseLocal
import copy
CALIBRATION_DIR = "realsense_utils/camera_calibration_files"


def transform_pcd(pcd, transform):
    if pcd.shape[1] != 4:
        pcd = np.concatenate((pcd, np.ones((pcd.shape[0], 1))), axis=1)
    pcd_new = np.matmul(transform, pcd.T)[:-1, :].T
    return pcd_new

class Camera:

    def __init__(self, cam_idx):

        self.cam_idx = cam_idx
        self.setup_cams()

    def setup_cams(self):
        serials = [
            '143122065292', # back right
            '843112073228', # front right bottom
            '215122255998', # fron right top
            '840412060551'] # side

        prefix = 'cam_'
        camera_names = [f'{prefix}{i}' for i in range(len(serials))]
        cam_index = [int(idx) for idx in self.cam_idx]
        cam_list = [camera_names[int(idx)] for idx in cam_index]        

        calib_dir = CALIBRATION_DIR
        calib_filenames = [os.path.join(calib_dir, f'cam_{idx}_calib_base_to_cam.json') for idx in cam_index]

        self.cams = MultiRealsenseLocal(cam_names=cam_list, calib_filenames=calib_filenames)
        
        ctx = rs.context() # Create librealsense context for managing devices

        # Define some constants
        resolution_width = 640 # pixels
        resolution_height = 480 # pixels
        frame_rate = 30  # fps

        # pipelines = enable_devices(serials, ctx, resolution_width, resolution_height, frame_rate)
        self.pipelines = enable_devices(np.array(serials)[cam_index], ctx, resolution_width, resolution_height, frame_rate)
        self.cam_interface = RealsenseLocal()

    def get_real_pcd(self):
        pcd_pts = []
        pcd_dict_list = []
        cam_int_list = []
        cam_poses_list = []
        rgb_imgs = []
        depth_imgs = []
        for idx, cam in enumerate(self.cams.cams):
            cam_intrinsics = self.cam_interface.get_intrinsics_mat(self.pipelines[idx])
            rgb, depth = self.cam_interface.get_rgb_and_depth_image(self.pipelines[idx])

            cam.cam_int_mat = cam_intrinsics
            cam._init_pers_mat()
            cam_pose_world = cam.cam_ext_mat
            cam_int_list.append(cam_intrinsics)
            cam_poses_list.append(cam_pose_world)

            valid = depth < cam.depth_max
            valid = np.logical_and(valid, depth > cam.depth_min)
            depth_valid = copy.deepcopy(depth)
            depth_valid[np.logical_not(valid)] = 0.0 # not exactly sure what to put for invalid depth
            depth_imgs.append(depth_valid)

            pcd_cam = cam.get_pcd(in_world=False, filter_depth=False, rgb_image=rgb, depth_image=depth_valid)[0]
            pcd_cam_img = pcd_cam.reshape(depth.shape[0], depth.shape[1], 3)
            pcd_world = transform_pcd(pcd_cam, cam_pose_world)
            pcd_world_img = pcd_world.reshape(depth.shape[0], depth.shape[1], 3)
            pcd_dict = {
                'world': pcd_world,
                'cam': pcd_cam_img,
                'cam_img': pcd_cam,
                'world_img': pcd_world_img,
                'cam_pose_mat': cam_pose_world
            }

            pcd_pts.append(pcd_world)
            pcd_dict_list.append(pcd_dict)

        pcd_full = np.concatenate(pcd_pts, axis=0)
        return pcd_full

if __name__ == "__main__":
    
    cameras = Camera(cam_idx=[0, 1, 2, 3])
    pcd = cameras.get_real_pcd()

    pointcloud = open3d.geometry.PointCloud(open3d.utility.Vector3dVector(pcd))
    open3d.io.write_point_cloud("check.ply", pointcloud)
