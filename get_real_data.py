import open3d
import os
import numpy as np
# import pyrealsense2 as rs
# from realsense_utils.realsense import RealsenseLocal, enable_devices
from realsense_utils.simple_multicam import MultiRealsenseLocal
import copy

CALIBRATION_DIR = "realsense_utils/camera_calibration_files"


def transform_pcd(pcd, transform):
    if pcd.shape[1] != 4:
        pcd = np.concatenate((pcd, np.ones((pcd.shape[0], 1))), axis=1)
    pcd_new = np.matmul(transform, pcd.T)[:-1, :].T
    return pcd_new


class RealSenseCameras:
    def __init__(self, cam_idx):
        self.cam_idx = cam_idx
        self.setup_cams()

    def setup_cams(self):
        serials = [
            "143122065292",  # back right
            "843112073228",  # front right bottom
            "215122255998",  # fron right top
            "840412060551",
        ]  # side

        prefix = "cam_"
        camera_names = [f"{prefix}{i}" for i in range(len(serials))]
        cam_index = [int(idx) for idx in self.cam_idx]
        cam_list = [camera_names[int(idx)] for idx in cam_index]

        calib_dir = CALIBRATION_DIR
        calib_filenames = [
            os.path.join(calib_dir, f"cam_{idx}_calib_base_to_cam.json")
            for idx in cam_index
        ]

        self.cams = MultiRealsenseLocal(
            cam_names=cam_list, calib_filenames=calib_filenames
        )

        # ctx = rs.context()  # Create librealsense context for managing devices

        # # Define some constants
        # resolution_width = 640  # pixels
        # resolution_height = 480  # pixels
        # frame_rate = 30  # fps

        # # pipelines = enable_devices(serials, ctx, resolution_width, resolution_height, frame_rate)
        # self.pipelines = enable_devices(
        #     np.array(serials)[cam_index],
        #     ctx,
        #     resolution_width,
        #     resolution_height,
        #     frame_rate,
        # )
        # self.cam_interface = RealsenseLocal()

    def get_rgb_depth(self):
        rgb_imgs = []
        depth_imgs = []
        configs = []

        for idx, cam in enumerate(self.cams.cams):
            cam_intrinsics = self.cam_interface.get_intrinsics_mat(self.pipelines[idx])
            rgb, depth = self.cam_interface.get_rgb_and_depth_image(self.pipelines[idx])

            cam.cam_int_mat = cam_intrinsics
            cam._init_pers_mat()

            rgb_imgs.append(rgb)
            depth_imgs.append(depth)
            configs.append(
                {"intrinsics": cam.cam_int_mat, "extrinsics": cam.cam_ext_mat}
            )

        return {"colors": rgb_imgs, "depths": depth_imgs, "configs": configs}


if __name__ == "__main__":
    cameras = RealSenseCameras(cam_idx=[0, 1, 2, 3])
    obs = cameras.get_rgb_depth()

    import pickle

    with open("obs.pkl", "wb") as f:
        pickle.dump(obs, f)

    # pointcloud = open3d.geometry.PointCloud(open3d.utility.Vector3dVector(pcd))
    # open3d.io.write_point_cloud("check.ply", pointcloud)
