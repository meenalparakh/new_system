import pickle
import open3d as o3d
import struct
from glob import glob
import os

import numpy as np
import pandas as pd

def write_pointcloud(filename,xyz_points,rgb_points=None):

    """ creates a .pkl file of the point clouds generated
    """

    assert xyz_points.shape[1] == 3,'Input XYZ points should be Nx3 float array'
    if rgb_points is None:
        rgb_points = np.ones(xyz_points.shape).astype(np.uint8)*255
    assert xyz_points.shape == rgb_points.shape,'Input RGB colors should be Nx3 float array and have same size as input XYZ points'

    # Write header of .ply file
    fid = open(filename,'wb')
    fid.write(bytes('ply\n', 'utf-8'))
    fid.write(bytes('format binary_little_endian 1.0\n', 'utf-8'))
    fid.write(bytes('element vertex %d\n'%xyz_points.shape[0], 'utf-8'))
    fid.write(bytes('property float x\n', 'utf-8'))
    fid.write(bytes('property float y\n', 'utf-8'))
    fid.write(bytes('property float z\n', 'utf-8'))
    fid.write(bytes('property uchar red\n', 'utf-8'))
    fid.write(bytes('property uchar green\n', 'utf-8'))
    fid.write(bytes('property uchar blue\n', 'utf-8'))
    fid.write(bytes('end_header\n', 'utf-8'))

    # Write 3D points to .ply file
    for i in range(xyz_points.shape[0]):
        fid.write(bytearray(struct.pack("fffccc",xyz_points[i,0],xyz_points[i,1],xyz_points[i,2],
                                        rgb_points[i,0].tobytes(),rgb_points[i,1].tobytes(),
                                        rgb_points[i,2].tobytes())))
    fid.close()


# from pyntcloud import PyntCloud
# open a file, where you stored the pickled data

if __name__ == "__main__":
    experiment_dir = "/Users/meenalp/Desktop/MEng/system_repos/llmrobot/experiment_results"
    exp_name = "scene_ABP"

    observations = glob(experiment_dir + f"/{exp_name}" + "/info*.pkl")
    timestamps = sorted([obs.split(os.sep)[-1][5:-4] for obs in observations])

    final_name = experiment_dir + f"/{exp_name}/info_{timestamps[0]}.pkl" 

    file = open(final_name, 'rb')
    output_file = "output.ply"

    # dump information to that file
    data = pickle.load(file)

    # close the file
    file.close()

    objects = data['object_dicts']

    #pcd for objects id 1

    oids = list(objects.keys())

    for id in oids:

        pts = objects[id]['pcd']
        rgb = objects[id]['rgb']
        write_pointcloud(f'output_{id}.ply', pts, rgb)

# cloud = PyntCloud(pd.DataFrame(
#     # same arguments that you are passing to visualize_pcl
#     data=np.hstack((pts, rgb)),
#     columns=["x", "y", "z", "red", "green", "blue"]))

# cloud.to_file("id3.ply")
# ply_point_cloud = o3d.data.PLYPointCloud()
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(pts)
# o3d.visualization.draw_geometries([pcd]) 

# mesh = o3dtut.get_bunny_mesh()

# alpha = 0.03
# print(f"alpha={alpha:.3f}")
# mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
# mesh.compute_vertex_normals()
# o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

# import numpy as np
# import pyvista as pv

# # points is a 3D numpy array (n_points, 3) coordinates of a sphere
# cloud = pv.PolyData(pcd)
# cloud.plot()

# volume = cloud.delaunay_3d(alpha=.5)
# shell = volume.extract_geometry()
# shell.plot()
