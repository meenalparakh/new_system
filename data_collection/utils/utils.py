from __future__ import print_function

# import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
# import sys
# import matplotlib.pyplot as plt
# from detectron2.structures import BoxMode
# from sklearn.cluster import DBSCAN, KMeans
import open3d
# import pycocotools


############################################################################
##                vector 3D functions                                     ##
############################################################################

def get_transformed_pt(X, pt):
    scale = 1.0
    x, y, z = pt
    new_pt = X @ np.array([[scale*x],[scale*y],[scale*z],[1.0]])
    return list(new_pt[:3,0])

def get_positive_normal(plane_params, pcd):
    plane_model = np.array(plane_params).reshape((4, 1))
    xyz = np.asarray(pcd.points)
    N = xyz.shape[0]

    extended = np.concatenate((xyz, np.ones((N, 1))), axis=1)
    distances = (extended @ plane_model)

    margin_low = 0.02
    margin_high = 0.15

    pts_above = ((distances < margin_high) & (distances > margin_low)).sum()
    pts_below = ((distances > (-margin_high)) & (distances < (-margin_low))).sum()

    # pts_above = np.sum(distances > 0.02)
    # pts_below = np.sum(distances < -0.02)

    normal_vec = plane_model[:3, 0]
    unit_normal_vec = normal_vec/np.linalg.norm(normal_vec)
    if pts_above > pts_below:
        return unit_normal_vec
    else:
        return -unit_normal_vec

def get_frame_rotation(vec):
    if abs(vec[2]) > 1e-2:
        vec1 = np.array([1, 0, -vec[0]/vec[2]])
    elif abs(vec[1]) > 1e-2:
        vec1 = np.array([1, -vec[0]/vec[1], 0])
    else:
        vec1 = np.array([-vec[2]/vec[0], 0, 1])
        
    vec1 = vec1/np.linalg.norm(vec1)
    vec2 = np.cross(vec1, vec)
    R_WCurrent = np.column_stack((vec2, vec1, vec))
    return R_WCurrent

def generate_random_camera_matrices(n, p_type='near'):
    near_rad = [0.25, 0.75]
    far_rad = [1.50, 2.00]

    cam_extrs = []
    geoms = []

    default_rotation = R.from_euler('ZYX', [np.pi/2, 0, -np.pi/2], 
                degrees=False)


    if p_type == 'near':
        rs = np.random.uniform(near_rad[0], near_rad[1], n)
    else:
        rs = np.random.uniform(far_rad[0], far_rad[1], n)

    for r in rs:
        default_position = r*np.array([1, 0, 0])
        T = np.eye(4)
        # print(default_rotation, default_position)

        T[:3,:3] = default_rotation.as_matrix()
        T[:3,3] = default_position
        # geoms.extend(plot_ref_frame(T))

        x_angle = np.random.uniform(0, 2*np.pi)
        y_angle = np.random.uniform(-np.pi, 0)
        z_angle = np.random.uniform(0, 2*np.pi)

        # print('Angles', x_angle, y_angle, z_angle)
        matrix = R.from_euler('ZYX', [z_angle, y_angle, x_angle], degrees=False)
        rotate_by = np.eye(4); rotate_by[:3,:3] = matrix.as_matrix()

        cam_extr = rotate_by @ T
        geoms.extend(plot_ref_frame(cam_extr))
        cam_extrs.append(cam_extr)

    return cam_extrs, geoms


############################################################################
##                 coordinate frame visualization                         ##
############################################################################

def draw_geometries(pcds):
    open3d.visualization.draw_geometries(pcds)

def vector_magnitude(vec):
    magnitude = np.sqrt(np.sum(vec**2))
    return(magnitude)


def calculate_zy_rotation_for_arrow(vec):
    # Rotation over z axis of the FOR
    gamma = np.arctan2(vec[1], vec[0])
    Rz = np.array([[np.cos(gamma),-np.sin(gamma),0],
                   [np.sin(gamma),np.cos(gamma),0],
                   [0,0,1]])
    # Rotate vec to calculate next rotation
    vec = Rz.T@vec.reshape(-1,1)
    vec = vec.reshape(-1)
    # Rotation over y axis of the FOR
    beta = np.arctan2(vec[0], vec[2])
    Ry = np.array([[np.cos(beta),0,np.sin(beta)],
                   [0,1,0],
                   [-np.sin(beta),0,np.cos(beta)]])
    return(Rz, Ry)

def create_arrow(scale=10):
    cone_height = scale*0.2
    cylinder_height = scale*0.8
    cone_radius = scale/10
    cylinder_radius = scale/20
    mesh_frame = open3d.geometry.TriangleMesh.create_arrow(cone_radius=cone_radius,
        cone_height=cone_height,
        cylinder_radius=cylinder_radius,
        cylinder_height=cylinder_height)
    return(mesh_frame)

def get_arrow(origin=[0, 0, 0], end=None, vec=None, color=[1,0,0]):
    scale = 0.05
    Ry = Rz = np.eye(3)
    T = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    T[:3, -1] = origin
    if end is not None:
        vec = np.array(end) - np.array(origin)
    elif vec is not None:
        vec = np.array(vec)
    if end is not None or vec is not None:
        # scale = vector_magnitude(vec)
        Rz, Ry = calculate_zy_rotation_for_arrow(vec)
    mesh = create_arrow(scale)
    # Create the arrow
    mesh.rotate(Ry, center=np.array([0, 0, 0]))
    mesh.rotate(Rz, center=np.array([0, 0, 0]))
    mesh.translate(origin)
    mesh.paint_uniform_color(color)

    return(mesh)

def plot_ref_frame(T):
    cam_origin = get_transformed_pt(T, [0,0,0])
    cam_x = get_transformed_pt(T, [1,0,0])
    cam_y = get_transformed_pt(T, [0,1,0])
    cam_z = get_transformed_pt(T, [0,0,1])
    arrow_x = get_arrow(origin=cam_origin, end=cam_x, color=[1,0,0])
    arrow_y = get_arrow(origin=cam_origin, end=cam_y, color=[0,1,0])
    arrow_z = get_arrow(origin=cam_origin, end=cam_z, color=[0,0,1])
    return [arrow_x, arrow_y, arrow_z]

