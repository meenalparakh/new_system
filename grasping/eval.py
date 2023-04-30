#!/usr/bin/env python3

import os
import sys
import torch
import numpy as np
import copy
import random
import meshcat

from grasping.model.contactnet import ContactNet
import grasping.model.utils.config_utils as config_utils
import grasping.model.mesh_utils as mesh_utils
import argparse
from torch_geometric.nn import fps
from grasping.test_meshcat_pcd import meshcat_pcd_show as viz_points
from grasping.test_meshcat_pcd import sample_grasp_show as viz_grasps

import pickle

random.seed(0)
np.random.seed(0)

def obtain_pcd_n_mask(scene_dir, obj_id):
    with open(os.path.join(scene_dir, "pcd_grasp_info.pkl"), 'rb') as f:
        d = pickle.load(f)
    xyz, seg, obj_dicts = d["xyzs"], d["segs"], d["obj_id"]

    if obj_id < 0:
        obj_id = list(obj_dicts.keys())
    if not isinstance(obj_id, list):
        obj_id = [obj_id]

    mask = []
    for oid in obj_id:
        mask_ = (seg == obj_dicts[oid]["mask_id"]).reshape((xyz.shape[0], 1))
        mask.append(mask_)
    return xyz, mask, obj_id

def initialize_net(config_file, load_model, save_path, args, device=None):
    print('initializing net')
    torch.cuda.empty_cache()
    config_dict = config_utils.load_config(config_file)
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Grasping running on device", device)

    model = ContactNet(config_dict, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    if load_model==True:
        print('loading model')
        checkpoint = torch.load(save_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    return model, optimizer, config_dict

def cgn_infer(cgn, pcd, obj_mask=None, threshold=0.5):

    print("pcd shape:", pcd.shape)
    if obj_mask is not None:    
        print("mask shape:", obj_mask.shape)

    cgn.eval()
    n = pcd.shape[0]
    if n > 20000:
        print("Size larger than 20,000, downsampling the pcd")
        downsample = np.array(random.sample(range(n), 20000))
    else:
        print("Size smaller than 20,000, oversampling the pcd")
        remaining = 20000 - n
        idx = np.arange(n)
        other_idx = np.random.randint(0, n, size=remaining)
        downsample = np.concatenate((idx, other_idx))
    pcd = pcd[downsample, :]

    print("Final pcd shape:", pcd.shape)

    pcd = torch.Tensor(pcd).to(dtype=torch.float32).to(cgn.device)
    batch = torch.zeros(pcd.shape[0]).to(dtype=torch.int64).to(cgn.device)
    idx = fps(pcd, batch, 2048/pcd.shape[0])
    #idx = torch.linspace(0, pcd.shape[0]-1, 2048).to(dtype=torch.int64).to(cgn.device)

    if obj_mask is not None:
        obj_mask = torch.Tensor(obj_mask[downsample]).to(cgn.device)
        print("device for idx and obj_mask", idx.device, obj_mask.device)
        obj_mask = obj_mask[idx]
    else:
        obj_mask = torch.ones(idx.shape[0])

    points, pred_grasps, confidence, pred_widths, _, pred_collide = cgn(pcd[:, 3:], pos=pcd[:, :3], batch=batch, idx=idx, obj_mask=[obj_mask])
    sig = torch.nn.Sigmoid()
    confidence = sig(confidence)
    confidence = confidence.reshape(-1,1)
    pred_grasps = torch.flatten(pred_grasps, start_dim=0, end_dim=1).detach().cpu().numpy()

    confidence = (obj_mask.detach().cpu().numpy() * confidence.detach().cpu().numpy()).reshape(-1)
    pred_widths = torch.flatten(pred_widths, start_dim=0, end_dim=1).detach().cpu().numpy()
    points = torch.flatten(points, start_dim=0, end_dim=1).detach().cpu().numpy()

    threshold = np.max(confidence)*0.9

    print("success mask shape", confidence.shape)
    print("grasps shape:", pred_grasps.shape)
    
    success_mask = (confidence > threshold).nonzero()[0]
    if len(success_mask) == 0:
        print('failed to find successful grasps')
        return None, None, None
    # print(pred_grasps[success_mask].shape)
    return pred_grasps[success_mask], confidence[success_mask], downsample
    # return pred_grasps[success_mask], confidence[success_mask], downsample


def visualize(pcd, grasps, mc_vis=None):
    if mc_vis is None:
        mc_vis = meshcat.Visualizer(zmq_url='tcp://127.0.0.1:6000')
        mc_vis['scene/'].delete()
        mc_vis['home/'].delete()
        mc_vis.delete()
    viz_points(mc_vis, pcd, name='pointcloud', color=(0,0,0), size=0.002)
    grasp_kp = get_key_points(grasps)
    viz_grasps(mc_vis, grasp_kp, name='gripper/', freq=1)

def get_key_points(poses, include_sym=False):
    gripper_object = mesh_utils.create_gripper('panda', root_folder='./')
    gripper_np = gripper_object.get_control_point_tensor(poses.shape[0])
    hom = np.ones((gripper_np.shape[0], gripper_np.shape[1], 1))
    gripper_pts = np.concatenate((gripper_np, hom), 2).transpose(0,2,1)
    pts = np.matmul(poses, gripper_pts).transpose(0,2,1)
    return pts

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--viz', type=bool, default=True, help='whether or not to debug visualize in meshcat')
    parser.add_argument('--load_path', type=str, default='./checkpoints/current.pth', help='path to load model from')
    parser.add_argument('--config_path', type=str, default='./model/', help='path to config yaml file')
    parser.add_argument('--scene-dir', type=str, help='path to scene_directory')
    parser.add_argument('--scene-type', type=str, help='path to scene_directory', default="real")
    parser.add_argument('--obj-id', type=int, help='path to scene_directory', default=-1)
    parser.add_argument('--model', type=str, default='sg_score')
    parser.add_argument('--pos_weight', default=1.0)
    parser.add_argument('--threshold', default=0.9, type=float, help='success threshold for grasps')
    args = parser.parse_args()

    ### Initialize model
    contactnet, optim, config = initialize_net(args.config_path, load_model=True, save_path=args.load_path, args=args)

    ### Get pcd, pass into model
    print('inferring.')
    pointcloud, obj_masks, obj_id = obtain_pcd_n_mask(args.scene_dir, obj_id=args.obj_id)
    print("point cloud obtained from", args.scene_dir)
    grasps = {}
    for id, mask in zip(obj_id, obj_masks):
        pred_grasps, pred_success, downsample = cgn_infer(contactnet, pointcloud, mask, threshold=args.threshold)
        grasps[id] = {"pred_grasps": pred_grasps, "pred_success": pred_success}
        n = 0 if pred_success is None else pred_grasps.shape[0]
        print('model pass.', id, n, 'grasps found.')

    with open(os.path.join(args.scene_dir, "contactnet_grasps.pkl"), "wb") as f:
        pickle.dump(grasps, f)
    print("pickle dump finished")

    ### Visualized
    # visualize(pointcloud, pred_grasps)
