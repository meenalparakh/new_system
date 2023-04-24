#!/usr/bin/env python3

import os
import sys
import torch
import pybullet as p
import numpy as np
import lcm
import threading
import rospy
import time
import cv2 as cv
import copy
import random
import faulthandler

os.environ["PYOPENGL_PLATFORM"] = "egl"

from franka_interface import ArmInterface

from realsense_lcm.config.default_multi_realsense_cfg import get_default_multi_realsense_cfg
from realsense_lcm.utils.pub_sub_util import RealImageLCMSubscriber, RealCamInfoLCMSubscriber
from realsense_lcm.multi_realsense_publisher_visualizer import subscriber_visualize

cn_path = os.path.join(os.getenv('HOME'), 'graspnet/graspnet/subgoal-net/')
sys.path.append(cn_path)
ik_path = os.path.join(os.getenv('HOME'), 'pybullet-planning/')
sys.path.append(ik_path)

# from model.subgoalnet import SubgoalNet
from model.contactnet import ContactNet
import model.utils.config_utils as config_utils
from train import initialize_loaders, initialize_net
from scipy.spatial.transform import Rotation as R
import argparse
from torch_geometric.nn import fps
from test_meshcat_pcd import viz_pcd as V
from franka_ik import FrankaIK
from simple_multicam import MultiRealsense
from real_world_interface import WorldInterface
#from panda_ndf_utils.panda_mg_wrapper import FrankaMoveIt
from evaluation.collision_checker import PointCollision

class PandaReal():
    def __init__(self, model, config, robot='franka', viz=True):

        self.panda = ArmInterface()
        self.panda.set_joint_position_speed(3)
        self.joint_names = self.panda._joint_names
        print('joint names: ', self.joint_names)
        self.ik_helper = FrankaIK(gui=True, base_pos=[0, 0, 0], robot=robot)
        self.model = model
        self.config = config
        self.world = WorldInterface(viz)

        self.robot_name = robot
        self.lcc = PointCollision(None, robotiq=robot=='franka_2f140')


    def build_6d_grasps(self, contact_pts, z1, z2, w, d, device):
        '''
        builds full 6 dimensional grasps based on generated vectors, width, and pointcloud
        returns point between contact points at top of fingers. will need to transpose to specific gripper
        '''
        max_width = 0.2
        num_bins = 21
        bin_size = max_width/(num_bins-1)

        base_dirs = z1/(torch.unsqueeze(torch.linalg.norm(z1, dim=1), dim=0).transpose(0, 1))
        inner = torch.sum((base_dirs * z2), dim=1)
        prod = torch.unsqueeze(inner, -1)*base_dirs
        approach_dirs = (z2 - prod)/(torch.unsqueeze(torch.linalg.norm(z2, dim=1), dim=0).transpose(0, 1))

        if w.shape[1] != 1:
            wb = torch.argmax(w,1)
        else:
            wb = w
        w = bin_size*wb + bin_size/2
        grasps = torch.eye(4).repeat(len(contact_pts),1,1).to(device)
        grasps[:,:3,0] = base_dirs
        grasps[:,:3,2] = -approach_dirs
        grasp_y = torch.cross(grasps.clone()[:,:3,2], grasps.clone()[:,:3,0])
        grasps[:,:3,1] = grasp_y / torch.linalg.norm(grasp_y, dim=1, keepdims=True)
        grasps[:,:3,3] = contact_pts - d*grasps.clone()[:,:3,2].to(device) + (w/2).reshape(-1,1)*grasps.clone()[:,:3,0].to(device)
        return grasps
        
    def infer(self, pcd, obj_mask, threshold=0.001):
        '''
        Forward pass rendered point cloud into the model
        '''

        self.model.eval()
        pcd = torch.Tensor(pcd).cuda()
        batch_list = torch.zeros(pcd.shape[0]).long().to(self.model.device)
        fps_pcd = copy.deepcopy(pcd).view(-1, 3).cuda()
        idx = fps(fps_pcd, batch_list, 2048/fps_pcd.shape[0])

        pcd = pcd.view(-1,3).to(self.model.device).float()

        pred_grasp_poses, output_list = self.model(pcd[:, 3:], pcd[:, :3], batch_list.to(self.model.device), idx.to(self.model.device))
        points, pred_s, pred_b, pred_a, pred_w, pred_d = output_list

        sig = torch.nn.Sigmoid()
        pred_s = sig(pred_s)
        sm = torch.nn.Softmax(dim=1)
        pred_w = sm(pred_w)

        depth_mask = pred_d.reshape(-1) < max_depth
        bin_size = 0.01
        pred_w_scalar = (bin_size*torch.argmax(pred_w,1) + bin_size/2).reshape(-1)
        width_mask = pred_w_scalar < max_width
        ee_mask = depth_mask * width_mask
        pcd = pcd[idx]
        grasps = self.build_6d_grasps(points, pred_b, pred_a, pred_w, pred_d+0.1034, self.model.device)
        grasps = grasps[obj_mask[idx.detach().cpu().numpy()]]
        
        kps = self.model.get_key_points(torch.tensor(grasps))
        V(pcd.detach().cpu().numpy(), 'pcd', clear=True)
        V(kps.detach().cpu().numpy(), 'g/', grasps=True)
        
        return grasps, idx

    def offset_grasp(self, grasp, dist):
        offset = np.eye(4)
        offset[2, 3] = -dist
        offset = np.matmul(offset, np.linalg.inv(grasp))
        offset = np.matmul(grasp, offset)
        grasp = np.matmul(offset, grasp)
        return grasp

    def rotate_grasp(self, grasp, theta):
        z_r = R.from_euler('z', theta, degrees=False)
        z_rot = np.eye(4)
        z_rot[:3,:3] = z_r.as_matrix()
        z_rot = np.matmul(z_rot, np.linalg.inv(grasp))
        z_rot = np.matmul(grasp, z_rot)
        grasp = np.matmul(z_rot, grasp)
        return grasp
    
    def pose2tuple(self, pose):
        pos = pose[:3,-1]
        quat = R.from_matrix(pose[:3,:3]).as_quat()
        return tuple([*pos, *quat])
    
    def motion_plan(self, grasp, end_grasp, current_joints):
        '''
        input:
            grasp - a single 4x4 pose of end effector
            current_joints - current pose of the robot
        output:
            joint trajectory for entire motion plan (including pre-grasp pose)
        '''
        self.reset_joints = list(self.panda.joint_angles().values())
        # rotate grasp by pi/2 about the z axis (urdf fix)
        g_list = [grasp, end_grasp]
        fixed_grasps = []
        for g in g_list:
            #g = self.offset_grasp(g, 0.105)
            fixed_grasps.append(self.rotate_grasp(g, np.pi/2))

        grasp = fixed_grasps[0]
        end_grasp = fixed_grasps[1]
        pose = self.pose2tuple(grasp)
        sol_jnts = list(self.get_ik(pose))

        pre_grasp = self.offset_grasp(grasp, 0.1)n
        pre_place_grasp = self.offset_grasp(end_grasp, 0.1)
        
        lift_grasp = copy.deepcopy(grasp)
        lift_grasp[2, 3] += 0.2            
        
        pre_pose = self.pose2tuple(pre_grasp)
        lift_pose = self.pose2tuple(lift_grasp)
        pre_place_pose = self.pose2tuple(pre_place_grasp)
        end_pose = self.pose2tuple(end_grasp)
        
        pre_jnts = self.get_ik(pre_pose)
        lift_jnts = self.get_ik(lift_pose)
        pre_place_jnts = self.get_ik(pre_place_pose)
        place_jnts = self.get_ik(end_pose)
        if None in [pre_jnts, sol_jnts, lift_jnts, pre_place_jnts, place_jnts]:
            print('having a bad time with IK')
            return None

        pick_plan = self.get_plan([self.pb_robot.arm.get_jpos(), pre_jnts, sol_jnts], pcd=pcd)
        place_plan = self.get_plan([sol_jnts, lift_jnts, place_jnts], pcd=pcd)
        reset_plan = self.get_plan([place_jnts, pre_place_jnts, self.reset_joints], pcd=pcd)
 
        if pick_plan is not None and place_plan is not None and reset_plan is not None:
            return (pick_plan, place_plan, reset_plan)
        else:
            return None

    def get_ik(self, pose, pcd=None):
        jnts = self.ik_helper.get_feasible_ik(pose, target_link=False, pcd=pcd)
        # if jnts is None:
        #     jnts = self.ik_helper.get_ik(pose)
        return jnts                

    def get_plan(self, waypoint_list, pcd=None):
        start_jnts = waypoint_list[0]
        end_jnts = waypoint_list[1]
        plan = []
        for i, w in enumerate(waypoint_list[:-1]):
            subplan = self.ik_robot.plan_joint_motion(w, waypoint_list[i+1], pcd=pcd)
            if subplan is not None:
                plan += subplan
            else:
                print('could not find plan')
                return None
        return plan
    
    def plan2dict(self, plan):
        dict_plan = [dict(zip(self.joint_names, val.tolist())) for val in plan]
        return dict_plan

    def execute(self, plan):
        # plan is a tuple!
        self.panda.hand.open()
        s0 = self.panda.execute_position_path(self.plan2dict(plan[0]))
        self.panda.hand.close()
        s1 = self.panda.execute_position_path(self.plan2dict(plan[1]))
        time.sleep(5)
        val = input('press enter to reset')
        self.panda.hand.open()
        s2 = self.panda.execute_position_path(self.plan2dict(plan[2]))
        #self.panda.move_to_neutral()
        print('execute attempt:', s0, s1)
        return (s0, s1)
        
        
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--viz', type=bool, default=True, help='whether or not to debug visualize in meshcat')
    parser.add_argument('--load_path', type=str, default='./checkpoints/width_dl_dc/current.pth', help='path to load model from')
    parser.add_argument('--config_path', type=str, default='./model/', help='path to config yaml file')
    parser.add_argument('--aruco', default=True)
    parser.add_argument('--robot', default='franka', help='options are: franka, franka_wide, franka_long, franka_2f140')
    args = parser.parse_args()

    faulthandler.enable()
    
    rospy.init_node('panda')
    done = False
    while not done and not rospy.is_shutdown():
        # Initialize model, initialize robot stuff
        if args.load_path is not None:
            load = True
            load_path = os.path.join(os.getenv('GACGN'), args.load_path)

        contactnet, optim, config = initialize_net(config_path, load_model=load, load_path=load_path)
        data_config = config['data']
        panda_robot = PandaReal(contactnet, config, args.robot, args.viz)

        if args.aruco:
            obj_tf = panda_robot.world.ar_subgoal()
            print('object transform acquired.')

        # Get pcd, pass into model
        pcd_list = []
        manual = False
        val = input('press enter to record pointcloud')
        init_pcd, rgb = panda_robot.world.get_pcd_ar(pcd_filter=manual)
        if manual:
            obj_mask = panda_robot.world.manual_segment(init_pcd)
        else:
            obj_mask = panda_robot.world.segment(rgb)
            obj_mask = obj_mask[np.nonzero(init_pcd[:,0]>0.2)]
            init_pcd = init_pcd[np.nonzero(init_pcd[:,0]>0.2)]
        V(init_pcd, 'init', clear=True)

        init_pcd_hom = np.concatenate((init_pcd, np.ones((init_pcd.shape[0], 1))), axis=1)
        obj_pcd = np.array([obj_mask]).T*init_pcd_hom
        if not args.aruco:
            obj_tf = panda_robot.world.choose_subgoal(obj_pcd)
            V(obj_pcd, 'obj')
            trans_tf = np.eye(4)
            rot_tf = np.eye(4)
            while not happy:
                obj_pcd = np.array([obj_mask]).T*init_pcd_hom
                trans_tf[:3,3] = [pose['x'], pose['y'], pose['z']]
                rot_tf[:3,:3] = R.from_euler(pose['r_axis'], -np.pi/2).as_matrix()
                # obj_pcd = np.matmul(obj_tf, obj_pcd.T).T[:,:3]
                obj_pcd = np.matmul(rot_tf, copy.deepcopy(obj_pcd).T)
                obj_pcd = np.matmul(trans_tf, obj_pcd).T[:,:3]
                for k in pose.keys():
                    i = input(k + ': ')
                    if i != '':
                        if k != 'r_axis':
                            i = float(i)
                        pose[k] = i            
                print(pose)
                V(obj_pcd, 'obj')
                happy = input('enter y if satisfied')=='y'
            obj_tf = np.matmul(trans_tf, rot_tf)
        else:
            obj_tf = panda_robot.world.refine_subgoal(obj_tf, obj_pcd)
            obj_pcd = np.matmul(obj_tf, obj_pcd.T).T[:,:3]
            V(obj_pcd, 'obj')

        end_pcd = np.array([np.logical_not(obj_mask)]).T*init_pcd
        end_pcd += obj_pcd
        n = 10
        pred_grasps, downsample = panda_robot.infer([init_pcd,end_pcd], obj_mask, threshold=0)

        occnet_mask = panda_robot.lcc.filter_grasps(pred_grasps.cpu().numpy(), obj_tf, end_pcd)
        pred_grasps = pred_grasps[occnet_mask]
        
        # if pred_grasps.shape[0] > n:
        #     idx = np.argpartition(pred_success, -n)[-n:]
        #     pred_grasps = np.flip(pred_grasps[idx], axis=0)
        #     pred_success = np.flip(pred_success[idx])

        plan = None
        V(end_pcd, 'end')
        for i, grasp in enumerate(pred_grasps):
            #V(grasp, f'gripper_{i}', gripper=True)

            end_grasp = np.matmul(obj_tf, grasp)
            # end_grasp = np.matmul(trans_tf, end_grasp)

            z_off = np.eye(4)
            z_off[2,3] = 0.00
            z_off = np.matmul(z_off, np.linalg.inv(grasp))
            z_off = np.matmul(grasp, z_off)
            grasp = np.matmul(z_off, grasp)
            z_off = np.eye(4)
            z_off[2,3] = 0.00
            z_off = np.matmul(z_off, np.linalg.inv(end_grasp))
            z_off = np.matmul(end_grasp, z_off)
            end_grasp = np.matmul(z_off, end_grasp)

            V(grasp, f'gripper', gripper=True)
            V(end_grasp, 'end_gripper', gripper=True)
            try_execute = input("enter y to plan, else skip: ")
            if try_execute == 'y':
                current_joints = panda_robot.panda.joint_angles()
                plan = panda_robot.motion_plan(grasp, end_grasp, list(current_joints.values()))
                if plan is None:
                    print('trying flipped grasp')
                    grasp = panda_robot.rotate_grasp(grasp. np.pi)
                    end_grasp = panda_robot.rotate_grasp(end_grasp, np.pi)
                    plan = panda_robot.motion_plan(grasp, end_grasp, list(current_joints.values()))
                if plan is not None:
                    for jpos in plan[0]:
                        panda_robot.ik_helper.set_jpos(jpos)
                        time.sleep(0.1)
                    time.sleep(0.5)
                    for jpos in plan[1]:
                        panda_robot.ik_helper.set_jpos(jpos)
                        time.sleep(0.1)

                    execute = input("enter y to run execute, else skip: ")
                    if execute == 'y':
                        success = panda_robot.execute(plan)
            else:
                print('skipping')
            from IPython import embed; embed()
        
        done = True
        break
