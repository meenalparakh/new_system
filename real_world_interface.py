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
import cv2
import copy
import random
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, _create_text_labels
from detectron2.data import MetadataCatalog, DatasetCatalog

from evaluation.aruco_dict import ARUCO_DICT, aruco_display

os.environ["PYOPENGL_PLATFORM"] = "egl"

from realsense_lcm.config.default_multi_realsense_cfg import get_default_multi_realsense_cfg
from realsense_lcm.utils.pub_sub_util import RealImageLCMSubscriber, RealCamInfoLCMSubscriber
from realsense_lcm.multi_realsense_publisher_visualizer import subscriber_visualize

from scipy.spatial.transform import Rotation as R
import argparse
from torch_geometric.nn import fps
from test_meshcat_pcd import viz_pcd as V
from simple_multicam import MultiRealsense

class WorldInterface():
    def __init__(self, viz=True):
        '''
        Interface for handling cameras, point clouds, object placements
        '''
        self._start_rs_sub()
        self._setup_detectron()

    def lc_th(self, lc):
        while True:
            lc.handle_timeout(1)
            time.sleep(0.001)

    def _setup_detectron(self):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model

        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        self.predictor = DefaultPredictor(self.cfg)
            
    def _start_rs_sub(self):
        '''
        start a thread that runs the LCM subscribers to the realsense camera(s)
        should be called once at the beginning, sets subscribers into self.img_subscribers
        '''
        lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=1")
        rs_cfg = get_default_multi_realsense_cfg()
        serials = rs_cfg.SERIAL_NUMBERS

        serials = [serials[0]] # if multiple cameras are publishing, can pick just one view to use

        rgb_topic_name_suffix = rs_cfg.RGB_LCM_TOPIC_NAME_SUFFIX
        depth_topic_name_suffix = rs_cfg.DEPTH_LCM_TOPIC_NAME_SUFFIX
        info_topic_name_suffix = rs_cfg.INFO_LCM_TOPIC_NAME_SUFFIX
        pose_topic_name_suffix = rs_cfg.POSE_LCM_TOPIC_NAME_SUFFIX

        prefix = rs_cfg.CAMERA_NAME_PREFIX
        camera_names = [f'{prefix}{i}' for i in range(len(serials))]

        # update the topic names based on each individual camera
        rgb_sub_names = [f'{cam_name}_{rgb_topic_name_suffix}' for cam_name in camera_names]
        depth_sub_names = [f'{cam_name}_{depth_topic_name_suffix}' for cam_name in camera_names]
        info_sub_names = [f'{cam_name}_{info_topic_name_suffix}' for cam_name in camera_names]
        pose_sub_names = [f'{cam_name}_{pose_topic_name_suffix}' for cam_name in camera_names]

        self.img_subscribers = []
        for i, name in enumerate(camera_names):
            img_sub = RealImageLCMSubscriber(lc, rgb_sub_names[i], depth_sub_names[i])
            info_sub = RealCamInfoLCMSubscriber(lc, pose_sub_names[i], info_sub_names[i])
            self.img_subscribers.append((name, img_sub, info_sub))

        self.cams = MultiRealsense(camera_names, cfg=None)

        lc_thread = threading.Thread(target=self.lc_th, args=(lc,))
        lc_thread.daemon = True
        lc_thread.start()

        return lc_thread

    def get_images(self):
        '''
        render rgb and depth from cameras.
        important: must call self.start_rs_sub() first (called upon init)
        '''
        rgb = []
        depth = []
        cam_poses = []
        cam_ints = []
        for (name, img_sub, info_sub) in self.img_subscribers:
            rgb_image, depth_image = img_sub.get_rgb_and_depth(block=True)
            cam_int = info_sub.get_cam_intrinsics(block=True)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            rgb.append(rgb_image)
            depth.append(depth_image)
            cam_ints.append(cam_int)
            
        return rgb, depth, cam_poses, cam_ints

    def get_pcd_ar(self, pcd_filter=True):
        rgb_list, depth_list, cam_pose_list, cam_int_list = self.get_images()
        print(rgb_list)

        pcd_pts = []
        for idx, cam in enumerate(self.cams.cams):
            #rgb, depth = img_subscribers[idx][1].get_rgb_and_depth(block=True)
            rgb, depth = rgb_list[idx], depth_list[idx]
            self.cam_intrinsics = cam_int_list[idx]

            cam.cam_int_mat = self.cam_intrinsics
            cam._init_pers_mat()
            self.cam_pose_world = cam.cam_ext_mat

            depth = depth * 0.001
            valid = depth < cam.depth_max
            valid = np.logical_and(valid, depth > cam.depth_min)
            depth_valid = copy.deepcopy(depth)
            depth_valid[np.logical_not(valid)] = 0.0 # not exactly sure what to put for invalid depth

            pcd_cam = cam.get_pcd(in_world=False, filter_depth=False, rgb_image=rgb, depth_image=depth_valid)[0]
            pcd_cam = pcd_cam[np.where(pcd_cam[:,2] > 0.1)]
            #pcd_world = util.transform_pcd(pcd_cam, self.cam_pose_world)
            pcd_world = np.matmul(self.cam_pose_world, np.hstack([pcd_cam, np.ones((pcd_cam.shape[0], 1))]).T).T[:, :-1]

            if pcd_filter:
                pcd_world = pcd_world[np.nonzero(pcd_world[:,0] > 0.2)]
            
            pcd_pts.append(pcd_world)

        pcd_full = np.concatenate(pcd_pts, axis=0)
        return pcd_full, rgb_list[0]

    def manual_segment(self, pcd):
        '''
        Returns mask of target object in scene (with some hard coded priors abt where the object is)
        '''
        x_lim = 0
        y_lim = 0
        z_lim = 0
        x_mask = np.where((pcd[:, 0] > x_lim))
        y_mask = np.where((pcd[:, 1] < y_lim))
        z_mask = np.where((pcd[:, 2] > z_lim))
        mask = np.intersect1d(x_mask, y_mask)
        mask = np.intersect1d(mask, z_mask)

        return mask

    def segment(self, im):
        '''
        Runs pre-trained model of Detectron2 on the camera image and uses user interface to specify object of interest.
        '''
        outputs = self.predictor(im)
        scores = outputs["instances"].scores
        classes = outputs["instances"].pred_classes.tolist()
        print(outputs["instances"].pred_boxes)
        masks = outputs["instances"].pred_masks
        
        # We can use `Visualizer` to draw the predictions on the image.
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=3)
        labels = _create_text_labels(classes, scores, v.metadata.get("thing_classes", None))
        print(labels)

        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imshow('image', out.get_image()[:, :, ::-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        seg_idx = input('please enter the index in the above^ list of the object you want: ')
        print('input was', seg_idx, 'which is', labels[int(seg_idx)])
        seg_idx = int(seg_idx)
        # from IPython import embed; embed()

        obj_mask = masks[seg_idx].flatten().cpu().numpy()
        return obj_mask
    
    def choose_subgoal(self, obj_pcd):
        '''
        brute-force user interface for specifying object transform
        '''
        obj_tf = np.eye(4)
        pose = {'x':0.22, 'y':-0.1, 'z':-0.1, 'r_axis':'x'}
        happy = False
        print('enter object pose')
        V(obj_pcd, 'obj')
        trans_tf = np.eye(4)
        rot_tf = np.eye(4)
        while not happy:
            trans_tf[:3, 3] = [pose['x'], pose['y'], pose['z']]
            rot_tf[:3,:3] = R.from_euler(pose['r_axis'], -np.pi/2).as_matrix()
            trial_obj_pcd = np.matmul(obj_tf, copy.deepcopy(obj_pcd).T).T[:,:3]
            V(trial_obj_pcd, 'obj')
            happy = input('are you happy? (y/n) ') == 'y'
            for k in pose.keys():
                i = input(k + ': ')
                if i != '':
                    if k != 'r_axis':
                        i = float(i)
                    pose[k] = i
            print('current pose is', pose)
            obj_tf = np.matmul(trans_tf, rot_tf)

        return obj_tf

    def refine_subgoal(self, obj_tf, obj_pcd):
        '''
        refine the translation of a subgoal specified with aruco
        '''
        pose = {'x':0., 'y':0., 'z':0}
        happy = False
        trans = np.eye(4)
        print('enter object pose')
        V(obj_pcd, 'obj')
        while not happy:
            trans[:3, 3] = [pose['x'], pose['y'], pose['z']]
            obj_tf = np.matmul(trans, obj_tf)
            trial_obj_pcd = np.matmul(obj_tf, copy.deepcopy(obj_pcd).T).T[:,:3]
            V(trial_obj_pcd, 'obj')
            happy = input('are you happy? (y/n) ') == 'y'
            for k in pose.keys():
                i = input(k + ': ')
                if i != '':
                    i = float(i)
                else:
                    i = 0.0
                pose[k] = i
            print('current pose is', obj_tf)

        return obj_tf
    
    def detect_aruco(self, image):
        '''
        initial transform estimation using aruco tags. this must be further refined for accuracy
        '''
        h,w,_ = image.shape
        width=600
        height = int(width*(h/w))        
        # image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT['DICT_4X4_50'])
        arucoParams = cv2.aruco.DetectorParameters_create()
        corners, ids, rejected = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)

        detected_markers = aruco_display(corners, ids, rejected, image)
        # cv2.imshow("Image", detected_markers)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        if ids is None:
            print('could not detect tag. try again?')
            return None

        return detected_markers, corners, ids

    def ar_subgoal(self):
        # verify that the supplied ArUCo tag exists and is supported by OpenCV
        if ARUCO_DICT.get('DICT_4X4_50', None) is None:
            print(f"ArUCo tag type  is not supported")
            sys.exit(0)

        # 1. detect initial tag pose
        result = None
        while result is None:
            val = input('please hold the AR tag at its initial pose. press enter when ready.')
            _, start_im = self.get_pcd_ar()
            print('image recorded. detecting intial tag position...')
            result = self.detect_aruco(start_im)
        detected_markers, start_corners, ids = result
        
        # 2. detect end tag pose
        result = None
        while result is None:
            val = input('please hold the AR tag at its end pose. press enter when ready.')
            _, end_im = self.get_pcd_ar()
            print('image recorded. detecting end tag position...')
            result = self.detect_aruco(end_im)
        detected_markers, end_corners, ids = result

        # 3. get poses of start and end
        poses = []
        distortion_coefficients = 0 #np.array([0, 0, 0, 0, 0]) # k1, k2, p1, p2, k3
        for corners, im in zip([start_corners, end_corners], [start_im, end_im]):
            if len(corners) > 0:
                # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
                rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[0], 0.02, self.cam_intrinsics, distortion_coefficients)
                rot_mat = R.from_rotvec(rvec[0][0]).as_matrix()

                pose_mat_cam = np.eye(4)
                pose_mat_cam[:3,:3] = rot_mat
                pose_mat_cam[:3,3] = tvec
                pose_mat_world = np.matmul(self.cam_pose_world, pose_mat_cam)
                poses.append(pose_mat_world)

                #cv2.aruco.drawAxis(im, self.cam_intrinsics, distortion_coefficients, rvec, tvec, 0.01)
                result_img = cv2.drawFrameAxes(im, self.cam_intrinsics, distortion_coefficients, rvec[0][0], tvec[0][0],0.03)
                # cv2.imshow('pose', result_img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

        # 4. get transform between start and end
        start_pose = poses[0]
        end_pose = poses[1]
        subgoal_tf = np.matmul(end_pose, np.linalg.inv(start_pose))
        # from IPython import embed; embed()

        
        return subgoal_tf
