from __future__ import print_function

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
# import sys
# import matplotlib.pyplot as plt
# from sklearn.cluster import DBSCAN, KMeans
import open3d
import os
# from detectron2.structures import BoxMode
import pickle
from copy import deepcopy
import random

############################################################################
##                 data post processing                                   ##
############################################################################

def get_bounding_box(binary_mask, margin=2):
    row_lims = binary_mask.any(axis=1)
    col_lims = binary_mask.any(axis=0)

    height, width = binary_mask.shape
    row_min = 0
    for idx in range(len(row_lims)):
        if row_lims[idx]:
            row_min = idx
            break

    row_max = 0
    for idx in range(len(row_lims)-1, 0, -1):
        if row_lims[idx]:
            row_max = idx
            break

    col_min = 0
    for idx in range(len(col_lims)):
        if col_lims[idx]:
            col_min = idx
            break

    col_max = 0
    for idx in range(len(col_lims)-1, 0, -1):
        if col_lims[idx]:
            col_max = idx
            break

    if ((row_max - row_min) < 2) or ((col_max - col_min) < 2):
        return None

    row_min = max(row_min - margin, 0)
    row_max = min(row_max + margin, height-1)
    col_min = max(col_min - margin, 0)
    col_max = min(col_max + margin, width-1)

    return (row_min, col_min, row_max, col_max)

def get_segmentation_data(mask, diff_colors, amodal_masks, 
            object_centers, save_dir=None, max_num_pixels=1000):
    segments_info = []
    mask_int = np.zeros(mask.shape[:2])
    colors = np.uint8(255*diff_colors[:,:3])

    area = []
    for i in range(len(colors)):
        color = colors[i]

        binary_mask = (np.abs(mask - color).sum(axis=2) < 4)
        mask_int[binary_mask] = (i+1)

        bb = get_bounding_box(binary_mask)
        
        ### if binary mask is small ignore the object

        print(f'Sum of binary mask: {binary_mask.sum()}')
        if bb is None or (binary_mask.sum() < max_num_pixels):
            continue
        binary_mask = (binary_mask*255).astype(np.uint8)
        binary_mask = (cv2.medianBlur(binary_mask, 3))/255

        # amodal_mask = amodal_masks[i]
        # amodal_mask = (amodal_mask[:, :, 0] > 250)
        # bb_amodal = get_bounding_box(amodal_mask)
        
        # amodal_mask = (amodal_mask*255).astype(np.uint8)
        # amodal_mask = (cv2.medianBlur(amodal_mask, 3))/255
        amodal_mask = None
        bb_amodal = None

        area.append([binary_mask.sum()])
        segments_info.append((i, bb, binary_mask, bb_amodal, amodal_mask, object_centers[i]))

    print(area)
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for i, _, binary_mask, _, amodal_mask, _ in segments_info:
            cv2.imwrite(os.path.join(save_dir, f'{i}.png'), binary_mask*255)
            # cv2.imwrite(os.path.join(save_dir, f'amodal_{i}.png'), amodal_mask*255)

    return segments_info, mask_int 

def get_dict(id, projected_images_dir, 
                masks_dir,
                colormap_fname, 
                height, width, cam_extr,
                segments_info, 
                segments_save_dir,
                object_category,
                flag):
    from detectron2.structures import BoxMode
    import pycocotools


    d = {
        'file_name': os.path.join(projected_images_dir, colormap_fname),
        'mask_filename': os.path.join(masks_dir, 'labels_' + colormap_fname),
        'image_id': id,
        'width': width,
        'height': height,
        'cam_extr': cam_extr,
        'binary_masks_dir': segments_save_dir,
        'annotations': [],
        'flag': flag
    }

    # print(flag)
    if flag == 'A' or flag == 'B':
        d['projected_image'] = d['file_name']
        d['file_name'] = colormap_fname
        # print('inside get dict', d['file_name'])

    for segment in segments_info:
        i, bb, _, bb_amodal, _, wf_object_center = segment
        # if bb is None:
        #     continue
        r1, c1, r2, c2 = bb
        # r1_, c1_, r2_, c2_ = bb_amodal

        binary_mask_fname = os.path.join(d['binary_masks_dir'], f'{i}.png')
        bitmap = (cv2.imread(binary_mask_fname, cv2.IMREAD_GRAYSCALE)/255).astype(np.uint8)
        bitmap = pycocotools.mask.encode(np.asarray(bitmap, order="F"))

        segment_dict = {
            # 'bbox_visible': (c1_, r1_, c2_, r2_),
            'bbox': (c1, r1, c2, r2),
            'bbox_mode': BoxMode.XYXY_ABS,
            'wf_center': wf_object_center,
            'idx': i,
            # 'segmentation': pycocotools.mask.encode(np.asarray(binary_mask, order="F")),
            'segmentation': bitmap, # i,
            'category_id': 0,
            'object_category': object_category
        }

        d['annotations'].append(segment_dict)

    return d

    # for i in range(len(data_lst)):
    #     data_pt = data_lst[i]
    #     segments_dir = data_pt['binary_masks_dir']
    #     for j in range(len(data_pt['annotations'])):
    #         anno = data_pt['annotations'][j]
    #         label_idx = anno['idx']

    #         binary_mask_fname = os.path.join(segments_dir, f'{label_idx}.png')
    #         bitmap = (cv2.imread(binary_mask_fname, cv2.IMREAD_GRAYSCALE)/255).astype(np.uint8)
    #         bitmap = pycocotools.mask.encode(np.asarray(bitmap, order="F"))
    #         data_lst[i]['annotations'][j]['segmentation'] = bitmap

    # return data_lst



# def get_unseen_object_dicts(summary_fname):

#     '''
#     all dicts would have a field that decide which type of image it is.
#     projected:         include pcd images as well
#                         all images would have a field referring to the pairs other image location

#     append_projected:  the projected images ould be included as separate datapoints

#     '''

#     if not os.path.exists(summary_fname):
#         print(f'Warning, {summary_fname} does not exist, returning empty array')
#         return []

#     with open(summary_fname, 'rb') as f:
#         data_lst = pickle.load(f)

#     # for i in range(len(data_lst)):
#     #     data_pt = data_lst[i]
#     #     segments_dir = data_pt['binary_masks_dir']
#     #     for j in range(len(data_pt['annotations'])):
#     #         anno = data_pt['annotations'][j]
#     #         label_idx = anno['idx']

#     #         binary_mask_fname = os.path.join(segments_dir, f'{label_idx}.png')
#     #         bitmap = (cv2.imread(binary_mask_fname, cv2.IMREAD_GRAYSCALE)/255).astype(np.uint8)
#     #         bitmap = pycocotools.mask.encode(np.asarray(bitmap, order="F"))
#     #         data_lst[i]['annotations'][j]['segmentation'] = bitmap

#     return data_lst

# def dataset_train_wrapper(data_dir, image_dir, num_A=1000000, num_B=1000000, 
#                             num_C=0, num_D=0, _all=True, skip_renaming=False,
#                             append_projected=False):

def dataset_wrapper(data_dir, read_files=["A", "C"]):
    cum_lst = []
    fnames = [os.path.join(data_dir, f"summary_{a}.pkl") for a in read_files]
    for f in fnames:
        if not os.path.exists(f):
            print(f, "does not exist. Not loading")
            continue
        with open(f, 'rb') as file:
            lst = pickle.load(file)
            cum_lst.extend(lst)
    return cum_lst


    # with open(os.path.join(data_dir, "summary_B.pkl"), 'rb') as f:
    #     lst_B = pickle.load(f)
    # with open(os.path.join(data_dir, "summary_C.pkl"), 'rb') as f:
    #     lst_C = pickle.load(f)
    # with open(os.path.join(data_dir, "summary_D.pkl"), 'rb') as f:
    #     lst_D = pickle.load(f)
      

    # # if not all:
    # num_A = min(num_A, len(lst_A))
    # num_B = min(num_B, len(lst_B))
    # num_C = min(num_C, len(lst_C))
    # num_D = min(num_D, len(lst_D))

    # cum_lst.extend(lst_C[:num_C])
    # cum_lst.extend(lst_D[:num_D])

    # if not skip_renaming:
    #     for data_pt in lst_A[:num_A]:
    #         data_pt['file_name'] = os.path.join(image_dir, data_pt['file_name'])
    #     # cum_lst.extend(lst_A[:num_A])

    #     for data_pt in lst_B[:num_B]:
    #         data_pt['file_name'] = os.path.join(image_dir, data_pt['file_name'])
    #     # cum_lst.extend(lst_B[:num_B])

    # cum_lst.extend(lst_A[:num_A])
    # cum_lst.extend(lst_B[:num_B])
    # print(f'Num images in {data_dir}: A ({num_A}),'
    #             f' B ({num_B}), C({num_C}), D({num_D})')

    # # input()

    # if append_projected:
        
    #     print('Appending point cloud projection data')
    #     lst_Ap = deepcopy(lst_A[:num_A])
    #     for data_pt in lst_Ap:
    #         data_pt['file_name'] = data_pt['projected_image']
    #     cum_lst.extend(lst_Ap)

    #     lst_Bp = deepcopy(lst_B[:num_B])
    #     for data_pt in lst_Bp:
    #         data_pt['file_name'] = data_pt['projected_image']
    #     cum_lst.extend(lst_Bp)

    # return cum_lst

def dataset_val_wrapper(data_dir, image_dir, image_type='both'):
    '''
    image_type can be 'far', 'close', 'both'
    '''
    if image_type=='both':
        lst = ['summary_A_val.pkl', 'summary_B_val.pkl']
    if image_type=='far':
        lst = ['summary_B_val.pkl']
    if image_type=='close':
        lst = ['summary_A_val.pkl']    

    cum_lst = []
    for fname in lst:
        lst = get_unseen_object_dicts(os.path.join(data_dir, fname))
        for data_pt in lst:
            data_pt['file_name'] = os.path.join(image_dir, data_pt['file_name'])
        cum_lst.extend(lst)

    print(image_dir, len(cum_lst))
    return cum_lst

def dataset_test_wrapper(data_dir, image_dir, image_type='both'):
    # print(data_dir, image_type)
    if image_type=='both':
        lst = ['summary_A_test.pkl', 'summary_B_test.pkl']
    if image_type=='far':
        lst = ['summary_B_test.pkl']
    if image_type=='close':
        lst = ['summary_A_test.pkl']    

    cum_lst = []
    for fname in lst:
        lst = get_unseen_object_dicts(os.path.join(data_dir, fname))
        for data_pt in lst:
            data_pt['file_name'] = os.path.join(image_dir, data_pt['file_name'])
        cum_lst.extend(lst)

    print(image_dir, len(cum_lst))
    return cum_lst


# def remove_test_images(current_dir, prefix="test"):
#     image_path = os.path.join(current_dir, 'images')
#     config_path = os.path.join(current_dir, 'configs')

#     image_fnames = glob.glob(image_path + f'/{prefix}_*.png')
#     config_fnames = glob.glob(config_path + f'/{prefix}_*.pkl')
#     for fname in [*image_fames, *config_fnames]:
#         os.remove(fname)

#     print(f'{prefix} files removed')

#############################################################################



def divide_test_data(data_dir, split_ratio=[0.8, 0.1, 0.1]):

    def divide_summary_files(summary, b1_file, b2_file, b3_file):
        with open(summary, 'rb') as f:
            data_lst = pickle.load(f)

        num_data_pts = len(data_lst)
        num_act_test = int(split_ratio[-1]*num_data_pts)
        num_val = int(split_ratio[1]*num_data_pts)
        num_train = num_data_pts - num_val - num_act_test

        random.shuffle(data_lst)

        train = data_lst[:num_train]
        val = data_lst[num_train: num_train+num_val]
        test = data_lst[num_train+num_val:]
        print(len(train), len(val), len(test))
        
        with open(b1_file, 'wb') as f:
            pickle.dump(train, f)

        with open(b2_file, 'wb') as f:
            pickle.dump(val, f)

        with open(b3_file, 'wb') as f:
            pickle.dump(test, f)

    print(split_ratio, "inside divide test data")

    summary = os.path.join(data_dir, 'summary_B.pkl')   
    if os.path.exists(summary):  
        train_file = os.path.join(data_dir, 'summary_B_train.pkl')
        val_file = os.path.join(data_dir, 'summary_B_val.pkl')
        test_file = os.path.join(data_dir, 'summary_B_test.pkl')
        divide_summary_files(summary, train_file, val_file, test_file)

    summary = os.path.join(data_dir, 'summary_A.pkl')  
    if os.path.exists(summary):  
        train_file = os.path.join(data_dir, 'summary_A_train.pkl')
        val_file = os.path.join(data_dir, 'summary_A_val.pkl')
        test_file = os.path.join(data_dir, 'summary_A_test.pkl')
        divide_summary_files(summary, train_file, val_file, test_file)


# def separate_validation_data(colmap_dir, ratio):



# def copy_test_data(test_dir, destination_dir, prefix='test', ext='.png'):
#     file_lst = []
#     for fname in os.listdir(test_dir):
#         if fname.endswith(ext):
#             new_path = os.path.join(destination_dir, prefix + '_' + fname) 
#             shutil.copy(os.path.join(test_dir, fname), new_path)
#             file_lst.append(new_path)
    
#     print(f'Test data ({ext}) copied from {test_dir} to {destination_dir}: {len(file_lst)} files.')
#     return file_lst


# file structure
# labeled_data
#   - train projected images (and associated actual image path)
#   - "test" projected images (and associated actual image path)
#   - take out some from above "test" projected images to form the validation folder
#   - random image projections (close)
#   - random image projections (far)
#   - each of the above would be a separate folder

# then register each of train, validation and test dataset

# let us name different categories of images

# type: 'A'
# close images - registered and merged pcd 
# projection of close images 

# type: 'B
# far images - registered but not merged 
# projection of far images 

# type: 'C'
# random images close

# type: 'D'
# random images far

# type: 'E'
# test images
