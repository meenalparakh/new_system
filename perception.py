##############################################################################
## Before running the file make sure detectron and detic are installed
##############################################################################

import cv2 
import numpy as np


def get_object_bbs_labels(images):
    '''
    if there are N images
    returned is N instances, each instance class 
    containing a list of bbs and a corresponding list of labels
    
    the bbs (and the labels) are filtered as long as the bb covers 
    non zero region in the pcd projection.

    For the filtered bbs (and labels), find the crop embedding using clip
    '''
    
    pass


def get_segmentation_mask(image, bb):
    
    '''
    for each bb in the image, get the segmentation mask, using SAM
    
    returns a dictionary, that contains for each segment_id in the mask, 
    the corresponding label and the embedding

    this function will be used for each image to obtain a list of segs
    and image_embeddings (see robot_env) class
    '''

    pass

def find_object(text_prompt, avg_obj_embeddings):
    
    '''
    returns the object index in the obj_embeddings that best match the 
    text_prompt
    '''

