# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import os
import sys
import cv2
import pickle

from detectron2.config import get_cfg

sys.path.insert(0, 'third_party/CenterNet2/')
from centernet.config import add_centernet_config

from detic.config import add_detic_config
from detic.predictor import VisualizationDemo
from argparse import ArgumentParser


from collections import namedtuple
Args = namedtuple("Args", ["vocabulary", "custom_vocabulary"])
# args = Args(vocabulary="lvis", custom_vocabulary=["bowl"])
# args_custom = Args(vocabulary="custom", custom_vocabulary=["bowl", "mug", "shelf", "cardboard_box"])


def setup_cfg():
    cfg = get_cfg()
    # cfg.MODEL.DEVICE="cpu"
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file("configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml")
    cfg.merge_from_list(['MODEL.WEIGHTS', 'models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth'])
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.5
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand' # load later
    cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
    cfg.freeze()
    return cfg

def get_predictions(rgbs, _args):
    cfg = setup_cfg()
    demo = VisualizationDemo(cfg, _args)

    pred_lst = []
    output_dir = "detic_predictions"
    os.makedirs(output_dir, exist_ok=True)

    for idx, im in enumerate(rgbs):
        print(f"finding objects in {idx}")

        # converting the input image to BGR format 
        # (the image needs to be BGR, and 0-255, np.uint8)

        img = im[:, :, ::-1].astype(np.uint8)
        predictions, visualized_output = demo.run_on_image(img)
        pred_lst.append(predictions)
        visualized_output.save(os.path.join(output_dir, f"predictions_{idx}.png"))

    names = demo.metadata.get("thing_classes", None)
    return pred_lst, names

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--image-dir")
    args = parser.parse_args()
    
    rgbs = []

    with open(os.path.join(args.image_dir, "args.pkl"), 'rb') as f:
        image_fnames, vocab_args = pickle.load(f)

    rgbs = []
    for f in image_fnames:
        rgb = cv2.imread(os.path.join(args.image_dir, f))
        rgbs.append(rgb)

    preds_lst, names = get_predictions(rgbs, vocab_args)

    with open("predictions_summary.pkl", 'wb') as f:
        pickle.dump([preds_lst, names], f)
