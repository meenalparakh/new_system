# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
# import warnings
import pickle
import cv2
import tqdm
import sys
import mss

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

sys.path.insert(0, 'third_party/CenterNet2/')
from centernet.config import add_centernet_config
from detic.config import add_detic_config

from detic.predictor import VisualizationDemo

# Fake a video capture object OpenCV style - half width, half height of first screen using MSS
class ScreenGrab:
    def __init__(self):
        self.sct = mss.mss()
        m0 = self.sct.monitors[0]
        self.monitor = {'top': 0, 'left': 0, 'width': m0['width'] / 2, 'height': m0['height'] / 2}

    def read(self):
        img =  np.array(self.sct.grab(self.monitor))
        nf = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return (True, nf)

    def isOpened(self):
        return True
    def release(self):
        return True

class Args:
    def __init__(self, tmp_args):
        scene_dir = tmp_args.scene_dir
        self.num_views = tmp_args.num_views
        self.config_file = "configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml"
        print(scene_dir)

        if tmp_args.data_lst_fname == "None":
            data_lst_fname = os.path.join(scene_dir, f'data_lst_{tmp_args.num_views}.pkl')
        else:
            data_lst_fname = os.path.join(scene_dir, tmp_args.data_lst_fname) 
        with open(data_lst_fname, 'rb') as f:
            self.data_lst = pickle.load(f)
        self.input = [os.path.join(scene_dir, 'images', d["file_name"]) for d in self.data_lst]
        self.output = os.path.join(scene_dir, "detic_results")
        os.makedirs(self.output, exist_ok=True)
        self.cpu = True
        self.confidence_threshold = 0.5
        self.custom_vocabulary = ''
        self.vocabulary = 'lvis'
        self.pred_all_class = True
        self.opts = ['MODEL.WEIGHTS', 'models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth']

# constants
WINDOW_NAME = "Detic"

def setup_cfg(args):
    cfg = get_cfg()
    if args.cpu:
        cfg.MODEL.DEVICE="cpu"
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand' # load later
    if not args.pred_all_class:
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", help="Take inputs from webcam.")
    parser.add_argument("--cpu", action='store_true', help="Use CPU only.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--vocabulary",
        default="lvis",
        choices=['lvis', 'openimages', 'objects365', 'coco', 'custom'],
        help="",
    )
    parser.add_argument(
        "--custom_vocabulary",
        default="",
        help="",
    )
    parser.add_argument("--pred_all_class", action='store_true')
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False


def main_wrapper(args):
    mp.set_start_method("spawn", force=True)
    # setup_logger(name="fvcore")
    # logger = setup_logger()
    # logger.info("Arguments: " + str(args))
    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg, args)

    if args.input:
        pred_lst = []
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            print(path)
            img = read_image(path, format="BGR")
            # start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img)
            # logger.info(
            #     "{}: {} in {:.2f}s".format(
            #         path,
            #         "detected {} instances".format(len(predictions["instances"]))
            #         if "instances" in predictions
            #         else "finished",
            #         time.time() - start_time,
            #     )
            # )
            pred_lst.append(predictions)

            print("output file:", args.output)
            if args.output:
                os.makedirs(args.output, exist_ok=True)
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                visualized_output.save(out_filename)

        names = demo.metadata.get("thing_classes", None)
        with open(os.path.join(args.output, "predictions_summary.pkl"), 'wb') as f:
            pickle.dump({"predictions": pred_lst, "names": names}, f)
        return pred_lst

# from detic.modeling.text.text_encoder import build_text_encoder

# def get_clip_embeddings(vocabulary, prompt='a '):
#     text_encoder = build_text_encoder(pretrain=True)
#     text_encoder.eval()
#     texts = [prompt + x for x in vocabulary]
#     emb = text_encoder(texts).detach().contiguous().cpu().numpy()
#     return emb

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--scene-dir",
        help="scene_dir",
    )
    parser.add_argument(
        "--num-views",
        help="number of images",
        default=10
    )
    parser.add_argument(
        "--data-lst-fname",
        help="scene_dir",
        default="None"
    )

    tmp_args = parser.parse_args()
    args = Args(tmp_args)
    main_wrapper(args)

    # mp.set_start_method("spawn", force=True)
    # args = get_parser().parse_args()
    # setup_logger(name="fvcore")
    # logger = setup_logger()
    # logger.info("Arguments: " + str(args))

    # cfg = setup_cfg(args)

    # demo = VisualizationDemo(cfg, args)

    # if args.input:
    #     if len(args.input) == 1:
    #         args.input = glob.glob(os.path.expanduser(args.input[0]))
    #         assert args.input, "The input path(s) was not found"
    #     for path in tqdm.tqdm(args.input, disable=not args.output):
    #         img = read_image(path, format="BGR")
    #         start_time = time.time()
    #         predictions, visualized_output = demo.run_on_image(img)
    #         logger.info(
    #             "{}: {} in {:.2f}s".format(
    #                 path,
    #                 "detected {} instances".format(len(predictions["instances"]))
    #                 if "instances" in predictions
    #                 else "finished",
    #                 time.time() - start_time,
    #             )
    #         )

    #         if args.output:
    #             if os.path.isdir(args.output):
    #                 assert os.path.isdir(args.output), args.output
    #                 out_filename = os.path.join(args.output, os.path.basename(path))
    #             else:
    #                 assert len(args.input) == 1, "Please specify a directory with args.output"
    #                 out_filename = args.output
    #             visualized_output.save(out_filename)