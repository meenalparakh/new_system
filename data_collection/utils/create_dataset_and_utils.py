import sys
sys.path.append('../')
import torch, detectron2
import shutil
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from data_collection.scene_wrapper import Scene
import pycocotools

import numpy as np
import os, json, cv2, random

from detectron2 import structures
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import ColorMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from train_methods.eval_trainer import MyTrainer
from detectron2.data import detection_utils as utils
from data_collection.dataset_wrapper import SceneDataset
from data_collection.utils.data_post_processing_utils import dataset_train_wrapper, dataset_val_wrapper, dataset_test_wrapper
from copy import deepcopy
import pickle
# from utils.modify_coco import load_coco_json


DATASET_DIR = ['/workspace/segmentation/supporting_data/data']
DATASET_DICTS = '/workspace/segmentation/supporting_data/dataset_pickle_dicts/'
DATASET_CHECK_DIR = '/workspace/segmentation/supporting_data/dataset_check/'

# TEST_ACTUAL_LABELS = '/workspace/segmentation/supporting_data/test_labels_original'
TEST_IMAGES_DIR = '/workspace/segmentation/supporting_data/labeled_test_images'
APP_LABELS_DIR = '/workspace/segmentation/supporting_data/app_labels'


if not os.path.exists(DATASET_DICTS):
    os.makedirs(DATASET_DICTS)


def check_dataset_inlined(dataloader, check_images_dir, name, metadata, inp_format="BGR"):
    dirname = os.path.join(check_images_dir, name)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    def output(vis, fname):
        filepath = os.path.join(dirname, fname)
        print("Saving to {} ...".format(filepath))
        vis.save(filepath)

    scale = 1.0
    for batch_idx, batch in enumerate(dataloader):
        for idx, per_image in enumerate(batch):
            print("imageid:", per_image["image_id"])
            # Pytorch tensor is in (C, H, W) format
            img = per_image["image"].permute(1, 2, 0).cpu().detach().numpy()
            img = utils.convert_image_to_rgb(img, inp_format)

            visualizer = Visualizer(img, metadata=metadata, scale=scale)
            target_fields = per_image["instances"].get_fields()
            labels = [metadata.thing_classes[i] for i in target_fields["gt_classes"]]
            vis = visualizer.overlay_instances(
                labels=labels,
                boxes=target_fields.get("gt_boxes", None),
                masks=target_fields.get("gt_masks", None),
            )
            output(vis, f"{batch_idx}_{idx}" + ".jpg")
            # break

        if batch_idx == 25:
            break


def get_dict_from_app(json_file):

    with open(json_file, 'r') as f:
         app_dict = json.load(f)

    d = {
        'file_name': os.path.join(TEST_IMAGES_DIR, app_dict["item"]["name"]),
        'image_id': app_dict["item"]["source_info"]["item_id"],
        'width': app_dict["item"]["slots"][0]["width"],
        'height': app_dict["item"]["slots"][0]["height"],
        'annotations': [],
    }
    print(f"saving label for", d["file_name"])

    for segment in app_dict["annotations"]:
        bb = segment["bounding_box"]
        polygon = []
        for path in segment["polygon"]["paths"]:
            pts = []
            for pt in path:
                pts.extend([pt["x"], pt["y"]])
            polygon.append(pts)

        segment_dict = {
            'bbox': [bb["x"], bb["y"], bb["w"], bb["h"]],
            'bbox_mode': BoxMode.XYWH_ABS,
            'segmentation': polygon,
            'category_id': 0,
        }
        d['annotations'].append(segment_dict)

    return d

def save_human_labeled_test_data():
    lst = []
    for json_file in os.listdir(APP_LABELS_DIR):
        if json_file.endswith(".json"):
            d = get_dict_from_app(os.path.join(APP_LABELS_DIR, json_file))
            lst.append(d)

    with open(os.path.join(DATASET_DICTS, 'human_labeled_test.pkl'), 'wb') as f:
        pickle.dump(lst, f)

    return lst

def get_human_labeled_test_data():
    with open(os.path.join(DATASET_DICTS, "human_labeled_test.pkl"), 'rb') as f:
        lst = pickle.load(f)
    return lst


def create_train_data(image_type='both', scene_type=["table_only", "real_objects"]):     # image type can be "close", or "both"
    all_data_dirs = ['/workspace/segmentation/supporting_data/data']
    lst = []

    n_close = 500
    n_far = 500 if image_type == 'both' else 0

    for dataset_dir in all_data_dirs:
        dataset = SceneDataset(dataset_dir)
        scene_name_lst = list(dataset.summary['scenes'].keys())
        lst.extend(dataset.train_datatset_scenes(scene_name_lst, _all=False,
                        num_A=n_close, num_B=n_far, num_C=0, num_D=0,
                        scene_type=scene_type, append_projected=False))

    if len(scene_type) == 2:
        name = "tro"
    elif scene_type[0] == "table_only":
        name = "to"
    else:
        name = "ro"

    random.shuffle(lst)

    with open(os.path.join(DATASET_DICTS, f"train_{image_type}_{name}.pkl"), 'wb') as f:
        pickle.dump(lst, f)

    print(f'Length of train data {image_type} size', len(lst))
#    random.shuffle(lst)
    return lst

def get_train_data(image_type="both", scene_type=["table_only", "real_objects"]):

    if len(scene_type) == 2:
        name = "tro"
        real_objects = get_train_data("both", ["real_objects"])
        table_only = get_train_data("both", ["table_only"])
        real_objects.extend(table_only)
        return real_objects

    elif scene_type[0] == "table_only":
        name = "to"
    else:
        name = "ro"

    with open(os.path.join(DATASET_DICTS, f"train_{image_type}_{name}.pkl"), 'rb') as f:
        lst = pickle.load(f)
    return lst

def create_train_data_sized(s=25000):
    lst = get_train_data(image_type="both", scene_type=["real_objects"])
#    random.shuffle(lst)

    n = min(s, len(lst))
    random.shuffle(lst)

    with open(os.path.join(DATASET_DICTS, f"train_both_real_{n}.pkl"), 'wb') as f:
        pickle.dump(lst[:n], f)

    print('Length of train data size', len(lst[:n]))
    return lst[:n]

def get_train_data_sized(s=10000):
    with open(os.path.join(DATASET_DICTS, f"train_both_real_{s}.pkl"), 'rb') as f:
        lst = pickle.load(f)
    return lst

def create_val_data(image_type='far'):
    lst = []
    for dataset_dir in DATASET_DIR:
        dataset = SceneDataset(dataset_dir)
        scene_name_lst = list(dataset.summary['scenes'].keys())
        _lst = dataset.val_datatset_scenes(scene_name_lst,
                                    image_type=image_type)
        lst.extend(_lst)

    random.shuffle(lst)
    with open(os.path.join(DATASET_DICTS, f"val_{image_type}.pkl"), 'wb') as f:
        pickle.dump(lst, f)

    print(f'Length of val {image_type}', len(lst))
    random.shuffle(lst)
    return lst

def get_val_data(n=0, image_type='far'):
    with open(os.path.join(DATASET_DICTS, f"val_{image_type}.pkl"), 'rb') as f:
        lst = pickle.load(f)
    if n==0:
        return lst

    return lst[:n]


def create_test_data_from_scenes():
    lst = []
    for dataset_dir in DATASET_DIR:
        dataset = SceneDataset(dataset_dir)
        scene_name_lst = list(dataset.summary['scenes'].keys())


        for scene_name in scene_name_lst:
            scene_dir = dataset.summary['scenes'][scene_name]
            image_dir = os.path.join(scene_dir, 'images')
            data_dir = os.path.join(dataset.data_dir, scene_name)

            scene = Scene(scene_dir, scene_name)
            if scene.summary['comment'] != '':
                continue

            close_dicts = dataset_test_wrapper(data_dir, image_dir, image_type='close')
            far_dicts = dataset_test_wrapper(data_dir, image_dir, image_type='far')

            close_dicts = random.sample(close_dicts, k=min(5, len(close_dicts)))
            far_dicts = random.sample(far_dicts, k=min(5, len(far_dicts)))

            lst.extend(close_dicts)
            lst.extend(far_dicts)

    with open(os.path.join(DATASET_DICTS, "test_closeToTrain5.pkl"), 'wb') as f:
        pickle.dump(lst, f)

    print('Length of test data from train-related-scenes', len(lst))
    return lst

def create_test_data_from_test_scenes():
    lst = []
    dataset_dir = '/workspace/segmentation/supporting_data/test_data_final_labeled'

    dataset = SceneDataset(dataset_dir)
    scene_name_lst = list(dataset.summary['scenes'].keys())
    print("name lst", scene_name_lst)

    for scene_name in scene_name_lst:
        scene_dir = dataset.summary['scenes'][scene_name]
        image_dir = os.path.join(scene_dir, 'images')
        data_dir = os.path.join(dataset.data_dir, scene_name)

        scene = Scene(scene_dir, scene_name)
        print(scene_name, "............")
        if scene.summary['comment'] != '':
            continue
        print(scene_name, "............")


        scene.train_val_test_split(data_dir, [0.0, 0.0, 1.0])
        close_dicts = dataset_test_wrapper(data_dir, image_dir, image_type='close')
        far_dicts = dataset_test_wrapper(data_dir, image_dir, image_type='far')

        close_dicts = random.sample(close_dicts, k=min(5, len(close_dicts)))
        far_dicts = random.sample(far_dicts, k=min(5, len(far_dicts)))

        lst.extend(close_dicts)
        lst.extend(far_dicts)

    with open(os.path.join(DATASET_DICTS, "test_TestScenes5.pkl"), 'wb') as f:
        pickle.dump(lst, f)

    print('Length of test data from test scenes', len(lst))
    return lst


def create_test_data_from_unlabeled():
    lst = []
    dataset_dir = '/workspace/segmentation/supporting_data/test_data_final'
    dataset = SceneDataset(dataset_dir)
    for scene_name, scene_location in dataset.summary['scenes'].items():
        image_dir = os.path.join(scene_location, "images")
        for image_fname in os.listdir(image_dir):
            if image_fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                lst.append({'file_name': os.path.join(image_dir, image_fname)})


    old_clutters = []
    dataset_dir = '/workspace/segmentation/supporting_data/test_data'
    dataset = SceneDataset(dataset_dir)
    for scene_name, scene_location in dataset.summary['scenes'].items():
        image_dir = os.path.join(scene_location, "images")
        scene_lst = []
        for image_fname in os.listdir(image_dir):
            if image_fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                scene_lst.append({'file_name': os.path.join(image_dir, image_fname)})
        old_clutters.extend(random.sample(scene_lst, k=5))

    lst.extend(old_clutters)

    with open(os.path.join(DATASET_DICTS, "test_Unlabeled.pkl"), 'wb') as f:
        pickle.dump(lst, f)

    print('Length of test data from unlabeled', len(lst))

    return lst

def get_test_data(data_types=['closeToTrain5', 'TestScenes5', 'Unlabeled']):

    lst = []
    for suffix in data_types:
        with open(os.path.join(DATASET_DICTS, f"test_{suffix}.pkl"), 'rb') as f:
            lst.extend(pickle.load(f))

    return lst

def get_test_image_dir():
    target_dir = TEST_IMAGES_DIR
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    lst = datasets["test_labeled_human"]()
    for data_dict in lst:
        source = data_dict['file_name']
        fname = os.path.normpath(source).split(os.sep)[-1]
        dest = os.path.join(target_dir, fname)
        shutil.copyfile(source, dest)

    print(f"test images saved to {target_dir}")

# datasets = {
#     'train_full': (lambda : get_train_data('both', scene_type=["real_"])),
#     'train_close': (lambda : get_train_data('close')),
#     # 'train_8000': (lambda : get_train_data_sized(s=8000)),
#     # 'train_7000': (lambda : get_train_data_sized(s=7000)),
#     'val_far': (lambda : get_val_data(image_type='far')),
#     'val_both100': (lambda : get_val_data(n=100, image_type='both')),
#     'val_both': (lambda : get_val_data('both')),
#     'test_all': (lambda : get_test_data()),
#     'test_unlabeled': (lambda: get_test_data(['Unlabeled'])),
#     'test_labeled': (lambda: get_test_data(['closeToTrain5', 'TestScenes5'])),
#     'test_labeled_human': (lambda : get_human_labeled_test_data())
#     }

datasets = {
    'train_close_real': (lambda : get_train_data('close', scene_type=["real_objects"])),
    'train_both_real': (lambda : get_train_data('both', scene_type=["real_objects"])),
    'train_both_real_8000': (lambda : get_train_data_sized(8000)),
    'train_both_synthetic': (lambda : get_train_data('both', scene_type=["table_only"])),
    'train_both_real_synthetic': (lambda : get_train_data('both', scene_type=["table_only", "real_objects"])),
    'val_both100': (lambda : get_val_data(n=100, image_type='both')),
    'val_both': (lambda : get_val_data('both')),
    'test_labeled_human': (lambda : get_human_labeled_test_data()),
}

def register_datasets():

    DatasetCatalog.register("train_close_real", datasets['train_close_real'])
    MetadataCatalog.get("train_close_real").set(thing_classes=[""])

    DatasetCatalog.register("train_both_real", datasets['train_both_real'])
    MetadataCatalog.get("train_both_real").set(thing_classes=[""])

    DatasetCatalog.register("train_both_real_8000", datasets['train_both_real_8000'])
    MetadataCatalog.get("train_both_real_8000").set(thing_classes=[""])

    DatasetCatalog.register("train_both_synthetic", datasets['train_both_synthetic'])
    MetadataCatalog.get("train_both_synthetic").set(thing_classes=[""])

    DatasetCatalog.register("train_both_real_synthetic", datasets['train_both_real_synthetic'])
    MetadataCatalog.get("train_both_real_synthetic").set(thing_classes=[""])

    DatasetCatalog.register("val_both100", datasets['val_both100'])
    MetadataCatalog.get("val_both100").set(thing_classes=[""])

    DatasetCatalog.register("val_both", datasets['val_both'])
    MetadataCatalog.get("val_both").set(thing_classes=[""])

    DatasetCatalog.register("test_labeled_human", datasets['test_labeled_human'])
    MetadataCatalog.get("test_labeled_human").set(thing_classes=[""])


# def register_datasets():

#     DatasetCatalog.register("train_full", datasets['train_full'])
#     MetadataCatalog.get("train_full").set(thing_classes=[""])

#     DatasetCatalog.register("train_close", datasets['train_close'])
#     MetadataCatalog.get("train_close").set(thing_classes=[""])

#     DatasetCatalog.register("train_7000", datasets['train_7000'])
#     MetadataCatalog.get("train_7000").set(thing_classes=[""])

#     DatasetCatalog.register("train_8000", datasets['train_8000'])
#     MetadataCatalog.get("train_8000").set(thing_classes=[""])

#     DatasetCatalog.register("val_far", datasets['val_far'])
#     MetadataCatalog.get("val_far").set(thing_classes=[""])

#     DatasetCatalog.register("val_far100", datasets['val_far100'])
#     MetadataCatalog.get("val_far100").set(thing_classes=[""])

#     DatasetCatalog.register("val_both", datasets['val_both'])
#     MetadataCatalog.get("val_both").set(thing_classes=[""])

#     DatasetCatalog.register("test_all", datasets['test_all'])
#     MetadataCatalog.get("test_all").set(thing_classes=[""])

#     DatasetCatalog.register("test_labeled", datasets['test_labeled'])
#     MetadataCatalog.get("test_labeled").set(thing_classes=[""])

#     DatasetCatalog.register("test_unlabeled", datasets['test_unlabeled'])
#     MetadataCatalog.get("test_unlabeled").set(thing_classes=[""])

#     DatasetCatalog.register("test_labeled_human", datasets['test_labeled_human'])
#     MetadataCatalog.get("test_labeled_human").set(thing_classes=[""])



# def create_coco_new_labels():
#     image_root = "/workspace/segmentation/coco_dataset/coco/images/train2017"
#     json_file =  "/workspace/segmentation/coco_dataset/coco/annotations/instances_train2017.json"
#     lst = load_coco_json(json_file, image_root, dataset_name="coco")

#     fname = os.path.join(DATASET_DICTS, "coco_dataset.pkl")
#     with open(fname, 'wb') as f:
#         pickle.dump(lst, f)

#     return lst

# def get_coco_dataset():
#     fname = os.path.join(DATASET_DICTS, "coco_dataset.pkl")
#     with open(fname, 'rb') as f:
#         lst = pickle.load(f)
#     return lst


# def predictions_to_dataset(dataloader, predictor,
#                 threshold=0.5, tmp_save_fname=None):

#     predictions_dict_lst = []
#     count = 0
#     for batch_idx, batch in enumerate(dataloader):
#         for idx, d in enumerate(batch):

#             print('Image:', d["file_name"])
#             im = cv2.imread(d["file_name"])
#             outputs = predictor(im)

#             pred_boxes = outputs["instances"].pred_boxes.to('cpu')
#             our_mthd_boxes = d["instances"].gt_boxes.to('cpu')

#             IOUs = structures.pairwise_iou(our_mthd_boxes, pred_boxes).numpy()
#             max_ious = np.amax(IOUs, axis=0)
#             selected = (max_ious > threshold)

#             width, height = im.shape[:2]
#             image_dict = {
#                 'file_name': d["file_name"],
#                 'image_id': idx,
#                 'width': width,
#                 'height': height,
#                 'annotations': [],
#             }

#             pred_masks = outputs["instances"].pred_masks.to('cpu')

#             print("Total objects on table (out model)",
#                          our_mthd_boxes.tensor.shape[0], " Detected:",selected.sum())
#             # print("Selected (0 or 1):", selected, "total", selected.sum())
#             for object_id in range(pred_boxes.tensor.shape[0]):
#                 if selected[object_id]:
#                     segment_dict = {
#                         'bbox': pred_boxes.tensor[object_id].tolist(),
#                         'bbox_mode': BoxMode.XYXY_ABS,
#                         'segmentation':  pycocotools.mask.encode(np.asarray(
#                                                     pred_masks[object_id].numpy(),
#                                                     order="F")),
#                         'category_id': 0
#                     }
#                     image_dict["annotations"].append(segment_dict)
#             print(f"Boxes appended:", len(image_dict["annotations"]))

#             predictions_dict_lst.append(image_dict)
#             count += 1
#             print("Current count", count)

#     if tmp_save_fname is not None:
#         with open(tmp_save_fname, 'wb') as f:
#             pickle.dump(predictions_dict_lst, f)

#     return predictions_dict_lst


# def get_model_labeled_data(cfg, cfg_dict, dataloader, name=None,
#             prefix=None, revaluate=False):

#     model_name = cfg_dict['config_file']
#     if name is None:
#         name =  os.path.normpath(model_name).split(os.sep)[-1][:-5]

#     name = f"{prefix}_{name}.pkl"
#     dict_fname = os.path.join(DATASET_DICTS, name)
#     if os.path.exists(dict_fname) and (not revaluate):
#         with open(dict_fname, 'rb') as f:
#             lst = pickle.load(dict_fname)
#         return lst

#     # lst = dict_lst
#     _cfg = deepcopy(cfg)
#     _cfg.augmentate = False
#     _cfg.resize_aug = False
#     _cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_name)

#     predictor = DefaultPredictor(_cfg)
#     lst = predictions_to_dataset(dataloader, predictor, threshold=0.5,
#             tmp_save_fname=dict_fname)

#     return lst
