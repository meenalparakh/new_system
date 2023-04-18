# from data_collection.utils.data_post_processing_utils import get_unseen_object_dicts
import numpy as np
import os, json, cv2, random

# from data_collection.utils.data_post_processing_utils import dataset_train_wrapper, dataset_val_wrapper, dataset_test_wrapper
from data_collection.scene_wrapper import Scene
# from detectron2.data import MetadataCatalog, DatasetCatalog
import shutil
import pickle
SPLITS = {
    'train': [0.8, 0.12, 0.08],
    'val': [0.4, 0.3, 0.3],
    'test': [0.0, 0.4, 0.6]
}

# def register_dataset(dataset_dir):

#     dataset = SceneDataset(dataset_dir)
#     scene_name_lst = list(dataset.summary['scenes'].keys())   # all the scenes in the dataset

#     def get_train_data():
#         return dataset.train_datatset_scenes(scene_name_lst, 
#                         200, 200, 200, 200, append_projected=True)

#     def get_val_data():
#         return dataset.val_datatset_scenes(scene_name_lst)

#     def get_test_data():
#         return dataset.test_datatset_scenes(scene_name_lst)

#     DatasetCatalog.register("train_dataset", get_train_data)
#     MetadataCatalog.get("train_dataset").set(thing_classes=[""])

#     DatasetCatalog.register("val_dataset", get_val_data)
#     MetadataCatalog.get("val_dataset").set(thing_classes=[""])

#     DatasetCatalog.register("test_dataset", get_test_data)
#     MetadataCatalog.get("test_dataset").set(thing_classes=[""])


class SceneDataset:
    def __init__(self, data_dir):
        self.data_dir = os.path.abspath(data_dir)

        summary_file = os.path.join(self.data_dir, 'dataset_summary.json')
        self.summary_fname = summary_file

        if os.path.exists(summary_file):
            with open(summary_file, 'r') as f:
                self.summary = json.load(f)

        else:
            self.summary = {
                'scenes':{},  # contains scene names
                'data_dir': self.data_dir
            }
            self.save_summary()

    def save_summary(self):
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        with open(self.summary_fname, 'w') as f:
            json.dump(self.summary, f)

    def clear_dir(self):
        if os.path.exists(self.data_dir):
            shutil.rmtree(self.data_dir)

    def delete_scene_dir(self, scene_name):
        scene_location = self.summary['scenes'].pop(scene_name, None)
        if scene_location is not None:
            if os.path.exists(scene_location):
                shutil.rmtree(scene_location)
        print(f'Scene {scene_name} has been removed from dataset and the corresponding dir removed')
        self.save_summary()

    def save_labeled_data_from_pcd(self, scene, n_close=0, n_far=0, 
                n_random_near=0, n_random_far=0):

        data_dir = os.path.join(self.data_dir, scene.scene_name)
        pcd, clustered_pcd, diff_objects_info, object_pcds = \
                    scene.get_pcd_cluster_info(get_object_pcds=True)

        scene.get_labeled_data_fn(pcd, clustered_pcd, diff_objects_info, object_pcds,
                                data_dir, 'close', n_close)

        scene.get_labeled_data_fn(pcd, clustered_pcd, diff_objects_info, object_pcds,
                                data_dir, 'far', n_far)

        scene.get_labeled_data_fn(pcd, clustered_pcd, diff_objects_info, object_pcds,
                                data_dir, 'random-near', n_random_near)

        scene.get_labeled_data_fn(pcd, clustered_pcd, diff_objects_info, object_pcds,
                                data_dir, 'random-far', n_random_far)
        return data_dir

    # def get_


    def run_scene_pipeline(self, scene_name, n_close=200, n_far=200, n_random_near=100, n_random_far=300, 
                val_split=0.1, test_split=0.1, skip_data_collection=False, skip_extracting_data=False,
                on_click=False): # example

        scene_location = self.summary['scenes'][scene_name]
        scene = Scene(scene_location, scene_name)

        if not skip_data_collection:
            scene.collect_data(n_close, data_type='close', write_=True, on_click=on_click)
            scene.collect_data(n_far, data_type='far', write_=True, on_click=on_click)
        if not skip_extracting_data:
            scene.run_colmap()
            scene.merge_pcd()
            scene.clean_pcd()
            scene.cluster_pcd()
            scene_data_dir = self.save_labeled_data_from_pcd(scene,
                        n_close=1, n_far=1, n_random_near=n_near, n_random_far=n_far)
            scene.train_val_test_split(scene_data_dir, [1-val_split-test_split, val_split, test_split])

        # self.save_summary()

    def register_scene(self, scene_location, scene_name, write_=True):
        self.summary['scenes'][scene_name] = scene_location
        if write_:
            self.save_summary()

    def collect_scenes(self, scene_name_lst, scenes_dir,
             n_close=200, n_far=200, on_click=False):
        
        for scene_name in scene_name_lst:
            scene_location = os.path.join(scenes_dir, scene_name)
            print('Collecting data for scene', scene_name, 'storing at', scene_location)
            self.register_scene(scene_location, scene_name, write_=True)
            self.run_scene_pipeline(scene_name, n_close=n_close, n_far=n_far, 
                    skip_extracting_data=True, on_click=on_click)


    def save_labels_from_collected_scenes(self, scene_name_lst, 
                        n_random_near=100, n_random_far=200):
        for scene_name in scene_name_lst:
            self.run_scene_pipeline(scene_name, n_random_near=n_random_near,
                        n_random_far=n_random_far, val_split=0.1, 
                        test_split=0.1, skip_data_collection=True)

    def get_table_only_scenes(self):
        scene_name_lst = self.summary['scenes']
        table_only = []
        for scene_name in scene_name_lst:
            scene_dir = self.summary['scenes'][scene_name]
            scene = Scene(scene_dir, scene_name)
            if scene.summary['table_only']:
                table_only.append(scene_name)

        return table_only


    def split_scenes(self, scene_lst=None, split_ratio=[0.90, 0.05, 0.05]):
        print('Splitting scenes')
        if scene_lst is None:
            scene_lst = list(self.summary['scenes'].keys())

        table_only = self.get_table_only_scenes()
        print(f"table only scenes: {table_only}")

        scene_lst = list(set(scene_lst).difference(set(table_only)))

        num_scenes = len(scene_lst)
        num_val = int(split_ratio[1]*num_scenes)
        num_test = int(split_ratio[2]*num_scenes)
        num_train = num_scenes - num_val - num_test

        random.shuffle(scene_lst)
        scene_type_dict = {
                'train': scene_lst[:num_train] + table_only,
                'val': scene_lst[num_train:num_train+num_val],
                'test': scene_lst[num_train+num_val:],
        }

        fname = os.path.join(self.summary['data_dir'], 'scene_train_type_info.pkl')

        with open(fname, 'wb') as f:
            pickle.dump(scene_type_dict, f)

        self.summary['scene_division'] = fname
        self.save_summary()
        print('Splitting scenes finished')

    def save_labels_from_clustered_pcd(self, scene_name_lst, 
                        n_random_near=0, n_random_far=0, splits=SPLITS):

        with open(self.summary['scene_division'], 'rb') as f:
            scene_division = pickle.load(f)

        scene_type = {}
        for k in scene_division:
            for scene_name in scene_division[k]:
                if k == 'val':
                    print(f'{k} scene: {scene_name}')
                scene_type[scene_name] = k


        for scene_name in scene_name_lst:

            scene_location = self.summary['scenes'][scene_name]
            scene = Scene(scene_location, scene_name)

            scene_data_dir = self.save_labeled_data_from_pcd(scene,
                        n_close=1, n_far=1, n_random_near=n_random_near, 
                        n_random_far=n_random_far)

            print(scene_name, ":", scene_type[scene_name])
            scene_split_ratio = splits[scene_type[scene_name]]
            scene.train_val_test_split(scene_data_dir, scene_split_ratio)

        # self.save_summary()

            # self.run_scene_pipeline(scene_name, n_random_near=n_random_near,
            #             n_random_far=n_random_far, val_split=0.1, 
            #             test_split=0.1, skip_data_collection=True)

    def train_datatset_scenes(self, scene_name_lst, _all=True, num_A=1000, num_B=1000, 
                    num_C=0, num_D=0, scene_type=["table_only", "real_objects"], append_projected=False):

        train_lst = []
        for scene_name in scene_name_lst:
            scene_dir = self.summary['scenes'][scene_name]

            scene = Scene(scene_dir, scene_name)
            if scene.summary['comment'] != '':
                continue
            cond1 = scene.summary["table_only"] and ("table_only" in scene_type)
            cond2 = (not scene.summary["table_only"]) and ("real_objects" in scene_type)

            if cond2:
                image_dir = os.path.join(scene_dir, 'images')
                data_dir = os.path.join(self.data_dir, scene_name)

                lst = dataset_train_wrapper(data_dir, image_dir, _all=_all, 
                                num_A=num_A, num_B=num_B, num_C=num_C, num_D=num_D,
                            append_projected=append_projected)
                train_lst.extend(lst)
            if cond1:
                for config_name in scene.summary['config_lst']:
                    image_dir = None
                    data_dir = os.path.join(self.data_dir, config_name)
                    lst = dataset_train_wrapper(data_dir, None, _all=_all, 
                                        num_A=num_A, num_B=num_B, num_C=num_C, num_D=num_D,
                                    append_projected=append_projected, skip_renaming=True)
                    print("table_only__________________", len(lst))
                    train_lst.extend(lst)

        return train_lst

    def val_datatset_scenes(self, scene_name_lst, image_type='far'):

        val_lst = []
        for scene_name in scene_name_lst:
            scene_dir = self.summary['scenes'][scene_name]
            image_dir = os.path.join(scene_dir, 'images')
            data_dir = os.path.join(self.data_dir, scene_name)

            scene = Scene(scene_dir, scene_name)
            if scene.summary['comment'] != '':
                continue

            if scene.summary["table_only"]:
                continue

            lst = dataset_val_wrapper(data_dir, image_dir, image_type=image_type)
            val_lst.extend(lst)

        return val_lst

    def test_datatset_scenes(self, scene_name_lst, image_type='far'):

        test_lst = []
        for scene_name in scene_name_lst:
            scene_dir = self.summary['scenes'][scene_name]
            image_dir = os.path.join(scene_dir, 'images')
            data_dir = os.path.join(self.data_dir, scene_name)

            scene = Scene(scene_dir, scene_name)
            if scene.summary['comment'] != '':
                continue

            if scene.summary["table_only"]:
                continue

            lst = dataset_test_wrapper(data_dir, image_dir, image_type=image_type)
            test_lst.extend(lst)

        return test_lst
        
    
