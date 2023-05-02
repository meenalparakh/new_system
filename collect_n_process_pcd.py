import sys

sys.path.append("../")
import os
from data_collection.scene_wrapper import Scene
from data_collection.dataset_wrapper import SceneDataset
import argparse
from datetime import datetime
import signal
from data_collection.utils.data_collection_utils import get_intr_from_cam_params
import pickle
import random
import glob


def handler(signum, frame):
    msg = "Ctrl-c was pressed. Exiting."
    exit(1)


signal.signal(signal.SIGINT, handler)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-close", type=int, default=30)
    parser.add_argument("--n-far", type=int, default=0)
    parser.add_argument("--skip-collection", action="store_true", default=False)
    parser.add_argument("--data-dir", type=str, default="../data/labeled_data/")
    parser.add_argument("--scenes-dir", type=str, default="../data/scene_data")
    parser.add_argument("--scene-name", type=str, default="")
    parser.add_argument("--delete-scene-dir", action="store_true", default=False)
    parser.add_argument("--delete-configs", action="store_true", default=False)
    parser.add_argument("--skip-colmap", action="store_true", default=False)
    parser.add_argument("--skip-process", action="store_true", default=False)
    parser.add_argument("--on-click", action="store_true", default=False)
    parser.add_argument("--cluster", action="store_true", default=False)
    parser.add_argument("--data-gen", action="store_true", default=False)
    parser.add_argument("--visualize", action="store_true", default=False)
    parser.add_argument("--table-only", action="store_true", default=False)
    parser.add_argument("--create-new-configs", type=int, default=0)
    args = parser.parse_args()

    dataset = SceneDataset(args.data_dir)
    scenes_dir = args.scenes_dir

    # dataset.collect_scenes(['first', 'second'], '/workspace/segmentation/scenes', n_close=10, n_far=5)
    if args.scene_name == "":
        scene_name = datetime.now().strftime("%Y%m%d_%H%M")
    else:
        scene_name = args.scene_name

    scene_location = os.path.join(scenes_dir, scene_name)
    scene = Scene(scene_location, scene_name, table_only=args.table_only)

    print("Total scene count:", len(dataset.summary["scenes"]))
    print(f"Scene: {scene_name}")
    if args.delete_scene_dir:
        dataset.delete_scene_dir(scene_name)
        exit()

    if not args.skip_collection:
        dataset.collect_scenes(
            [scene_name],
            scenes_dir,
            n_close=args.n_close,
            n_far=args.n_far,
            on_click=args.on_click,
        )

    dataset.register_scene(scene_location, scene_name)
    # if args.set_revert_normal:

    if not args.skip_colmap:
        scene.run_colmap(skip_if_done=False)

    if not args.skip_process:
        scene.merge_and_clean_pcd(
            skip_if_done=False,
            visualize=args.visualize,
            revert_normal=scene.summary["revert_normal"],
        )

    if args.cluster:
        scene.cluster_pcd(skip_if_done=False, visualize=args.visualize)

    if args.delete_configs:
        scene.delete_configs()

    # if args.create_new_configs > 0:
    #     for i in range(args.create_new_configs):
    #         scene.create_object_pcd_with_cluster(visualize=args.visualize)

    print("Ending", scene_name)
    print("Total scenes:", len(dataset.summary["scenes"]))
    # scene_data_dir = dataset.save_labeled_data_from_pcd(scene,
    #             n_close=1, n_far=1, n_random_near=20, n_random_far=100)
    # scene_data_dir = os.path.join(dataset.data_dir, scene_name)
    # scene.train_val_test_split(scene_data_dir, [0.8, 0.1, 0.1])

    if args.data_gen:
        if scene.summary["table_only"]:
            scene.get_data_for_all_configurations(dataset.data_dir)
        else:
            scene_data_dir = dataset.save_labeled_data_from_pcd(
                scene, n_close=1, n_far=1, n_random_near=10, n_random_far=10
            )
