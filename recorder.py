from colorama import Fore
from glob import glob
import numpy as np
import os
import pickle
import shutil
from datetime import datetime
from IPython import embed
import cv2
# from llm_robot.utils import path_util

class Recorder:

    def __init__(self, experiment_name):

        '''
        initialize a new recorder for each experiment,
        set the experiment's name.

        then each function in the robot will use the recorder to 
        '''

        self.EXPERIMENTS_DIR = "/Users/meenalp/Desktop/MEng/system_repos/llmrobot/experiment_results"
        # self.PLAN_RESULTS = "plan_results"

        os.makedirs(self.EXPERIMENTS_DIR, exist_ok=True)
        # os.makedirs(self.PLAN_RESULTS, exist_ok=True)

        self.experiment_dir = os.path.join(self.EXPERIMENTS_DIR, experiment_name)
        if os.path.exists(self.experiment_dir):
            response = input("the experiment dir already exists. To delete press 'yesyes', and press 'n' to conitnue replay:")
            if response == "yesyes":
                shutil.rmtree(self.experiment_dir)
                os.makedirs(self.experiment_dir)

        else:
            os.makedirs(self.experiment_dir)

            # else:

        # self.timestamp_lst = []

        # with open(os.path.join(self.experiment_dir, "timestamps.pkl"))

    # def save_images(self, dir, images):
    #     os.makedirs(dir)

    #     for im in images:


    def record_obs(self, obs, segs, object_dicts, bg_pcd, detection=True, sam=True):

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")

        '''
        todo: save them appropriately in .png files for images
        '''

        with open(os.path.join(self.experiment_dir, f"info_{timestamp}.pkl"), 'wb') as f:
            pickle.dump({"obs": obs,
                         "segs": segs,
                         "object_dicts": object_dicts, 
                         "bg_pcd": bg_pcd},
                         f)

        if detection:    
            shutil.copytree(src=os.path.join(path_util.get_detic_folder(), f"detic_predictions"),
                            dst=os.path.join(self.experiment_dir, f"detic_predictions_{timestamp}"))

        if sam:
            shutil.copytree(src=path_util.get_sam_result_folder(),
                            dst=os.path.join(self.experiment_dir, f"sam_predictions_{timestamp}"))

        self.timestamp = timestamp
        print("Observation stored")
        

    
    def add_description(self, description, new_dict):

        with open(os.path.join(self.experiment_dir, f"description_{self.timestamp}.txt"), 'w') as f:
            f.write(description)

        if os.path.exists(os.path.join(self.experiment_dir, f"info_{self.timestamp}.pkl")):
            with open(os.path.join(self.experiment_dir, f"info_{self.timestamp}.pkl"), 'rb') as f:
                info = pickle.load(f)
        else:
            info = {}

        info["object_dicts"] = new_dict

        with open(os.path.join(self.experiment_dir, f"info_{self.timestamp}.pkl"), 'wb') as f:
            pickle.dump(info, f)

        print("description added and the dict updated")

    def record_grasp(self, obj_id, grasp_world, grasp_relative):
        with open(os.path.join(self.experiment_dir, f"grasp_{obj_id}_{self.timestamp}.pkl"), 'wb') as f:
            pickle.dump([obj_id, grasp_world, grasp_relative], f)
        print("grasp stored")

    def record_place(self, obj_id, position):
        with open(os.path.join(self.experiment_dir, f"place_{obj_id}_{self.timestamp}.pkl"), 'wb') as f:
            pickle.dump([obj_id, position], f)
        print("grasp stored")


    def record_conversation(self, message_lst):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")

        with open(os.path.join(self.experiment_dir, f"conversation.pkl"), 'wb') as f:
        # with open(os.path.join(self.experiment_dir, f"conversation_{timestamp}.pkl"), 'wb') as f:
            pickle.dump(message_lst, f)

        print("Conversation recorded!")

    
    def get_location(self, fname):
        return os.path.join(self.experiment_dir, fname)

    # def record_plan(self, robot):

    def get_conversation(self):
        plans = glob(self.get_location(f"conversation.pkl"))
        if len(plans) == 1:
            with open(plans[0], 'rb') as f:
                conversation = pickle.load(f)
            return conversation
        return []

    # def restore_final(self, robot):


    def replay_recording(self, robot, with_actions=True):
        
        observations = glob(self.experiment_dir + "/info*.pkl")

        if len(observations) == 0:
            return False

        timestamps = sorted([obs.split(os.sep)[-1][5:-4] for obs in observations])
        print("Following timestamps found:", timestamps)

        for t in timestamps:
            self.timestamp = t
            with open(self.get_location(f"info_{t}.pkl"), 'rb') as f:
                info_dict = pickle.load(f)

            object_dict = info_dict["object_dicts"]
            segs = info_dict["segs"]
            obs = info_dict["obs"]
            
            for idx, c in enumerate(obs["colors"]):

                cv2.imwrite(os.path.join(self.experiment_dir, 
                                         f"image_{idx}.png"), 
                            c[:,:,::-1])


            try:
                with open(self.get_location(f"description_{t}.txt")) as f:
                    text_description = f.read()
            except FileNotFoundError:
                print("no description file exists")
                text_description = None

            robot.print_object_dicts(object_dict)
            print(text_description)

            robot.object_dicts = object_dict
            robot.scene_description = text_description

            bg_pcd = info_dict["bg_pcd"]
            robot.bg_pcd = bg_pcd

            # combined_pcds = np.concatenate(obs["pcd"], axis=0)
            # combined_pcd, combined_rgb = robot.get_combined_pcd(obs["colors"], obs["depths"])
            # robot.viz.view_pcd(combined_pcd, combined_rgb, name=f'combined')

            # print(object_dict.keys())
            robot.visualize_object_dicts(object_dict)
            # embed()

        
            any_grasps = glob(self.get_location(f"grasp_*_{t}.pkl"))
            print("grasps", any_grasps)

            # assert len(any_grasps) <= 1, "more than one grasps found at one timestamp"

            # if len(any_grasps) == 1:
            for idx in range(len(any_grasps)):
                with open(any_grasps[idx], 'rb') as f:
                    obj_id, grasp_world, grasp_relative = pickle.load(f)

                name = robot.object_dicts[obj_id].get("used_name", "unknown")


                grasp_pose = grasp_world
                robot.picked_pose = grasp_relative

                robot.viz.view_grasps(grasp_pose)
                # return grasp_pose
                pcd = robot.object_dicts[obj_id]["pcd"]
                color = robot.object_dicts[obj_id]["rgb"]
                robot.viz.view_pcd(pcd, color, "test_bowl")

                if with_actions:
                    pick_pose_mat = robot.pick_given_pose(grasp_pose)

                robot.object_dicts[obj_id]["pcd"] = ("in_air", grasp_pose[:3, :3])
                # print("Warning: the feedback is not actually checking the pick success. Make it conditioned")
                robot.feedback_queue.append(f"{name} was picked successfullly.")



            any_places = glob(self.get_location(f"place_*_{t}.pkl"))
            # assert len(any_places) <= 1, "more than one places found at one timestamp"

            # if len(any_places) == 1:
            for idx in range(len(any_places)):
                with open(any_places[idx], 'rb') as f:
                    obj_id, position = pickle.load(f)

                    place_pose_obj = np.eye(4)
                    place_pose_obj[:3, 3] = position
                    place_pose = place_pose_obj @ robot.picked_pose

                    if with_actions:
                        robot.robot.planning.go_home_plan()
                        robot.place_given_pose(place_pose)

                    # input(f"Place enter when placed at location {position}")
                    print("place completed")

                    # robot.object_dicts[obj_id]["pcd"] = ("changed", position)
                    name = robot.object_dicts[obj_id].get("used_name", "unknown")
                    print("Warning: the feedback not linked to successful placement")
                    robot.feedback_queue.append(f"Placing {name} was successful.")

            input("press enter to go to next time step")
            # robot.viz.mc_vis.delete()


        plans = glob(self.get_location(f"conversation.pkl"))
        if len(plans) == 1:
            with open(plans[0], 'rb') as f:
                conversation = pickle.load(f)

            for dialogue in conversation:
                role = dialogue["role"]

                if role == "system":
                    print(Fore.BLUE + "System: " + dialogue["content"])

                elif role == "user":
                    print(Fore.RED + "User: " + dialogue["content"])

                elif role == "assistant":
                    print(Fore.GREEN + "AI: " + dialogue["content"])


            print(Fore.BLACK)

        print("replay finished!")            
        return True

    def print_conversation(self):
        plans = glob(self.get_location(f"conversation.pkl"))
        if len(plans) == 1:
            with open(plans[0], 'rb') as f:
                conversation = pickle.load(f)

            for dialogue in conversation:
                role = dialogue["role"]

                if role == "system":
                    print(Fore.BLUE + "System: " + dialogue["content"])

                elif role == "user":
                    print(Fore.RED + "User: " + dialogue["content"])

                elif role == "assistant":
                    print(Fore.GREEN + "AI: " + dialogue["content"])


            print(Fore.BLACK)
        



    def store_obs(self):
        pass


    def get_obs(self):
        observations = glob(self.experiment_dir + "/info*.pkl")

        if len(observations) == 0:
            return False

        timestamps = sorted([obs.split(os.sep)[-1][5:-4] for obs in observations])
        print("Following timestamps found:", timestamps)

        for t in timestamps[:1]:
            self.timestamp = t
            with open(self.get_location(f"info_{t}.pkl"), 'rb') as f:
                info_dict = pickle.load(f)

            object_dict = info_dict["object_dicts"]
            segs = info_dict["segs"]
            obs = info_dict["obs"]
            
            for idx, c in enumerate(obs["colors"]):

                cv2.imwrite(os.path.join(self.experiment_dir, 
                                         f"image_{idx}.png"), 
                            c[:,:,::-1])
                
            return obs, object_dict

    def get_segments_and_labels(self):
        pass

    def get_segmented_pcd(self):
        pass

    def show_perception(self, saved_dir):
        pass


    # def replay_plan(saved_dir):
    #     pass





# how to record the perception
# the observation - the images, depths, pointclouds 
# the detection results, the segmentation results from sam
# the segmented point cloud for each object
# the generated scene description and updated object dicts
