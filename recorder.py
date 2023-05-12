from colorama import Fore
from glob import glob
import numpy as np
import os
import pickle
import shutil
from datetime import datetime

class Recorder:

    def __init__(self, experiment_name):

        '''
        initialize a new recorder for each experiment,
        set the experiment's name.

        then each function in the robot will use the recorder to 
        '''

        self.EXPERIMENTS_DIR = "../"
        # self.PLAN_RESULTS = "plan_results"

        os.makedirs(self.EXPERIMENTS_DIR, exist_ok=True)
        # os.makedirs(self.PLAN_RESULTS, exist_ok=True)

        self.experiment_dir = os.path.join(self.EXPERIMENTS_DIR, experiment_name)
        if os.path.exists(self.experiment_dir):
            response = input("the experiment dir already exists. To delete press 'y', and press 'n' to conitnue replay:")
            if response == "y":
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

    def show_images(self):
        observations = glob(self.experiment_dir + "/info*.pkl")
        timestamps = sorted([obs.split(os.sep)[-1][5:-4] for obs in observations])
        print(timestamps)
        
        from matplotlib import pyplot as plt
        from robot_env import MyRobot
        idx = 0
        

        robot = MyRobot(meshcat_viz=True)
        os.makedirs(os.path.join(self.experiment_dir, "images"), exist_ok=True)
        for t in timestamps:
            with open(self.get_location(f"info_{t}.pkl"), 'rb') as f:
                info_dict = pickle.load(f)

            object_dict = info_dict["object_dicts"]
            segs = info_dict["segs"]
            obs = info_dict["obs"]
            robot.visualize_object_dicts(object_dict, bg=False)
            input("press enter to continue")

            colors = obs["colors"]
            depths = obs["depths"]
            for c, d in zip(colors, depths):
                plt.imsave(os.path.join(self.experiment_dir, f"images/{idx}_color.png"), c)
                input("press enter")
                d[d > 100] = 5
                plt.imsave(os.path.join(self.experiment_dir, f"images/{idx}_depth.png"), d)
                input("press enter to get next pair of image")
                idx += 1


    def show_conversation(self):
        plans = glob(self.get_location(f"conversation_*.pkl"))
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

        with open(os.path.join(self.experiment_dir, f"conversation_{timestamp}.pkl"), 'wb') as f:
            pickle.dump(message_lst, f)

        print("Conversation recorded!")

    
    def get_location(self, fname):
        return os.path.join(self.experiment_dir, fname)

    def replay_recording(self, robot):
        
        observations = glob(self.experiment_dir + "/info*.pkl")
        timestamps = sorted([obs.split(os.sep)[-1][5:-4] for obs in observations])
        print(timestamps)

        for t in timestamps:
            with open(self.get_location(f"info_{t}.pkl"), 'rb') as f:
                info_dict = pickle.load(f)

            object_dict = info_dict["object_dicts"]
            segs = info_dict["segs"]
            obs = info_dict["obs"]

            with open(self.get_location(f"description_{t}.txt")) as f:
                text_description = f.read()

            robot.print_object_dicts(object_dict)
            print(text_description)

            robot.system_env.object_dicts = object_dict

            bg_pcd = info_dict["bg_pcd"]
            robot.vision_mod.bg_pcd = bg_pcd

            combined_pcds = np.concatenate(obs["pcd"], axis=0)
            robot.viz.view_pcd(combined_pcds.reshape((-1, 3)), name=f'combined')

            robot.visualize_object_dicts(object_dict)

        
            any_grasps = glob(self.get_location(f"grasp_*_{t}.pkl"))
            print("grasps", any_grasps)
            # from IPython import embed
            # embed()
            assert len(any_grasps) <= 1, "more than one grasps found at one timestamp"

            if len(any_grasps) == 1:
                with open(any_grasps[0], 'rb') as f:
                    obj_id, grasp_world, grasp_relative = pickle.load(f)

                name = robot.system_env.object_dicts[obj_id]["used_name"]


                grasp_pose = grasp_world
                robot.picked_pose = grasp_relative

                # return grasp_pose
                pcd = robot.system_env.object_dicts[obj_id]["pcd"]
                color = robot.system_env.object_dicts[obj_id]["rgb"]
                robot.viz.view_pcd(pcd, color, "test_bowl")

                # pick_pose_mat = robot.pick_given_pose(grasp_pose)

                robot.system_env.object_dicts[obj_id]["pcd"] = ("in_air", grasp_pose[:3, :3])
                print("Warning: the feedback is not actually checking the pick success. Make it conditioned")
                robot.feedback_queue.append(f"{name} was picked successfullly.")


            any_places = glob(self.get_location(f"place_*_{t}.pkl"))
            assert len(any_places) <= 1, "more than one places found at one timestamp"

            if len(any_places) == 1:
                with open(any_places[0], 'rb') as f:
                    obj_id, position = pickle.load(f)

                    place_pose_obj = np.eye(4)
                    place_pose_obj[:3, 3] = position
                    place_pose = place_pose_obj @ robot.picked_pose

                    # robot.robot.planning.go_home_plan()
                    # robot.place_given_pose(place_pose)

                    # input(f"Place enter when placed at location {position}")
                    print("place completed")

                    # robot.system_env.object_dicts[obj_id]["pcd"] = ("changed", position)
                    name = robot.system_env.object_dicts[obj_id]["used_name"]
                    print("Warning: the feedback not linked to successful placement")
                    robot.feedback_queue.append(f"Placing {name} was successful.")




        plans = glob(self.get_location(f"conversation_*.pkl"))
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
        return robot
    
    def show_images_and_scene_description(self):
        
        observations = glob(self.experiment_dir + "/info*.pkl")
        timestamps = sorted([obs.split(os.sep)[-1][5:-4] for obs in observations])
        print(timestamps)

        idx = 0
        from matplotlib import pyplot as plt
        for t in timestamps:
            with open(self.get_location(f"info_{t}.pkl"), 'rb') as f:
                info_dict = pickle.load(f)

            # object_dict = info_dict["object_dicts"]
            # segs = info_dict["segs"]
            obs = info_dict["obs"]
            

            with open(self.get_location(f"description_{t}.txt")) as f:
                text_description = f.read()

            print(text_description)

        



    def store_obs(self):
        pass


    def get_obs(self):
        pass 

    def get_segments_and_labels(self):
        pass

    def get_segmented_pcd(self):
        pass

    def show_perception(self, saved_dir):
        pass






if __name__ == "__main__":

    recorder = Recorder("plan_spoon_over_mug")
    recorder.show_conversation()