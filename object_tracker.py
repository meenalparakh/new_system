import numpy as np
from robot_env import MyRobot
import cv2
from multiprocessing.pool import ThreadPool
import threading
from threading import Thread
import pickle
import os

class MyObjectTracker:
    def __init__(self, tracker="goturn"):
        self.tracker = cv2.TrackerGOTURN_create()

    def register_bbs(self, image, bb):
        success = self.tracker.init(image, bb)
        return success
    
    def track(self):
        ok, bbox = self.tracker.update(frame)

def store_obs(robot, n=50):
    for i in range(n):
        obs = robot.get_obs()
        robot.obs_lst.append(obs)

if __name__ == "__main__":

    robot = MyRobot(gui=True)
    robot.reset("cup_over_bowl")

    with open("object_dicts.pkl", 'rb') as f:
        robot.object_dicts = pickle.load(f)    

    robot.obs_lst = []
    thread = Thread(target=store_obs, args=(robot,))
    thread.start()

    bowl_id = robot.find("bowl")
    robot.pick(bowl_id)

    thread.join()

    os.makedirs("threading_check", exists_ok=True)
    
    colors = [o["colors"] for o in robot.obs_lst]
    for idx, c in enumerate(colors):
        cv2.imwrite(f"threading_check/{idx}.png")

    
