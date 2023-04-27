import numpy as np
from scipy.spatial.transform import Rotation as R
import copy
import time

def pick_place_plan(grasp, end_grasp):
    '''
    input:
        grasp - a single 4x4 pose of end effector
        current_joints - current pose of the robot
    output:
        joint trajectory for entire motion plan (including pre-grasp pose)
    '''
    # self.reset_joints = list(self.panda.joint_angles().values())
    # rotate grasp by pi/2 about the z axis (urdf fix)
    g_list = [grasp, end_grasp]
    fixed_grasps = []
    for g in g_list:
        #g = self.offset_grasp(g, 0.105)
        fixed_grasps.append(rotate_grasp(g, np.pi/2))

    grasp = fixed_grasps[0]
    end_grasp = fixed_grasps[1]
    pose = pose2tuple(grasp)
    # sol_jnts = list(self.get_ik(pose))

    pre_grasp = offset_grasp(grasp, 0.1)
    pre_place_grasp = offset_grasp(end_grasp, 0.1)
    
    lift_grasp = copy.deepcopy(grasp)
    lift_grasp[2, 3] += 0.2            
    
    pre_pose = pose2tuple(pre_grasp)
    lift_pose = pose2tuple(lift_grasp)
    pre_place_pose = pose2tuple(pre_place_grasp)
    end_pose = pose2tuple(end_grasp)
    
    pose_lst = [
        pre_pose, 
        pose,
        lift_pose,
        end_pose,
        pre_place_pose
    ]

    return pose_lst

def pick_plan(grasp):

    grasp = rotate_grasp(grasp, np.pi/2)
    pose = pose2tuple(grasp)
    # sol_jnts = list(self.get_ik(pose))

    pre_grasp = offset_grasp(grasp, 0.1)
    
    lift_grasp = copy.deepcopy(grasp)
    lift_grasp[2, 3] += 0.2            
    
    pre_pose = pose2tuple(pre_grasp)
    lift_pose = pose2tuple(lift_grasp)

    return [pre_pose, pose, lift_pose]

def place_plan(end_grasp):
    end_grasp = rotate_grasp(end_grasp, np.pi/2)
    end_pose = pose2tuple(end_grasp)
    pre_place_grasp = offset_grasp(end_grasp, 0.1)
    pre_place_pose = pose2tuple(pre_place_grasp)

    return [end_pose, pre_place_pose]


def offset_grasp(grasp, dist):
    offset = np.eye(4)
    offset[2, 3] = -dist
    offset = np.matmul(offset, np.linalg.inv(grasp))
    offset = np.matmul(grasp, offset)
    grasp = np.matmul(offset, grasp)
    return grasp

def rotate_grasp(grasp, theta):
    z_r = R.from_euler('z', theta, degrees=False)
    z_rot = np.eye(4)
    z_rot[:3,:3] = z_r.as_matrix()
    z_rot = np.matmul(z_rot, np.linalg.inv(grasp))
    z_rot = np.matmul(grasp, z_rot)
    grasp = np.matmul(z_rot, grasp)
    return grasp

def pose2tuple(pose):
    pos = pose[:3,-1]
    quat = R.from_matrix(pose[:3,:3]).as_quat()
    return tuple([*pos, *quat])


class PandaReal():
    def __init__(self, robot='franka', viz=True):

        ######################################################################## 
        from franka_interface import ArmInterface
        from franka_ik import FrankaIK
        ########################################################################


        self.panda = ArmInterface()
        self.panda.set_joint_position_speed(3)
        self.joint_names = self.panda._joint_names
        print('joint names: ', self.joint_names)
        self.ik_helper = FrankaIK(gui=True, base_pos=[0, 0, 0], robot=robot)
        # self.world = WorldInterface(viz)

        self.robot_name = robot
        # self.lcc = PointCollision(None, robotiq=robot=='franka_2f140')

    
    def motion_plan(self, grasp, end_grasp, pcd=None):
        '''
        input:
            grasp - a single 4x4 pose of end effector
            current_joints - current pose of the robot
        output:
            joint trajectory for entire motion plan (including pre-grasp pose)
        '''
        self.reset_joints = list(self.panda.joint_angles().values())
        # rotate grasp by pi/2 about the z axis (urdf fix)
        g_list = [grasp, end_grasp]
        fixed_grasps = []
        for g in g_list:
            #g = self.offset_grasp(g, 0.105)
            fixed_grasps.append(rotate_grasp(g, np.pi/2))

        grasp = fixed_grasps[0]
        end_grasp = fixed_grasps[1]
        pose = pose2tuple(grasp)
        sol_jnts = list(self.get_ik(pose))

        pre_grasp = offset_grasp(grasp, 0.1)
        pre_place_grasp = offset_grasp(end_grasp, 0.1)
        
        lift_grasp = copy.deepcopy(grasp)
        lift_grasp[2, 3] += 0.2            
        
        pre_pose = pose2tuple(pre_grasp)
        lift_pose = pose2tuple(lift_grasp)
        pre_place_pose = pose2tuple(pre_place_grasp)
        end_pose = pose2tuple(end_grasp)
        
        pre_jnts = self.get_ik(pre_pose)
        lift_jnts = self.get_ik(lift_pose)
        pre_place_jnts = self.get_ik(pre_place_pose)
        place_jnts = self.get_ik(end_pose)
        if None in [pre_jnts, sol_jnts, lift_jnts, pre_place_jnts, place_jnts]:
            print('having a bad time with IK')
            return None

        pick_plan = self.get_plan([self.pb_robot.arm.get_jpos(), pre_jnts, sol_jnts], pcd=pcd)
        place_plan = self.get_plan([sol_jnts, lift_jnts, place_jnts], pcd=pcd)
        reset_plan = self.get_plan([place_jnts, pre_place_jnts, self.reset_joints], pcd=pcd)
 
        if pick_plan is not None and place_plan is not None and reset_plan is not None:
            return (pick_plan, place_plan, reset_plan)
        else:
            return None

    def get_ik(self, pose, pcd=None):
        jnts = self.ik_helper.get_feasible_ik(pose, target_link=False, pcd=pcd)
        # if jnts is None:
        #     jnts = self.ik_helper.get_ik(pose)
        return jnts                

    def get_plan(self, waypoint_list, pcd=None):
        plan = []

        for i, w in enumerate(waypoint_list[:-1]):
            subplan = self.ik_robot.plan_joint_motion(w, waypoint_list[i+1], pcd=pcd)
            if subplan is not None:
                plan += subplan
            else:
                print('could not find plan')
                return None
        return plan
    
    def plan2dict(self, plan):
        dict_plan = [dict(zip(self.joint_names, val.tolist())) for val in plan]
        return dict_plan

    def execute(self, plan):
        # plan is a tuple!
        self.panda.hand.open()
        s0 = self.panda.execute_position_path(self.plan2dict(plan[0]))
        self.panda.hand.close()
        s1 = self.panda.execute_position_path(self.plan2dict(plan[1]))
        time.sleep(5)
        val = input('press enter to reset')
        self.panda.hand.open()
        s2 = self.panda.execute_position_path(self.plan2dict(plan[2]))
        #self.panda.move_to_neutral()
        print('execute attempt:', s0, s1)
        return (s0, s1)

    def run(self, pick_pose, place_position):

        place_pose = np.copy(pick_pose)
        place_pose[:3, 3] = place_position

        plan = self.motion_plan(pick_pose, place_pose)

        ############################################################################
        for jpos in plan[0]:
            self.ik_helper.set_jpos(jpos)
            time.sleep(0.1)

        ############################################################################

        time.sleep(0.5)
        for jpos in plan[1]:
            self.ik_helper.set_jpos(jpos)
            time.sleep(0.1)
        ############################################################################

        success = self.execute(plan)


from polymetis import GripperInterface, RobotInterface
from real_robot_polymetis.franka_ik import FrankaIK
from real_robot_polymetis.traj_util import PolymetisTrajectoryUtil
from real_robot_polymetis.plan_exec_util import PlanningHelper

from airobot import log_info, log_warn, log_debug, log_critical, set_log_level


class PandaRealPolymetis():
    def __init__(self, robot='franka', viz=None):
        
        self.mc_vis = viz
        self.franka_ip = "173.16.0.1" 

        self.panda = RobotInterface(ip_address=self.franka_ip)

        self.ik_helper = FrankaIK(gui=True, base_pos=[0, 0, 0], occnet=False, robotiq=(self.args.gripper_type=='2f140'), mc_vis=self.mc_vis)
        # self.world = WorldInterface(viz)

        self.robot_name = robot

        self.gripper = GripperInterface(ip_address=self.franka_ip)

        traj_helper = PolymetisTrajectoryUtil(robot=self.panda)
        planning = PlanningHelper(
            mc_vis=self.mc_vis,
            robot=self.panda,
            gripper=self.gripper,
            ik_helper=self.ik_helper,
            traj_helper=traj_helper,
            # tmp_obstacle_dir=tmp_obstacle_dir
        )

        gripper_speed = planning.gripper_speed
        gripper_force = 40.0
        gripper_open_pos = 0.0 if self.args.gripper_type == 'panda' else self.gripper.get_state().max_width
        default_2f140_open_width = 0

        print(gripper_speed, " <= gripper speed")

        planning.set_gripper_speed(gripper_speed)
        planning.set_gripper_force(gripper_force)
        planning.set_gripper_open_pos(gripper_open_pos)
        planning.gripper_open()

        self.planning = planning

    def run(self, pick_pose, place_position):

        pick_pose = rotate_grasp(pick_pose, np.pi/2)

        place_pose = np.copy(pick_pose)
        place_pose[:3, 3] = place_position

        place_offset = np.eye(4)
        place_offset[2, 3] = -0.10

        self.planning.plan_full_path_with_grasp(pick_pose, place_pose, place_offset)



