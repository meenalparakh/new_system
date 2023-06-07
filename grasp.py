import os

# from util import *
import airobot as ar
import numpy as np
import time
import airobot.utils.common as ut
from scipy.spatial.transform import R

def pos_offset_along_approach_vec(approach_vec, offset_dis):
    """
    Args:
        approach_vec: approaching vector, 3-d list
        offset_dis: the offset distance along the approaching vector, positive means along, negative means opposite
    Returns: 3-d numpy array, the original coordinate plus this return value, you can get the translated coordinate
            along approaching vector
    """
    denominator = np.sqrt(
        approach_vec[0] ** 2 + approach_vec[1] ** 2 + approach_vec[2] ** 2
    )
    offset_z = approach_vec[2] / denominator * offset_dis
    offset_y = approach_vec[1] / denominator * offset_dis
    offset_x = approach_vec[0] / denominator * offset_dis
    return np.array([offset_x, offset_y, offset_z])


def control_robot(
    robot,
    pose,
    robot_category="franka",
    control_mode="linear",
    move_up=0.0,
    go_home=False,
    linear_offset=-0.022,
    action="pick",
):
    """
    Given the position and quaternion of target pose, choose the robot arm and control mode, then control the robot
    gripper to target pose
    Args:
        linear_offset: linear offset between gripper position and grasp position (center of two contact points)
        pose: pose[0] position, list of size 3
              pose[1]: often be quaternion, list of size 4; also can be rotation matrix or euler angels
              pose[2[: approaching vector, list of size 3
        robot_category: 'yumi_l' means left arm of YUMI robot, so is 'yumi_r', ur5e means ur5e robot, franka...
        control_mode: 'direct' means compute IK of target pose directly, 'linear' means first set gripper to target
                      orientation, then move the gripper across the line from current position to target position ans
                      keep the orientation unchanged
        move_up: after closing the gripper, whether move gripper up to a certain height
        go_home: return to original state
    """

    def dispatch_control_order(order, pos=None, ori=None):
        return {
            "ur5e:open": lambda: robot.arm.eetool.open(),
            "ur5e:close": lambda: robot.arm.eetool.close(),
            "ur5e:get_pose": lambda: robot.arm.get_ee_pose(),
            "ur5e:set_pose": lambda: robot.arm.set_ee_pose(pos=pos, ori=ori),
            "ur5e:move_xyz": lambda: robot.arm.move_ee_xyz(pos, eef_step=0.01),
            "ur5e:home": lambda: robot.arm.go_home(),
        }.get(order, lambda: None)()

    if robot_category == "franka":
        robot_category = "ur5e"  # The control commands of these two robots are the same in this simulation environment

    # there is a linear offset between gripper position and grasp position center,

    if action == "pick":
        actual_target_pos = pose[0] + pos_offset_along_approach_vec(
            pose[2], linear_offset
        )
        print(
            "target grasp pose: pos|quat|approach vector",
            actual_target_pos,
            pose[1],
            pose[2],
        )


        cur_euler = robot.arm.get_ee_pose()[3]
        pose1_euler = R.from_matrix(pose[1]).as_euler('xyz')

        yaw_mat = R.from_euler('xyz', [0, 0, np.pi]).as_matrix()
        new_rotation = np.matmul(pose[1], yaw_mat)
        pose2_euler = R.from_matrix(new_rotation).as_euler('xyz')

        diff = [np.sum(np.abs(pose1_euler - cur_euler)), 
                np.sum(np.abs(pose2_euler - cur_euler))]
        
        pose[1] = pose[1] if diff[1] < diff[2] else new_rotation


        dispatch_control_order(robot_category + ":open")
        if control_mode == "direct":
            dispatch_control_order(
                robot_category + ":set_pose", actual_target_pos, pose[1]
            )
        elif control_mode == "linear":
            temp_posi = pose[0] + pos_offset_along_approach_vec(pose[2], -0.25)
            dispatch_control_order(robot_category + ":set_pose", temp_posi, pose[1])
            cur_pos, cur_quat, _, cur_euler = dispatch_control_order(
                robot_category + ":get_pose"
            )
            delta_pos = np.array(actual_target_pos) - np.array(cur_pos)
            dispatch_control_order(robot_category + ":move_xyz", delta_pos)
        cur_pos, cur_quat, _, cur_euler = dispatch_control_order(
            robot_category + ":get_pose"
        )
        print("current (pos|quat|euler): ", cur_pos, cur_quat, cur_euler)
        dispatch_control_order(robot_category + ":close")
        dispatch_control_order(robot_category + ":move_xyz", pos=[0, 0, move_up])

        pose_mat = np.eye(4)
        pose_mat[:3, 3] = actual_target_pos
        pose_mat[:3, :3] = pose[1]

        return pose_mat