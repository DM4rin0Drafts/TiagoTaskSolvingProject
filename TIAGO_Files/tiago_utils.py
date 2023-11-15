import math
import os
import random
import re
from collections import namedtuple
from itertools import combinations
from utils.motion.motion_planners import graph


import numpy as np
SIDE_HEIGHT_OFFSET = 0.03


from utils.pybullet_tools.utils import multiply, get_link_pose, joint_from_name, set_joint_position, joints_from_names, \
    set_joint_positions, get_joint_positions, get_min_limit, get_max_limit, quat_from_euler, read_pickle, set_pose, \
    set_base_values, \
    get_pose, euler_from_quat, link_from_name, has_link, point_from_pose, invert, Pose, \
    unit_pose, joints_from_names, PoseSaver, get_aabb, get_joint_limits, get_joints, \
    ConfSaver, get_bodies, create_mesh, remove_body, single_collision, unit_from_theta, angle_between, violates_limit, \
    violates_limits, add_line, get_body_name, get_num_joints, approximate_as_cylinder, \
    approximate_as_prism, unit_quat, unit_point, clip, get_joint_info, tform_point, get_yaw, \
    get_pitch, wait_for_user, quat_angle_between, angle_between, quat_from_pose, compute_jacobian, \
    movable_from_joints, quat_from_axis_angle, LockRenderer, Euler, get_links, get_link_name, \
    draw_point, draw_pose, get_extend_fn, get_moving_links, link_pairs_collision, draw_point, get_link_subtree, \
    clone_body, get_all_links, set_color, pairwise_collision, tform_point, wait_for_duration, add_body_name, RED, GREEN, \
    YELLOW, apply_alpha


#Webot https://cyberbotics.com/doc/guide/tiago-steel
Tiago_GROUPS = {
    'base': ['base_footprint_joint'],
    'torso': ['torso_lift_joint'],                                              #ID: 24
    'head': ['head_1_joint', 'head_2_joint'], 
    'arm': ['arm_1_joint', 'arm_2_joint', 'arm_3_joint', 'arm_4_joint',         #ID: 34, 35, 36
            'arm_5_joint', 'arm_6_joint', 'arm_7_joint'],                       #ID: 37, 38, 39
    'gripper': ['gripper_left_finger_joint', 'gripper_right_finger_joint'], 
    'wheel': ['wheel_left_joint', 'wheel_right_joint']
}

GOAL_POSITION_SETUP = {
    "x_max": 0.75,
    "x_min": 0.45,
    "y_max": 0.5,
    "y_min": 0.3,
    "z": 0.63
}

DATABASES_DIR = '../databases'
IR_FILENAME = '{}_{}_ir.pickle'
IR_CACHE = {}

def get_database_file(filename):
    directory = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(directory, DATABASES_DIR, filename)


#####################################################
########### NOT NEEDED ##############################

Tiago_limits = {
    'arm_joint_1': [0.07, 2.68],
    'arm_joint_2': [-1.5, 1.02],
    'arm_joint_3': [-3.46, 1.5],
    'arm_joint_4': [-0.32, 2.27],
    'arm_joint_5': [-2.07, 2.07],
    'arm_joint_6': [-1.39, 1.39],
    'arm_joint_7': [-2.07, 2.07], 
    'gripper_left_finger_joint': [0, 0.05],
    'gripper_right_finger_joint': [0, 0.05]
}


Tiago_arm_limits = {
    'arm_joint_1': [0.07, 2.68],
    'arm_joint_2': [-1.5, 1.02],
    'arm_joint_3': [-3.46, 1.5],
    'arm_joint_4': [-0.32, 2.27],
    'arm_joint_5': [-2.07, 2.07],
    'arm_joint_6': [-1.39, 1.39],
    'arm_joint_7': [-2.07, 2.07], 
}

Tiago_head_limits = {
    'head_1_joint': [-1.24, 1.24], 
    'head_2_joint': [-0.98, 0.79]
}

Tiago_gripper_limits = {
    'joint_1': [0, 0.05],
    'joint_2': [0, 0.05]
}

Tiago_torso_limits = {
    'torso_lift_joint': [0, 0.35]
}

Tiago_wheel_limits = {
    'wheel_left_joint': [-3.14, 3.14],
    'wheel_right_joint': [-3.14, 3.14]
}

#####################################################
#####################################################


Tiago_Base_Link = 'base_footprint'

Tiago_URDF = "tiago_description/tiago_single.urdf"

# Special Arm configurations
REST_ARM = [0.4303, -1.4589, -0.5566, 2.0267, -1.3867, 1.3321, 0.1261]

TOP_GRAP = [1.5402, 1.0208, -0.008, 1.7240, 1.5701, -0.8393, 0.0344]
LEFT_GRAP = [2.6788, 0.0317, 1.5007, 1.7240, 1.5013, -0.9317, 1.7534]
RIGHT_GRAP = [0.3150, 0.0317, -1.5441, 1.7240, 1.5930, -1.0549, 1.7534]
FRONT_GRAP = [1.5402, 0.7283, -0.0080, 1.4934, 1.5471, 0.8855, 0.0115]
BACK_GRAP = [1.5402, 0.7283, -0.0080, 1.6231, 1.5471, -1.3937, 0.0573]

INITIAL_GRASP_POSITIONS = {
    'rest': REST_ARM,
    'top': TOP_GRAP, 
    'right': RIGHT_GRAP,
    'left': LEFT_GRAP,
    'front': FRONT_GRAP,
    'back': BACK_GRAP
}

CARRY_ARM_CONF = {
    'top:': TOP_GRAP, 
    'left_side': LEFT_GRAP,
    'right_side': RIGHT_GRAP 
}

TIAGO_TOOL_FRAMES = {
    'tiago': 'arm_tool_link'
}

"""
root link: *_gripper_palm_link    --> arm_tool_link, gripper_tool_link, gripper_link, gripper_grasping_frame
tool_link: *_gripper_tool_frame   --> 
"""


"""GET_GRASPS = {
    'top': get_top_grasps,
    'side': get_side_grasps
}"""


def get_initial_conf(grasp_type):
    return INITIAL_GRASP_POSITIONS[grasp_type]


def get_joints_from_body(robot, body_part):
    return joints_from_names(robot, Tiago_GROUPS[body_part])


def get_gripper_link(robot, arm):
    return link_from_name(robot, arm)



def open_arm(robot):
    for joint in get_joints_from_body(robot, "gripper"):
        set_joint_position(robot, joint, get_max_limit(robot, joint))

def close_arm(robot):
    for joint in get_joints_from_body(robot, "gripper"):
        set_joint_position(robot, joint, get_min_limit(robot, joint))


def set_group_conf(robot, body_part, positions):
    set_joint_positions(robot, get_joints_from_body(robot, body_part), positions)



def get_group_conf(robot, group):
    return get_joint_positions(robot, get_joints_from_body(robot, group))


# Box grasps

# GRASP_LENGTH = 0.04
GRASP_LENGTH = 0.
# GRASP_LENGTH = -0.01

# MAX_GRASP_WIDTH = 0.07
MAX_GRASP_WIDTH = np.inf

# Arm tool poses
TOOL_POSE = Pose(euler=Euler(pitch=np.pi / 2)) 


def check_goal(x, y, z):
    if (x < GOAL_POSITION_SETUP["x_max"] and x > GOAL_POSITION_SETUP["x_min"]
        and y < GOAL_POSITION_SETUP["y_max"] and y > GOAL_POSITION_SETUP["y_min"]
        and z == GOAL_POSITION_SETUP["z"]):
        return True
    else:
        return False



def get_side_grasps(body, under=False, tool_pose=TOOL_POSE, body_pose=unit_pose(),
                    max_width=MAX_GRASP_WIDTH, grasp_length=GRASP_LENGTH, top_offset=SIDE_HEIGHT_OFFSET):
    # TODO: compute bounding box width wrt tool frame
    # get the geometric center in body_pose frame, NOT world frame
    center, (w, l, h) = approximate_as_prism(body, body_pose=body_pose)
    translate_center = Pose(point=point_from_pose(body_pose) - center)
    grasps = []
    # x_offset = 0
    x_offset = h / 2 - top_offset
    for j in range(1 + under):
        swap_xz = Pose(euler=[0, -math.pi / 2 + j * math.pi, 0])
        if w <= max_width:
            translate_z = Pose(point=[x_offset, 0, l / 2 - grasp_length])
            for i in range(2):
                rotate_z = Pose(euler=[math.pi / 2 + i * math.pi, 0, 0])
                grasps += [multiply(tool_pose, translate_z, rotate_z, swap_xz,
                                    translate_center, body_pose)]  # , np.array([w])
        if l <= max_width:
            translate_z = Pose(point=[x_offset, 0, w / 2 - grasp_length])
            for i in range(2):
                rotate_z = Pose(euler=[i * math.pi, 0, 0])
                grasps += [multiply(tool_pose, translate_z, rotate_z, swap_xz,
                                    translate_center, body_pose)]  # , np.array([l])
    return grasps



def get_top_grasps(body, under=False, tool_pose=TOOL_POSE, body_pose=unit_pose(),
                max_width=MAX_GRASP_WIDTH, grasp_length=GRASP_LENGTH):
    # TODO: rename the box grasps
    center, (w, l, h) = approximate_as_prism(body, body_pose=body_pose)
    reflect_z = Pose(euler=[0, math.pi, 0])
    translate_z = Pose(point=[0, 0, h / 2 - grasp_length])
    translate_center = Pose(point=point_from_pose(body_pose) - center)
    grasps = []
    if w <= max_width:
        for i in range(1 + under):
            rotate_z = Pose(euler=[0, 0, math.pi / 2 + i * math.pi])
            grasps += [multiply(tool_pose, translate_z, rotate_z,
                                reflect_z, translate_center, body_pose)]
    if l <= max_width:
        for i in range(1 + under):
            rotate_z = Pose(euler=[0, 0, i * math.pi])
            grasps += [multiply(tool_pose, translate_z, rotate_z,
                                reflect_z, translate_center, body_pose)]
    return grasps





def get_side_grasps(body, under=False, tool_pose=TOOL_POSE, body_pose=unit_pose(),
                    max_width=MAX_GRASP_WIDTH, grasp_length=GRASP_LENGTH, top_offset=SIDE_HEIGHT_OFFSET):
    # TODO: compute bounding box width wrt tool frame
    # get the geometric center in body_pose frame, NOT world frame
    center, (w, l, h) = approximate_as_prism(body, body_pose=body_pose)
    translate_center = Pose(point=point_from_pose(body_pose) - center)
    grasps = []
    # x_offset = 0
    x_offset = h / 2 - top_offset
    for j in range(1 + under):
        swap_xz = Pose(euler=[0, -math.pi / 2 + j * math.pi, 0])
        if w <= max_width:
            translate_z = Pose(point=[x_offset, 0, l / 2 - grasp_length])
            for i in range(2):
                rotate_z = Pose(euler=[math.pi / 2 + i * math.pi, 0, 0])
                grasps += [multiply(tool_pose, translate_z, rotate_z, swap_xz,
                                    translate_center, body_pose)]  # , np.array([w])
        if l <= max_width:
            translate_z = Pose(point=[x_offset, 0, w / 2 - grasp_length])
            for i in range(2):
                rotate_z = Pose(euler=[i * math.pi, 0, 0])
                grasps += [multiply(tool_pose, translate_z, rotate_z, swap_xz,
                                    translate_center, body_pose)]  # , np.array([l])
    return grasps



def load_inverse_reachability(grasp_type):
    key = grasp_type
    if key not in IR_CACHE:
        filename = IR_FILENAME.format(grasp_type)
        path = get_database_file(filename)
        IR_CACHE[key] = read_pickle(path)['gripper_from_base']
    return IR_CACHE[key]






def learned_pose_generator(robot, gripper_pose, grasp_type):
    # TODO: record collisions with the reachability database
    gripper_from_base_list = load_inverse_reachability(grasp_type)
    random.shuffle(gripper_from_base_list)
    # handles = []
    for gripper_from_base in gripper_from_base_list:
        base_point, base_quat = multiply(gripper_pose, gripper_from_base)
        x, y, _ = base_point
        _, _, theta = euler_from_quat(base_quat)
        base_values = (x, y, theta)
        # handles.extend(draw_point(np.array([x, y, -0.1]), color=(1, 0, 0), size=0.05))
        # set_base_values(robot, base_values)
        # yield get_pose(robot)
        yield base_values