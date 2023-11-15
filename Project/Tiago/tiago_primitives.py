from __future__ import print_function

import copy
import pybullet as p
import random
import time
from itertools import islice
import matplotlib.pyplot as plt
import numpy as np
from Tiago.load_model import predict_grasp_direction, predict_reachability
from utils.pybullet_tools.body_utils import *

from Tiago.tiago_utils import Tiago_limits, get_group_conf, get_side_grasps, learned_pose_generator, \
                              get_top_grasps, get_joints_from_body, get_gripper_link, \
                              LEFT_GRAP, RIGHT_GRAP, BACK_GRAP, FRONT_GRAP, TOP_GRAP, open_arm, joints_from_names, TIAGO_CUSTOM_GRAP_LIMITS
from utils.pybullet_tools.utils import *


from utils. pybullet_tools.ikfast.tiago.ik import *


BASE_EXTENT = 3.5  # 2.5
BASE_LIMITS = (-BASE_EXTENT * np.ones(2), BASE_EXTENT * np.ones(2))
GRASP_LENGTH = 0.03
APPROACH_DISTANCE = 0.1 + GRASP_LENGTH
SELF_COLLISIONS = False


#######################################################


class sdg_sample_place(object):
    def __init__(self, scn):
        self.all_bodies = scn.all_bodies

    def __call__(self, input_tuple, seed=None):
        """
            Args Description: set object random of a place
                input_tuple: tuple of body_target-object and body_object where the target object should placed in

            return:
                body_grasp: position and orientation of target-object
        """
        body, surface = input_tuple     #robot, oberfläche --> tiago robot???
        others = list(set(self.all_bodies) - {body, surface})
        
        """1) Generation"""
        pose = sample_placement_seed(body, surface, seed)
        print("pose: ", pose)
        """2) Validation"""
        if (pose is None) or any(pairwise_collision(body, b) for b in others):
            return None

        body_pose = BodyPose(body, pose)
        return (body_pose,)  # return a tuple


class BodyGrasp(object):
    def __init__(self, body, grasp_pose, approach_pose, robot, attach_link):
        self.body = body
        self.grasp_pose = grasp_pose  # ee_frame in the measure_frame of the body
        self.approach_pose = approach_pose
        self.robot = robot
        self.link = attach_link

    # def constraint(self):
    #    grasp_constraint()
    def attachment(self):
        attach_pose = invert(self.grasp_pose)  # measure_frame in the ee_frame
        return Attachment(self.robot, self.link, attach_pose, self.body)

    def assign(self):
        return self.attachment().assign()

    def __repr__(self):
        return 'g{}'.format(id(self) % 1000)


class sdg_sample_grasp_dir(object):
    def __init__(self, cnn=False, target=None, target_info=None, robot=None):
        self.cnn = cnn
        self.target = target
        self.robot = robot
        ellipsoid_frame, obj_extent, _, _, _ = get_ellipsoid_frame(target, target_info, robot)
        self.mat_image = get_raytest_scatter3(target, ellipsoid_frame, obj_extent, robot)
        if cnn is None:
            figure, ax = plt.subplots(1)
            import matplotlib.patches as patches
            rect = patches.Rectangle((0, 16), 8, 32, edgecolor='b', facecolor="none", lw=5)
            rect1 = patches.Rectangle((16, 0), 32, 8, edgecolor='b', facecolor="none", lw=5)
            rect2 = patches.Rectangle((58, 16), 8, 32, edgecolor='b', facecolor="none", lw=5)
            rect3 = patches.Rectangle((16, 58), 32, 8, edgecolor='b', facecolor="none", lw=5)
            rect4 = patches.Rectangle((16, 16), 32, 32, edgecolor='b', facecolor="none", lw=5)
            ax.add_patch(rect)
            ax.add_patch(rect1)
            ax.add_patch(rect2)
            ax.add_patch(rect3)
            ax.add_patch(rect4)
            plt.imshow(self.mat_image, 'gray')
            plt.savefig("img" + str(np.random.randint(0, 100)))
            plt.show()
            time.sleep(100)

    def __call__(self, input_tuple, seed=None):
        body, = input_tuple # the target body (aka the box)
        list_available = [0, 1, 2, 3, 4]
        if self.cnn:
            i = 0
            while i < 10:
                i += 1
                direction = cnn_method(self.mat_image, get_pose(self.robot)[0], get_pose(self.target)[0])
                # print(direction)
                if direction is not None:
                    break
        elif seed is None:
            direction = list_available[random.sample(list_available, 1)[0]]
        else:
            direction = list_available[np.array([seed]).flatten()[0]]

        return (GraspDirection(body, direction),)


def cnn_method(mat_image, robot, target):
    # plt.imshow(mat_image, 'gray', vmin=0, vmax=1)
    # plt.show()
    dic = predict_grasp_direction(mat_image)
    # print(dic)
    z1 = robot[2]
    z2 = target[2]
    if dic.get(2)[0] >= 90 and z1 + 0.9 > z2:
        return 2
    elif dic.get(0)[0] >= 50:
        return 0
    elif dic.get(1)[0] >= 50:
        return 1
    elif dic.get(4)[0] >= 50:
        return 4
    elif dic.get(3)[0] >= 50:
        return 3
    return None


class sdg_sample_base_position(object):
    def __init__(self, all_bodies=[], nn=False, target=None, target_info=None):
        self.all_bodies = all_bodies
        self.nn = nn
        self.target = target
        self.target_info = target_info

    def __call__(self, input_tuple, num_of_attempts=30, seed=None):
        box_grasp, surface, grasp_dir = input_tuple
        grasp_dir = grasp_dir.direction
        assert grasp_dir is not None
        robot = box_grasp.robot
        obstacles = list(set(self.all_bodies) - {robot, surface})
        _, _, list_dist, list_dir_jj, list_z_jj = get_ellipsoid_frame(self.target, self.target_info, robot)
        if is_reachable(self.nn, grasp_dir, list_dist[grasp_dir], list_dir_jj[grasp_dir], list_z_jj[grasp_dir])\
                and (not any(pairwise_collision(robot, b) for b in obstacles)) and re(robot, box_grasp):
            body_conf = BodyConf(robot)
            return (body_conf,)

        box_id = box_grasp.body

        while num_of_attempts > 0:
            num_of_attempts -= 1
            """1) Generation"""
            pose = sample_placement(robot, surface)
            """2) Validation"""
            pose = set_orientation_to_object(pose, robot, box_id)
            if (pose is None) or any(pairwise_collision(robot, b) for b in obstacles):
                continue

            _, _, list_dist, list_dir_jj, list_z_jj = get_ellipsoid_frame(self.target, self.target_info, robot)
            if not (is_reachable(self.nn, grasp_dir, list_dist[grasp_dir], list_dir_jj[grasp_dir], list_z_jj[grasp_dir])\
                    and re(robot, box_grasp)):
                continue
            # pose = set_orientation_to_object(pose, robot, box_id)
            body_conf = BodyConf(robot)
            return (body_conf, )
        return None


def re(robot, box):
    robot = list(get_pose(robot)[0])
    app = list(box.approach_pose[0])
    grasp = list(box.grasp_pose[0])

    difference = []
    zip_object = zip(robot, app)
    for list1_i, list2_i in zip_object:
        difference.append(list1_i - list2_i)
    a = np.linalg.norm(difference)

    difference = []
    zip_object = zip(robot, grasp)
    for list1_i, list2_i in zip_object:
        difference.append(list1_i - list2_i)
    b = np.linalg.norm(difference)
    if a > b:
        return True
    return False


def set_orientation_to_object(pose, robot, box_id):
    theta = get_z_rotation(robot, box_id)
    quat = p.getQuaternionFromEuler([0, 0, theta])
    new_pose = (pose[0], quat)
    set_pose(robot, new_pose)
    return new_pose


def get_z_rotation(robot, box_id):
    robot_pose = np.array(get_pose(robot)[0])
    box_pose = np.array(get_pose(box_id)[0])
    target_pose = box_pose - robot_pose
    rad = math.atan2(target_pose[1], target_pose[0])
    return (rad + 2 * np.pi) % (2*np.pi)


def is_reachable(nn, grasp, dist, dir, z):
    if dist <= 0.8:
        if nn:
            res = predict_reachability(grasp, dist, dir, z)
            # print(res)
            return res > 0.5
        else:
            return True
    return False


class sdg_sample_grasp(object):
    def __init__(self, robot, dic_body_info):
        self.robot = robot
        self.end_effector_link = link_from_name(robot, TOOL_FRAMES[get_body_name(robot)])
        self.dic_body_info = dic_body_info

    def set_info(self, info):
        self.dic_body_info = info

    def search(self, input_tuple, seed=None):
        """return the ee_frame wrt the measure_frame of the object"""
        body, grasp_dir = input_tuple  # grasp_dir defined in ellipsoid_frame of the body

        assert body == grasp_dir.body
        grasp_dir = grasp_dir.direction

        ellipsoid_frame, obj_extent, list_dist, list_dir_jj, list_z_jj = get_ellipsoid_frame(body, self.dic_body_info,
                                                                                             self.robot)
        ex, ey, ez = obj_extent
        # mat_image = get_raytest_scatter3(body, ellipsoid_frame, obj_extent, 2)
        # plt.imshow(mat_image, 'gray', vmin=0, vmax=1)
        # plt.show()

        translate_z = Pose(point=[0, 0, -0.001])
        list_grasp = []

        if grasp_dir == 0:
            """ee at +X"""
            swap_z = Pose(euler=[np.pi, 0, 0])
            d1, d2 = 0., 0.  # [-0.5, 0.5]
            translate_point = Pose(point=[ex / 2, 0 + d1 * ey, ez / 2 + d2 * ez])
            for j in range(1):
                rotate_z = Pose(euler=[0, 0, j * 2 * np.pi + np.pi])
                grasp = multiply(translate_point, swap_z, rotate_z, translate_z)
                list_grasp.append(grasp)

            approach_pose = Pose(0.1 * Point(x=-1))  # pose bias wrt end-effector frame

        elif grasp_dir == 1:
            """ee at +Y"""
            swap_z = Pose(euler=[0, 0, - np.pi / 2])
            d1, d2 = 0., 0.  # [-0.5, 0.5]
            translate_point = Pose(point=[0 - d1 * ex, ey / 2, ez / 2 + d2 * ez])
            for j in range(1):
                rotate_z = Pose(euler=[j * -np.pi / 2, 0, 0])
                grasp = multiply(translate_point, swap_z, rotate_z, translate_z)
                list_grasp.append(grasp)

            approach_pose = Pose(0.1 * Point(x=-1))  # pose bias wrt end-effector frame

        elif grasp_dir == 2:
                """ee at +Z of the ellipsoid_frame"""
                swap_z = Pose(euler=[0, np.pi / 2, 0])
                d1, d2 = 0., -0.  # [-0.5, 0.5]
                translate_point = Pose(point=[0 - d2 * ex, 0 + d1 * ey, ez])
                for j in range(4):
                    rotate_z = Pose(euler=[j * np.pi / 2, 0,  0])
                    grasp = multiply(translate_point, swap_z, rotate_z, translate_z)
                    list_grasp.append(grasp)

                approach_pose = Pose(0.1 * Point(x=-1))  # pose bias wrt end-effector frame

        elif grasp_dir == 3:
            """ee at -X of the ellipsoid_frame"""
            swap_z = Pose(euler=[- np.pi, 0,  0])
            # translate_point: choose from the grasping surface with 2 dof
            d1, d2 = 0., 0.  # [-0.5, 0.5]
            translate_point = Pose(point=[-ex / 2, 0 - d1 * ey, ez / 2 + d2 * ez])
            for j in range(1):
                rotate_z = Pose(euler=[0, 0,  -j * 2 * np.pi])
                grasp = multiply(translate_point, swap_z, rotate_z, translate_z)
                list_grasp.append(grasp)

            approach_pose = Pose(0.1 * Point(x=-1))  # pose bias wrt end-effector frame

        elif grasp_dir == 4:
            """ee at -Y"""
            swap_z = Pose(euler=[0, 0, np.pi / 2])
            d1, d2 = 0., 0.  # [-0.5, 0.5]
            translate_point = Pose(point=[0 + d1 * ex, -ey / 2, ez / 2 + d2 * ez])
            for j in range(1):
                rotate_z = Pose(euler=[j * np.pi, 0, 0])
                grasp = multiply(translate_point, swap_z, rotate_z, translate_z)
                list_grasp.append(grasp)

            approach_pose = Pose(0.1 * Point(x=-1))  # pose bias wrt end-effector frame

        else:
            raise ValueError("ValueError exception thrown: grasp_dir is invalid: ", grasp_dir)

        """ee_frame wrt ellipsoid_frame"""
        grasp_pose = random.sample(list_grasp, 1)[0]
        
        """ee_frame wrt measure_frame: get_pose()"""
        grasp_pose = multiply(invert(get_pose(body)), pose_from_tform(ellipsoid_frame), grasp_pose)

        #approach_pose = Pose(0.1 * Point(x=-1))  # pose bias wrt end-effector frame
        body_grasp = BodyGrasp(body, grasp_pose, approach_pose, self.robot, self.end_effector_link)
        return (body_grasp,)  # return a tuple

    def __call__(self, input_tuple, seed=None):
        return self.search(input_tuple, seed=None)

DISABLED_COLLISION_PAIR = {(12, 0), (14, 0), (16, 0), (18, 0), (20, 0), (22, 0), (24, 49), (39, 41), (39, 42)}


class GraspDirection(object):
    def __init__(self, body, direction):
        if isinstance(body, tuple):
            body = body[0]
        self.body = body
        self.direction = direction

    def __repr__(self):
        return 'gd{}'.format(id(self) % 1000)

class BodyInfo(object):
    def __init__(self, scn, body):
        if isinstance(body, tuple):
            body = body[0]
        self.body = body
        self.info = scn.dic_body_info[body]

    def __repr__(self):
        return 'bi{}'.format(id(self) % 1000)

class sdg_ik_grasp(object):
    def __init__(self, robot, all_bodies=[], teleport=False, num_attempts=25):
        self.all_bodies = all_bodies
        self.teleport = teleport
        self.num_attempts = num_attempts
        self.movable_joints = get_movable_joints(robot)
        self.sample_fn = get_sample_fn(robot, self.movable_joints, custom_limits=Tiago_limits)
        self.robot = robot
        self.visualization_collision = False
        self.max_distance = MAX_DISTANCE

    def search(self, input_tuple, seed=None):
        body, pose, grasp = input_tuple  # pose is measured by get_pose()

        set_pose(body, pose.value)

        obstacles = list(set(self.all_bodies) - {grasp.body, 12, 14, 16, 18, 20, 22})

        grasp_pose_ee = multiply(pose.value, grasp.grasp_pose)  # in world frame
        approach_pose_ee = multiply(grasp_pose_ee, grasp.approach_pose)  # 右乘,以当前ee坐标系为基准进行变换
        # ori = p.getEulerFromQuaternion(grasp_pose_ee[1])
        # a_ori = p.getEulerFromQuaternion(approach_pose_ee[1])
        list_q_approach = []
        list_q_grasp = []
        list_test_collision = []
        list_command_approach = []

        for _ in range(self.num_attempts):
            # time.sleep(1)
            sampled_conf = self.sample_fn()
            set_joint_positions(self.robot, self.movable_joints, sampled_conf)  # Random seed

            q_approach = inverse_kinematics(self.robot, grasp.link, approach_pose_ee)
            q_grasp = inverse_kinematics(self.robot, grasp.link, grasp_pose_ee)

            if q_approach and q_grasp:
                
                list_q_approach.append(q_approach)
                list_q_grasp.append(q_grasp)
                set_joint_positions(self.robot, self.movable_joints, q_approach)
                no_collision = not any(
                    pairwise_collision(self.robot, b,
                                       visualization=self.visualization_collision,
                                       max_distance=self.max_distance)
                    for b in obstacles)
                set_joint_positions(self.robot, self.movable_joints, q_grasp)
                no_collision = no_collision and (not any(
                    pairwise_collision(self.robot, b,
                                       visualization=self.visualization_collision,
                                       max_distance=self.max_distance)
                    for b in obstacles))
                list_test_collision.append(no_collision)
                trajectory = None
                approach_conf = None
                if no_collision:
                    
                    approach_conf = BodyConf(self.robot, q_approach)
                    if self.teleport:
                        path = [q_approach, q_grasp]
                    else:
                        approach_conf.assign()
                        # The path from q_approach to q_grasp.
                        path = plan_direct_joint_motion(self.robot, approach_conf.joints, q_grasp,
                                                        obstacles=obstacles,
                                                        disabled_collisions=DISABLED_COLLISION_PAIR,
                                                        max_distance=self.max_distance)
                        if path:
                            trajectory = create_trajectory(self.robot, self.movable_joints, path)
                            command = Command([BodyPath(self.robot, path),
                                               Attach(body, self.robot, grasp.link),
                                               BodyPath(self.robot, path[::-1], attachments=[grasp])])
                # list_command_approach.append(command)
                list_command_approach.append(trajectory)
                if trajectory:
                    set_joint_positions(self.robot, self.movable_joints, list_q_grasp[0])
                    return approach_conf, trajectory, q_approach, q_grasp

        # jp = get_joint_positions(self.robot, self.movable_joints)
        # ee_pose = get_link_pose(self.robot, grasp.link)
        #
        # err1 = np.array(ee_pose[0]) - np.array(grasp_pose_ee[0])
        # err2 = np.array(ee_pose[0]) - np.array(approach_pose_ee[0])

        if list_q_approach and list_q_grasp:
            set_joint_positions(self.robot, self.movable_joints, list_q_grasp[0])
            return None, None, list_q_approach[0], list_q_grasp[0]
        return None, None, None, None

    def __call__(self, input_tuple, seed=None):
        approach_conf, command, q_approach, q_grasp = self.search(input_tuple, seed=None)

        if command is None:
            return None
        else:
            return approach_conf, command

DEBUG_FAILURE = False

class sdg_plan_free_motion(object):
    def __init__(self, robot, all_bodies=[], teleport=False, self_collisions=False):
        self.all_bodies = all_bodies
        self.teleport = teleport
        self.self_collisions = self_collisions
        self.robot = robot
        self.max_distance = MAX_DISTANCE

    def __call__(self, input_tuple, seed=None):
        conf1, conf2 = input_tuple

        assert ((conf1.body == conf2.body) and (conf1.joints == conf2.joints))
        if self.teleport:
            path = [conf1.configuration, conf2.configuration]
        else:
            conf1.assign()
            # obstacles = fixed + assign_fluent_state(fluents)
            obstacles = self.all_bodies
            path = plan_joint_motion(self.robot, conf2.joints, conf2.configuration, obstacles=obstacles,
                                     self_collisions=self.self_collisions, disabled_collisions=DISABLED_COLLISION_PAIR,
                                     max_distance=self.max_distance)
            if path is None:
                if DEBUG_FAILURE: user_input('Free motion failed')
                return None
        return (create_trajectory(self.robot, conf2.joints, path), )
        # command = Command([BodyPath(])
        # return (command,)  # return a tuple



class ApplyForce(object):
    def __init__(self, body, robot, link):
        self.body = body
        self.robot = robot
        self.link = link

    def bodies(self):
        return {self.body, self.robot}

    def iterator(self, **kwargs):
        return []

    def refine(self, **kwargs):
        return self

    def __repr__(self):
        return '{}({},{})'.format(self.__class__.__name__, self.robot, self.body)





class Attach(ApplyForce):
    def control(self, **kwargs):
        # TODO: store the constraint_id?
        add_fixed_constraint(self.body, self.robot, self.link)

    def reverse(self):
        return Detach(self.body, self.robot, self.link)



class Detach(ApplyForce):
    def control(self, **kwargs):
        remove_fixed_constraint(self.body, self.robot, self.link)

    def reverse(self):
        return Attach(self.body, self.robot, self.link)





class BodyPath(object):
    def __init__(self, body, path, joints=None, attachments=[]):
        if joints is None:
            joints = get_movable_joints(body)
        self.body = body  # robot
        self.path = path
        self.joints = joints
        self.attachments = attachments

    def bodies(self):
        return set([self.body] + [attachment.body for attachment in self.attachments])

    def iterator(self):
        # TODO: compute and cache these
        # TODO: compute bounding boxes as well
        for i, configuration in enumerate(self.path):
            set_joint_positions(self.body, self.joints, configuration)
            for grasp in self.attachments:
                grasp.assign()
            yield i

    def control(self, real_time=False, dt=0):
        # TODO: just waypoints
        if real_time:
            enable_real_time()
        else:
            disable_real_time()
        for values in self.path:
            for _ in joint_controller(self.body, self.joints, values):
                enable_gravity()
                if not real_time:
                    step_simulation()
                time.sleep(dt)

    # def full_path(self, q0=None):
    #     # TODO: could produce sequence of savers
    def refine(self, num_steps=0):
        return self.__class__(self.body, refine_path(self.body, self.joints, self.path, num_steps), self.joints,
                              self.attachments)

    def reverse(self):
        return self.__class__(self.body, self.path[::-1], self.joints, self.attachments)

    def distance(self):
        """
        Return the trip summed up with of each selected link during the path.
        """
        robot = self.body
        total = 0.
        for q1, q2 in zip(self.path, self.path[1:]):
            # total += distance_fn(q1.values, q2.values)
            links = list(get_links(robot))
            total += get_links_movement(robot, links[3:9], q1, q2)

        return total

    def __repr__(self):
        return '{}({},{},{},{})'.format(self.__class__.__name__, self.body, len(self.joints), len(self.path),
                                        len(self.attachments))


class BodyConf(object):
    def __init__(self, body, configuration=None, joints=None):
        if joints is None:
            joints = get_movable_joints(body)
        if configuration is None:
            configuration = get_joint_positions(body, joints)
        self.body = body
        self.joints = joints
        self.configuration = configuration

    def assign(self):
        set_joint_positions(self.body, self.joints, self.configuration)
        return self.configuration

    def __repr__(self):
        return 'q{}'.format(id(self) % 1000)



class sdg_motion_base_joint(object):
    def __init__(self, scn, max_attempts=25, custom_limits={}, teleport=False, **kwargs):
        self.max_attempts = max_attempts
        self.teleport = teleport
        self.custom_limits = custom_limits

        self.robot = scn.pr2
        self.obstacles = list(set(scn.env_bodies) | set(scn.regions))

        self.saver = BodySaver(self.robot)

    def search(self, input_tuple, seed=None):
        bq1, bq2 = input_tuple
        self.saver.restore()
        bq1.assign()

        for i in range(self.max_attempts):
            if self.teleport:
                path = [bq1, bq2]
            elif is_drake_pr2(self.robot):
                raw_path = plan_joint_motion(self.robot, bq2.joints, bq2.values, attachments=[],
                                             obstacles=self.obstacles, custom_limits=self.custom_limits,
                                             self_collisions=SELF_COLLISIONS,
                                             restarts=4, iterations=50, smooth=50)
                if raw_path is None:
                    print('Failed motion plan!')
                    continue
                path = [Conf(self.robot, bq2.joints, q) for q in raw_path]
            else:
                goal_conf = base_values_from_pose(bq2.value)
                raw_path = plan_base_motion(self.robot, goal_conf, BASE_LIMITS, obstacles=self.obstacles)
                if raw_path is None:
                    print('Failed motion plan!')
                    continue
                path = [BodyPose(self.robot, pose_from_base_values(q, bq1.value)) for q in raw_path]
            bt = Trajectory(path)
            cmd = Commands(State(), savers=[BodySaver(self.robot)], commands=[bt])
            return (cmd,)
        return None

    def __call__(self, input_tuple, seed=None):
        return self.search(input_tuple, seed=None)

#######################################################

class BodyPose(object):
    # def __init__(self, position, orientation):
    #    self.position = position
    #    self.orientation = orientation
    def __init__(self, body, value=None, support=None, init=False):
        self.body = body
        if value is None:
            value = get_pose(self.body)
        self.value = tuple(value)
        self.support = support
        self.init = init

    def assign(self):
        set_pose(self.body, self.value)

    def iterate(self):
        yield self

    def to_base_conf(self):
        values = base_values_from_pose(self.value)
        return Conf(self.body, range(len(values)), values)

    def __repr__(self):
        return 'p{}'.format(id(self) % 1000)



class Conf(object):
    def __init__(self, body, joints, values=None, init=False):
        self.body = body
        self.joints = joints
        if values is None:
            values = get_joint_positions(self.body, self.joints)
        self.values = tuple(values)
        self.init = init

    def assign(self):
        set_joint_positions(self.body, self.joints, self.values)

    def iterate(self):
        yield self

    def __repr__(self):
        return 'q{}'.format(id(self) % 1000)

#####################################

class Command(object):
    def __init__(self, body_paths):
        self.body_paths = body_paths

    # def full_path(self, q0=None):
    #     if q0 is None:
    #         q0 = Conf(self.tree)
    #     new_path = [q0]
    #     for partial_path in self.body_paths:
    #         new_path += partial_path.full_path(new_path[-1])[1:]
    #     return new_path

    def step(self):
        for i, body_path in enumerate(self.body_paths):
            for j in body_path.iterator():
                msg = '{},{}) step?'.format(i, j)
                user_input(msg)
                # print(msg)

    def execute(self, time_step=0.05):
        for i, body_path in enumerate(self.body_paths):
            for j in body_path.iterator():
                # time.sleep(time_step)
                wait_for_duration(time_step)

    def control(self, real_time=False, dt=0):  # TODO: real_time
        for body_path in self.body_paths:
            body_path.control(real_time=real_time, dt=dt)

    def refine(self, **kwargs):
        return self.__class__([body_path.refine(**kwargs)
                               for body_path in self.body_paths])

    def reverse(self):
        return self.__class__([body_path.reverse() for body_path in reversed(self.body_paths)])

    def __repr__(self):
        return 'c{}'.format(id(self) % 1000)


class Commands(object):
    def __init__(self, state, savers=[], commands=[]):
        self.state = state
        self.savers = tuple(savers)
        self.commands = tuple(commands)

    def assign(self):
        for saver in self.savers:
            saver.restore()
        return copy.copy(self.state)

    def apply(self, state, **kwargs):
        for command in self.commands:
            for result in command.apply(state, **kwargs):
                yield result

    def __repr__(self):
        return 'c{}'.format(id(self) % 1000)




#####################################

def get_target_point(conf):
    # TODO: use full body aabb
    robot = conf.body
    link = link_from_name(robot, 'torso_lift_link')
    # link = BASE_LINK
    # TODO: center of mass instead?
    # TODO: look such that cone bottom touches at bottom
    # TODO: the target isn't the center which causes it to drift
    with BodySaver(conf.body):
        conf.assign()
        lower, upper = get_aabb(robot, link)
        center = np.average([lower, upper], axis=0)
        point = np.array(get_group_conf(conf.body, 'base'))
        # point[2] = upper[2]
        point[2] = center[2]
        # center, _ = get_center_extent(conf.body)
        return point


def get_target_path(trajectory):
    # TODO: only do bounding boxes for moving links on the trajectory
    return [get_target_point(conf) for conf in trajectory.path]


#######################################################

#####################################



class Trajectory(Command):
    _draw = False

    def __init__(self, path):
        self.path = tuple(path)
        # TODO: constructor that takes in this info

    def apply(self, state, sample=1):
        handles = add_segments(self.to_points()) if self._draw and has_gui() else []
        for conf in self.path[::sample]:
            conf.assign()
            yield
        end_conf = self.path[-1]
        if isinstance(end_conf, BodyPose):
            state.poses[end_conf.body] = end_conf
        for handle in handles:
            remove_debug(handle)

    def control(self, dt=0, **kwargs):
        # TODO: just waypoints
        for conf in self.path:
            if isinstance(conf, BodyPose):
                conf = conf.to_base_conf()
            for _ in joint_controller_hold(conf.body, conf.joints, conf.values):
                step_simulation()
                time.sleep(dt)

    def to_points(self, link=BASE_LINK):
        # TODO: this is computationally expensive
        points = []
        for conf in self.path:
            with BodySaver(conf.body):
                conf.assign()
                # point = np.array(point_from_pose(get_link_pose(conf.body, link)))
                point = np.array(get_group_conf(conf.body, 'base'))
                point[2] = 0
                point += 1e-2 * np.array([0, 0, 1])
                if not (points and np.allclose(points[-1], point, atol=1e-3, rtol=0)):
                    points.append(point)
        points = get_target_path(self)
        return waypoints_from_path(points)

    def distance(self, distance_fn=get_distance):
        total = 0.
        for q1, q2 in zip(self.path, self.path[1:]):
            total += distance_fn(q1.values, q2.values)
        return total

    def iterate(self):
        for conf in self.path:
            yield conf

    def reverse(self):
        return Trajectory(reversed(self.path))

    # def __repr__(self):
    #    return 't{}'.format(id(self) % 1000)
    def __repr__(self):
        d = 0
        if self.path:
            conf = self.path[0]
            d = 3 if isinstance(conf, BodyPose) else len(conf.joints)
        return 't({},{})'.format(d, len(self.path))



def create_trajectory(robot, joints, path):
    return Trajectory(Conf(robot, joints, q) for q in path)



class Grasp(object):
    def __init__(self, grasp_type, body, value, approach, carry):
        self.grasp_type = grasp_type
        self.body = body
        self.value = tuple(value)  # gripper_from_object
        self.approach = tuple(approach)
        self.carry = tuple(carry)

    def attachment(self, robot):
        tool_link = link_from_name(robot, "gripper_grasping_frame") #TODO???? tool_frames
        return Attachment(robot, tool_link, self.value, self.body)

    def __repr__(self):
        return 'g{}'.format(id(self) % 1000)



"""
root link: *_gripper_palm_link    --> arm_tool_link, gripper_tool_link, gripper_link, gripper_grasping_frame
tool_link: *_gripper_tool_frame   --> 
"""
def iterate_approach_path(robot, pose, grasp, body=None):
    root_link = link_from_name(robot, 'gripper_tool_link')        #TODO ?
    tool_link = link_from_name(robot, "gripper_grasping_frame") #TODO ?
    tool_from_root = multiply(invert(get_link_pose(robot, tool_link)),
                              get_link_pose(robot, root_link))
    grasp_pose = multiply(pose.value, invert(grasp.value))
    approach_pose = multiply(pose.value, invert(grasp.approach))
    for tool_pose in interpolate_poses(grasp_pose, approach_pose):
        set_pose(body, multiply(tool_pose, tool_from_root))
        if body is not None:
            set_pose(body, multiply(tool_pose, grasp.value))
        yield



def get_ir_sampler(scn, custom_limits={}, max_attempts=25, collisions=True, learned=True):
    robot = scn.robots[0]
    fixed_movable = scn.all_bodies

    def gen_fn(obj, pose, grasp):
        obstacles = list(set(fixed_movable) - {obj})
        pose.assign()
        approach_obstacles = {obst for obst in obstacles if (not is_placement(obj, obst))}

        for _ in iterate_approach_path(robot, pose, grasp, body=obj):
            if any(pairwise_collision(obj, b) for b in approach_obstacles):             #TODO pairwise_collision(b) or ???
                return
        gripper_pose = multiply(pose.value, invert(grasp.value))  # w_f_g = w_f_o * (g_f_o)^-1
        default_conf = grasp.carry
        arm_joints = get_joints_from_body(robot, "arm")
        print("arm: ", arm_joints)
        base_joints = get_joints_from_body(robot, 'base')[0]
        print("base: ", base_joints)
        if learned:
            base_generator = learned_pose_generator(robot, gripper_pose, grasp_type=grasp.grasp_type)
        else:
            base_generator = uniform_pose_generator(robot, gripper_pose)
        lower_limits, upper_limits = get_custom_limits(robot, base_joints, custom_limits)
        while True:
            count = 0
            for base_conf in islice(base_generator, max_attempts):
                count += 1
                if not all_between(lower_limits, base_conf, upper_limits):
                    continue
                bq = Conf(robot, base_joints, base_conf)
                pose.assign()
                bq.assign()
                set_joint_positions(robot, arm_joints, default_conf)
                if any(pairwise_collision(robot, b) for b in obstacles + [obj]):
                    continue
                # print('IR attempts:', count)
                yield (bq,)
                break
            else:
                yield None

    return gen_fn


##################################################

def get_ik_fn(scn, custom_limits={}, collisions=True, teleport=False):
    robot = scn.robots[0]
    fixed_movable = scn.all_bodies

    if is_ik_compiled():
        print('Using ikfast for inverse kinematics')
    else:
        print('Using pybullet for inverse kinematics')

    def fn(arm, obj, pose, grasp, base_conf):
        obstacles = list(set(fixed_movable) - {obj})
        approach_obstacles = {obst for obst in obstacles if not is_placement(obj, obst)}
        gripper_pose = multiply(pose.value, invert(grasp.value))  # w_f_g = w_f_o * (g_f_o)^-1
        # approach_pose = multiply(grasp.approach, gripper_pose)
        approach_pose = multiply(pose.value, invert(grasp.approach))
        arm_link = get_gripper_link(robot, arm)
        arm_joints = get_joints_from_body(robot, 'arm')

        default_conf =  grasp.carry
        # sample_fn = get_sample_fn(robot, arm_joints)
        pose.assign()
        base_conf.assign()
        open_arm(robot, arm)
        set_joint_positions(robot, arm_joints, default_conf)  # default_conf | sample_fn()
        grasp_conf = tiago_inverse_kinematics(robot, arm, gripper_pose,
                                            custom_limits=custom_limits)  # , upper_limits=USE_CURRENT)
        # nearby_conf=USE_CURRENT) # upper_limits=USE_CURRENT,
        if (grasp_conf is None) or any(pairwise_collision(robot, b) for b in obstacles):  # [obj]
            # print('Grasp IK failure', grasp_conf)
            # if grasp_conf is not None:
            #    print(grasp_conf)
            #    #wait_for_user()
            return None
        # approach_conf = pr2_inverse_kinematics(robot, arm, approach_pose, custom_limits=custom_limits,
        #                                       upper_limits=USE_CURRENT, nearby_conf=USE_CURRENT)
        approach_conf = sub_inverse_kinematics(robot, arm_joints[0], arm_link, approach_pose,
                                               custom_limits=custom_limits)
        if (approach_conf is None) or any(pairwise_collision(robot, b) for b in obstacles + [obj]):
            # print('Approach IK failure', approach_conf)
            # wait_for_user()
            return None
        approach_conf = get_joint_positions(robot, arm_joints)
        attachment = grasp.attachment(robot, arm)
        attachments = {attachment.child: attachment}
        if teleport:
            path = [default_conf, approach_conf, grasp_conf]
        else:
            resolutions = 0.05 ** np.ones(len(arm_joints))
            grasp_path = plan_direct_joint_motion(robot, arm_joints, grasp_conf, attachments=attachments.values(),
                                                  obstacles=approach_obstacles, self_collisions=SELF_COLLISIONS,
                                                  custom_limits=custom_limits, resolutions=resolutions / 2.)
            if grasp_path is None:
                print('Grasp path failure')
                return None
            set_joint_positions(robot, arm_joints, default_conf)
            approach_path = plan_joint_motion(robot, arm_joints, approach_conf, attachments=attachments.values(),
                                              obstacles=obstacles, self_collisions=SELF_COLLISIONS,
                                              custom_limits=custom_limits, resolutions=resolutions,
                                              restarts=2, iterations=25, smooth=25)
            if approach_path is None:
                print('Approach path failure')
                return None
            path = approach_path + grasp_path
        mt = create_trajectory(robot, arm_joints, path)
        cmd = Commands(State(attachments=attachments), savers=[BodySaver(robot)], commands=[mt])
        return (cmd,)

    return fn



class State(object):
    def __init__(self, attachments={}, cleaned=set(), cooked=set()):
        self.poses = {body: BodyPose(body, get_pose(body))
                      for body in get_bodies() if body not in attachments}
        self.grasps = {}
        self.attachments = attachments
        self.cleaned = cleaned
        self.cooked = cooked

    def assign(self):
        for attachment in self.attachments.values():
            # attach.attachment.assign()
            attachment.assign()