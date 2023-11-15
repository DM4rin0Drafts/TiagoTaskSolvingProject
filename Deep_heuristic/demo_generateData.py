#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import argparse
from datetime import datetime
import time
from utils.pybullet_tools.kuka_primitives import BodyPose, BodyConf
from utils.pybullet_tools.utils import WorldSaver, connect, dump_world, get_pose, set_pose, Pose, \
    Point, set_default_camera, stable_z, disconnect, get_bodies, HideOutput, \
    create_box, load_pybullet, step_simulation, Euler, get_links, get_link_info, get_movable_joints, \
    set_joint_positions, set_camera, get_raytest_scatter, set_collision_group_mask, get_center_extent, tform_from_pose, \
    set_color, remove_body, pose_from_tform
from copy import copy
from utils.pybullet_tools.body_utils import *
from utils.pybullet_tools.kuka_primitives2 import sdg_ik_grasp, sdg_sample_grasp, GraspDirection, get_ee_pose, \
    BodyPose
import pickle as pk
import os

C_Red = [0.875, 0.200, 0.216, 1.0]
C_Wood = [0.827, 0.675, 0.463, 1.0]
C_LightWood = [0.992, 0.871, 0.682, 1.0]
C_Region = [0.890, 0.855, 0.824, 1.0]


def get_fixed(robot, movable):
    rigid = [body for body in get_bodies() if body != robot]
    fixed = [body for body in rigid if body not in movable]
    return fixed


def place_movable(certified):
    placed = []
    for literal in certified:
        if literal[0] == 'not':
            fact = literal[1]
            if fact[0] == 'trajcollision':
                _, b, p = fact[1:]
                set_pose(b, p.pose)
                placed.append(b)
    return placed


def get_area_p(list_region):
    sum_area = 0
    list_area = []
    for r in list_region:
        _, extent = get_center_extent(r)
        # area = extent[0] * extent[1]
        area = np.sqrt(extent[0] ** 2 + extent[1] ** 2)
        sum_area += area
        list_area.append(area)
    results = []
    for a in list_area:
        results.append(a / sum_area)
    return results


def get_rand(low=0.08, high=0.20):
    return np.random.uniform(low, high)


class PlanningScenario(object):
    def __init__(self):
        with HideOutput():
            self.arm_left = load_pybullet("darias_description/urdf/darias_L_primitive_collision.urdf",
                                          fixed_base=True)
            # self.arm_left = load_pybullet("../da_description/robots/left_no_hands.urdf",
            #                               fixed_base=True)

            # self.arm_left = load_pybullet("../lwr_description/kuka_lwr_left.urdf",
            #                               fixed_base=True)
            self.arm_base = load_pybullet("darias_description/urdf/darias_base.urdf", fixed_base=True)

            """import fixed objects"""
            self.floor = load_pybullet("../scenario_description/floor.urdf", fixed_base=True)

            self.cabinet_shelf = load_pybullet("../scenario_description/manipulation_worlds/urdf/cabinet_shelf.urdf",
                                               fixed_base=True)
            self.drawer_shelf = load_pybullet("../scenario_description/manipulation_worlds/urdf/drawer_shelf.urdf",
                                              fixed_base=True)
            self.pegboard = load_pybullet("../scenario_description/manipulation_worlds/urdf/pegboard.urdf",
                                          fixed_base=True)

            self.drawer_links = get_links(self.drawer_shelf)
            cabinet_links = get_links(self.cabinet_shelf)

            # on table
            self.region_t1 = load_pybullet("../scenario_description/region_table2.urdf", fixed_base=True)
            set_color(self.region_t1, C_Region)
            self.region_t2 = load_pybullet("../scenario_description/region_table1.urdf", fixed_base=True)
            set_color(self.region_t2, C_Region)
            self.region_t3 = load_pybullet("../scenario_description/region_table1.urdf", fixed_base=True)
            set_color(self.region_t3, C_Region)

            # on shelf
            self.region_s1 = load_pybullet("../scenario_description/region_shelf.urdf", fixed_base=True)
            set_color(self.region_s1, C_Region)
            self.region_s2 = load_pybullet("../scenario_description/region_shelf.urdf", fixed_base=True)
            set_color(self.region_s2, C_Region)

            # in drawer
            self.region_w1 = load_pybullet("../scenario_description/region_drawer.urdf", fixed_base=True)
            set_color(self.region_w1, C_Region)

            self.regions = [self.region_t1, self.region_t2, self.region_t3,
                            self.region_s1, self.region_s2,
                            self.region_w1]

            self.region_p = get_area_p(self.regions)

            """set position for fixed objects"""

            set_pose(self.cabinet_shelf, Pose(Point(x=-0.45, y=-0.8, z=stable_z(self.cabinet_shelf, self.floor))))
            set_pose(self.drawer_shelf, Pose(Point(x=-0.45, y=0.8, z=stable_z(self.drawer_shelf, self.floor))))
            set_pose(self.pegboard, Pose(Point(x=-0.60, y=0, z=stable_z(self.pegboard, self.floor))))

            sx = -0.42
            sy = 0.8
            set_pose(self.region_s1, Pose(Point(x=sx, y=sy, z=stable_z(self.region_s1, self.drawer_links[1]))))
            set_pose(self.region_s2, Pose(Point(x=sx, y=sy, z=0.522)))

            set_pose(self.region_t1, Pose(Point(x=0.49, y=0.2, z=stable_z(self.region_t1, self.floor))))
            set_pose(self.region_t2, Pose(Point(x=-0.27, y=0.38, z=stable_z(self.region_t1, self.floor))))
            set_pose(self.region_t3, Pose(Point(x=-0.27, y=-0.38, z=stable_z(self.region_t1, self.floor))))
            set_pose(self.region_w1, Pose(Point(x=-0.17, y=0.8, z=0.042)))

            # set_pose(box1, Pose(Point(x=0.25, y=0.80, z=stable_z(box1, region2))))

        self.dic_body_info = {}

        self.env_bodies = [self.arm_base, self.floor, self.cabinet_shelf, self.drawer_shelf, self.pegboard]

        self.robot_bodies = [self.arm_left]

        self.movable_bodies = []
        self.all_bodies = []

        # self.reset()

    def sample_region(self):
        idx = np.random.choice(len(self.regions), 1, p=self.region_p)[0]
        # return idx
        return self.regions[idx]

    def reset(self):
        for b in self.movable_bodies:
            remove_body(b)
        self.movable_bodies = []
        self.all_bodies = []

        with HideOutput():
            """set position for movable joints"""
            # initial_jts = np.array([0.8, 0.75, 0.4, -1.8, 0.8, -1.5, 0])
            initial_jts = np.array([0.1, 1.4, 1, 1.7, 0, 0, 0])
            # initial_jts = np.array([0, 0, 0, 0, 0, 0, 0])

            config_left = BodyConf(self.arm_left, initial_jts)
            config_left.assign()

            movable_drawers = get_movable_joints(self.drawer_shelf)
            set_joint_positions(self.drawer_shelf, movable_drawers, [0., 0.25])

            movable_door = get_movable_joints(self.cabinet_shelf)
            set_joint_positions(self.cabinet_shelf, movable_door, [-0.])

            """set position for movable objects"""

            obj1 = create_box(get_rand(), get_rand(), get_rand(), mass=0.5, color=(0.859, 0.192, 0.306, 1.0))
            obj2 = create_box(get_rand(), get_rand(), get_rand(), mass=0.5, color=(0.271, 0.706, 0.490, 1.0))
            obj3 = create_box(get_rand(), get_rand(), get_rand(), mass=0.5, color=(0.647, 0.498, 0.894, 1.0))

            self.movable_bodies = [obj1, obj2, obj3]

            self.all_bodies = list(set(self.movable_bodies) | set(self.env_bodies) | set(self.regions))
            region = self.sample_region()
            # region = self.region_t2

            list_remove = place_objects(self.movable_bodies, region, self.all_bodies)
            for b in list_remove:
                self.movable_bodies.remove(b)
                self.all_bodies.remove(b)

            # self.movable_bodies = [obj1]
            # set_pose(obj1, Pose(Point(x=0.50, y=0.4, z=stable_z(obj1,self.region_t1))))

            for b in self.movable_bodies:
                obj_center, obj_extent = get_center_extent(b)
                body_pose = get_pose(b)
                body_frame = tform_from_pose(body_pose)
                bottom_center = copy(obj_center)
                bottom_center[2] = bottom_center[2] - obj_extent[2] / 2
                bottom_frame = tform_from_pose((bottom_center, body_pose[1]))
                relative_frame_bottom = np.dot(bottom_frame, np.linalg.inv(body_frame))  # from pose to bottom
                center_frame = tform_from_pose((obj_center, body_pose[1]))
                relative_frame_center = np.dot(center_frame, np.linalg.inv(body_frame))

                self.dic_body_info[b] = (obj_extent, relative_frame_bottom, relative_frame_center)

            for b in self.movable_bodies + self.regions + self.env_bodies:
                set_collision_group_mask(b, int('0011', 2), int('0011', 2))
            for b in self.robot_bodies:
                set_collision_group_mask(b, int('0001', 2), int('0001', 2))

            set_camera(160, -35, 2.1, Point())

    def get_elemetns(self):
        self.reset()
        return self.arm_left, self.movable_bodies, self.regions


#######################################################


def gather_training_data():
    visualization = 0
    connect(use_gui=visualization)

    scn = PlanningScenario()
    robot = scn.arm_left

    tdata_workspace = []
    tdata_collision = []

    file_reach = 'reach.pk'
    file_direction = 'direction.pk'

    if os.path.exists(file_reach):
        with open(file_reach, 'rb') as f:
            tdata_workspace = pk.load(f)
    if os.path.exists(file_direction):
        with open(file_direction, 'rb') as f:
            tdata_collision = pk.load(f)

    print('existing_record, nn = {}/{}, cnn = {}/{}'.format(len([y for x, y in tdata_workspace if y]), len(tdata_workspace),
                                                            len([y for x, y in tdata_collision if y]),
                                                            len(tdata_collision)))

    for ep in range(10000):
        scn.reset()
        f_sample_grasp = sdg_sample_grasp(robot, scn.dic_body_info)
        f_ik_grasp = sdg_ik_grasp(robot, scn.all_bodies)

        for body in scn.movable_bodies:
            body_info = scn.dic_body_info[body]

            ellipsoid_frame, obj_extent, list_dist, list_dir_jj, list_z_jj = get_ellipsoid_frame(body, body_info,
                                                                                                 robot)
            if visualization:
                draw_shouldercenter_frame(robot, 3)
                draw_frame(ellipsoid_frame, 3)

            mat_image = None
            direction = 0
            for dist, fdir_jj, z_jj in zip(list_dist, list_dir_jj, list_z_jj):
                grasp_dir = GraspDirection(body, direction)
                grasp = f_sample_grasp.search((body, grasp_dir))[0]
                body_pose = BodyPose(body)
                approach_conf, command, q_approach, q_grasp = f_ik_grasp.search((body, body_pose, grasp))
                label = q_grasp is not None  # if the object is reachable
                tdata_workspace.append(
                    ((dist, fdir_jj, z_jj),
                     label))
                if q_grasp:
                    if mat_image is None:
                        mat_image = get_raytest_scatter3(body, ellipsoid_frame, obj_extent, robot)
                        if visualization:
                            plt.imshow(mat_image, 'gray')
                            plt.show()
                    label = command is not None  # if no collision exists
                    if visualization and command is not None:
                        draw_ee_frame(robot, 3, True)
                    tdata_collision.append(
                        ((direction, obj_extent[0], obj_extent[1], obj_extent[2], mat_image),
                         label))
                direction += 1

        with open(file_reach, 'wb') as f:
            pk.dump(tdata_workspace, f)
        with open(file_direction, 'wb') as f:
            pk.dump(tdata_collision, f)

        now = datetime.now()
        str_now = '{}'.format(now.strftime("%H:%M:%S"))
        print(
            'ep {}, {}    reach = {}/{}, direction = {}/{}'.format(ep, str_now, len([y for x, y in tdata_workspace if y]),
                                                                   len(tdata_workspace),
                                                                   len([y for x, y in tdata_collision if y]),
                                                                   len(tdata_collision)))

    disconnect()
    print('Finished.')


if __name__ == '__main__':
    # display_scenario()

    gather_training_data()
    # show_robot()
