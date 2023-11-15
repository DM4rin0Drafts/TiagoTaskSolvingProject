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
    set_color, remove_body, pose_from_tform, load_model, sample_placement, pairwise_collision

from copy import copy
from utils.pybullet_tools.body_utils import *

from Tiago.tiago_primitives import BodyPose, sdg_sample_place, sdg_sample_grasp, sdg_ik_grasp, sdg_motion_base_joint,\
    GraspDirection, sdg_plan_free_motion, sdg_sample_grasp_dir, sdg_sample_base_position, BodyConf, BodyInfo,\
    set_orientation_to_object
from Tiago.tiago_utils import open_arm, close_arm, set_group_conf, get_initial_conf

import pickle as pk
import os

C_Red = [0.875, 0.200, 0.216, 1.0]
C_Wood = [0.827, 0.675, 0.463, 1.0]
C_LightWood = [0.992, 0.871, 0.682, 1.0]
C_Region = [0.890, 0.855, 0.824, 1.0]


def get_rand(low=0.05, high=0.10):
    return np.random.uniform(low, high)

class PlanningScenario(object):
    def __init__(self):
        with HideOutput():

            """import fixed objects"""
            self.floor = load_pybullet("../scenario_description/short_floor.urdf", fixed_base=True)
            self.table = load_pybullet('models/table_collision/table.urdf', fixed_base=True)

            """ TIAGO ROBOT INIZIALIZATION """
            self.tiago = load_pybullet("../Tiago/tiago_description/tiago.urdf",
                                       position=[0, -.8, .02],
                                       fixed_base=False)


            initial_conf = get_initial_conf()

            # Configure Arm Position and Torso Position in the beginning of the simulation
            set_group_conf(self.tiago, 'arm', initial_conf)
            close_arm(self.tiago)
            open_arm(self.tiago)
            set_group_conf(self.tiago, 'torso', [0.20])

        self.dic_body_info = {}

        self.env_bodies = [self.floor]

        self.movable_bodies = []
        self.all_bodies = []

        # self.reset()


    def reset(self):
        for b in self.movable_bodies:
            remove_body(b)
        self.movable_bodies = []
        self.all_bodies = []

        with HideOutput():
            """set position for movable joints"""

            pose = sample_placement(self.tiago, self.floor)
            obstacles = list(set(self.all_bodies))
            counter = 0
            while ((pose is None) or any(pairwise_collision(robot, b) for b in obstacles)) and counter < 50:
                counter += 1
                pose = sample_placement(self.tiago, self.floor)

            """set position for movable objects"""

            obj1 = create_box(get_rand(), get_rand(), get_rand(), mass=0.5, color=(0.859, 0.192, 0.306, 1.0))
            obj2 = create_box(get_rand(), get_rand(), get_rand(), mass=0.5, color=(0.271, 0.706, 0.490, 1.0))
            obj3 = create_box(get_rand(), get_rand(), get_rand(), mass=0.5, color=(0.647, 0.498, 0.894, 1.0))

            self.movable_bodies = [obj1, obj2, obj3]

            self.all_bodies = list(set(self.movable_bodies) | set(self.env_bodies))

            list_remove = place_objects(self.movable_bodies, self.table, self.all_bodies)
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

            for b in self.movable_bodies + self.env_bodies:
                set_collision_group_mask(b, int('0011', 2), int('0011', 2))

            set_camera(160, -35, 2.1, Point())

    def get_elemetns(self):
        self.reset()
        return self.movable_bodies


#######################################################


def gather_training_data():
    visualization = 0
    connect(use_gui=visualization)

    scn = PlanningScenario()
    robot = scn.tiago
    tdata_workspace = []

    file_reach = 'training_data/reach_tiago.pk'

    if os.path.exists(file_reach):
        with open(file_reach, 'rb') as f:
            tdata_workspace = pk.load(f)

    print('existing_record, nn = {}/{}'.format(len([y for x, y in tdata_workspace if y]), len(tdata_workspace)))

    for ep in range(10000):
        scn.reset()

        f_sample_grasp = sdg_sample_grasp(robot, scn.dic_body_info)
        f_ik_grasp = sdg_ik_grasp(robot, scn.all_bodies)
        pose = get_pose(robot)

        for body in scn.movable_bodies:
            set_orientation_to_object(pose, robot, body)
            body_info = scn.dic_body_info[body]
            f_sample_grasp.set_info(body_info)
            body_pose = BodyPose(body)

            ellipsoid_frame, _, list_dist, list_dir_jj, list_z_jj = get_ellipsoid_frame(body, body_info, robot)
            if visualization:
                draw_shouldercenter_frame(robot, 3)
                draw_frame(ellipsoid_frame, 3)

            direction = 0
            for dist, fdir_jj, z_jj in zip(list_dist, list_dir_jj, list_z_jj):
                if dist > 1:
                    direction += 1
                    continue
                grasp_dir = GraspDirection(body, direction)
                grasp = f_sample_grasp.search((body, grasp_dir))[0]
                _, _, _, q_grasp = f_ik_grasp.search((body, body_pose, grasp))
                label = q_grasp is not None  # if the object is reachable
                if label:
                    tdata_workspace.append(((direction, dist, fdir_jj, z_jj), label))
                direction += 1

        with open(file_reach, 'wb') as f:
            pk.dump(tdata_workspace, f)

        now = datetime.now()
        str_now = '{}'.format(now.strftime("%H:%M:%S"))
        print(
            'ep {}, {}    reach = {}/{}'.format(ep, str_now, len([y for x, y in tdata_workspace if y]),
                                                                   len(tdata_workspace)))
        if len([y for x, y in tdata_workspace if y])/len(tdata_workspace) > 0.49:
            break

    disconnect()
    print('Finished.')


if __name__ == '__main__':

    gather_training_data()
