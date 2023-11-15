#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import time

from utils.pybullet_tools.darias_primitives import BodyConf
from utils.pybullet_tools.utils import step_simulation, WorldSaver, connect, set_pose, \
    Pose, \
    Point, set_default_camera, stable_z, SINK_URDF, STOVE_URDF, load_model, \
    disconnect, TABLE_URDF, get_bodies, HideOutput, create_box, load_pybullet, Euler, get_movable_joints, \
    set_joint_positions


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


class PlanningScenario(object):
    def __init__(self):
        with HideOutput():
            self.robot = load_pybullet("./tiago_description/tiago_single.urdf",
                                       fixed_base=True)


            #self.floor = load_model('/utils/models/short_floor.urdf', fixed_base=True)



            self.all_joints = get_movable_joints(self.robot)
            # self.left_joints = get_movable_joints(self.arm_left)

            self.reset()

    def reset(self):
        with HideOutput():
            # initial_jts = np.array([-0.3, -0.9, -1, -1.5, 0, 0, 0])
            #
            #
            # set_joint_positions(self.arm_right, self.right_joints, initial_jts)
            # set_joint_positions(self.arm_left, self.left_joints, (-1) * initial_jts)
            #
            # set_pose(self.celery, Pose(Point(x=0, y=0.4, z=stable_z(self.celery, self.floor))))
            # set_pose(self.radish, Pose(Point(x=0.1, y=0.6, z=stable_z(self.radish, self.floor)), Euler(yaw=0.5)))
            # set_pose(self.cabbage, Pose(Point(x=0, y=0.8, z=stable_z(self.cabbage, self.floor))))

            set_default_camera()

    def get_elemetns(self):
        self.reset()
        return


#######################################################

def display_scenario():
    connect(use_gui=True)
    
    scn = PlanningScenario()
    scn.get_elemetns()

    for i in range(10000):
        step_simulation()
        time.sleep(0.5)

    disconnect()
    print('Finished.')


if __name__ == '__main__':
    display_scenario()
