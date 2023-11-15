#!/usr/bin/env python


from __future__ import print_function
import numpy as np
import time
from numpy.core.fromnumeric import size
import pybullet as p

from copy import copy

from utils.pybullet_tools.darias_primitives import BodyConf
from utils.pybullet_tools.utils import LockRenderer, enable_gravity, step_simulation, WorldSaver, connect, set_pose, \
    Pose, Point, set_default_camera, stable_z, SINK_URDF, STOVE_URDF, load_model, \
    disconnect, TABLE_URDF, get_bodies, HideOutput, create_box, load_pybullet, Euler, get_movable_joints, \
    set_joint_positions, set_point, load_pybullet, step_simulation, Euler, get_links, get_link_info, get_movable_joints, set_joint_positions, \
    set_camera, get_center_extent, tform_from_pose, attach_viewcone, LockRenderer, load_model, set_point, get_pose, get_link_name

from Tiago.tiago_utils import open_arm, close_arm, set_group_conf, get_initial_conf



class BuildWorldScenario(object):
    def __init__(self):
        with HideOutput():
            with LockRenderer():
                #table = (länge = 1.5, breite = 1)
                self.pos_table = [0, 0, 0.58]
                self.table_config = [1.5, 1, 0.58]
                self.grasp_type = 'top'

                """ Load Table in the simulation"""
                self.table = load_model('models/table_collision/table.urdf', fixed_base=True)

                """ Load floor to simulation """
                self.floor = load_model('../Tiago/scenario_description/plane.urdf', fixed_base=True)


                """ TIAGO ROBOT INIZIALIZATION """
                startPosition = [0, -0.8, 0]
                startOrientation = p.getQuaternionFromEuler([0, 0, np.pi / 2])
                
                self.tiago = load_pybullet("../Tiago/tiago_description/tiago.urdf",
                                            position=startPosition, 
                                            fixed_base=True)

                self.setStartPositionAndOrienation(self.tiago, startPosition, startOrientation)

                initial_conf = get_initial_conf(self.grasp_type)

                #Configure Arm Position and Torso Position in the beginning of the simulation
                set_group_conf(self.tiago, 'arm', initial_conf)
                close_arm(self.tiago)                              
                open_arm(self.tiago)                                
                set_group_conf(self.tiago, 'torso', [0.35])          


                """ Load Boxes to Simulations """
                mass = 1        #in kg
                self.bd_body = {
                    "box1": create_box(.07, .07, .2, mass=mass, color=(0, 1, 0, 1)),
                }

                self.bd_body.update(dict((self.bd_body[k], k) for k in self.bd_body))
                self.setBoxPositionAndOrientation()

                enable_gravity()

        self.movable_bodies = [self.bd_body['box1']]
        self.env_bodies = [self.floor]
        self.regions = [self.table]

        self.all_bodies = list(set(self.movable_bodies) | set(self.env_bodies) | set(self.regions))     #all ids in model/body [0, 1, 2, ...]

        self.sensors = []
        self.robots = [self.tiago]

        self.gripper = None

        self.dic_body_info = {}

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

        set_camera(160, -35, 2.1, Point())
        self.saved_world = WorldSaver()


    def setBoxPositionAndOrientation(self):
        box1_pos = [-0.1, -0.1, self.pos_table[2] + 0.2 / 2]#self.load_random_box_position()
        self.setStartPositionAndOrienation(self.bd_body['box1'], box1_pos, self.load_start_orientation())


    def load_random_box_position(self):
        x = np.random.uniform(-self.table_config[0] / 2, self.table_config[0] / 2)
        y = np.random.uniform(-self.table_config[1] / 2, self.table_config[1] / 2)
        z = self.pos_table[2] + 0.1 / 2

        while(x > 0.45 and y > 0.3):
            x = np.random.uniform(-self.table_config[0] / 2 + 0.1, self.table_config[0] / 2 - 0.1)
            y = np.random.uniform(-self.table_config[1] / 2 + 0.1, self.table_config[1] / 2 - 0.1)

        return [x, y, z]
            

    def load_start_position(self):
        x = np.random.uniform(-5, 5) 
        y = np.random.uniform(-5, 5)

        if(x < 1.5 and x > 0.0):
            x = x + 1
        elif(x < 0.0 and x > -1.5):
            x = x - 1

        if(y < 1.5 and y > 0.0):
            y = y + 1
        elif(y < 0.0 and y > -1.5):
            y = y - 1

        #print("Position: {}, {}".format(x, y))
        return [x, y, 0]

    def load_start_orientation(self):
        w = np.random.uniform(0, 2 * np.pi)
        startOrientationRPY = [0, 0, w]

        #print("Orientation: {}".format(p.getQuaternionFromEuler(startOrientationRPY)))
        return p.getQuaternionFromEuler(startOrientationRPY)

    def setStartPositionAndOrienation(self, id, position, orientation):
        """
            ATTENTIONS: CALL THIS FUNCTION ONLY WHEN THE SIMULATION STARTS!!!!!!!!!
        """
        p.resetBasePositionAndOrientation(id, position, orientation)

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

    def get_elements(self):
        self.reset()
        return

    def save_world(self):
        pass

    def load_world(self):
        pass

    
def display_scenario():
    connect(use_gui=True)
    
    scn = BuildWorldScenario()
    scn.get_elements()

    for i in range(10000):
        step_simulation()
        time.sleep(0.5)

    disconnect()
    print('Finished.')
