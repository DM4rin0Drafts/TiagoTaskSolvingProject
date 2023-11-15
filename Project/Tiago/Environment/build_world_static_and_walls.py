#!/usr/bin/env python


from __future__ import print_function
import numpy as np
import time
from numpy.core.fromnumeric import size
import pybullet as p
import csv
import pickle
from Tiago.Environment.generator import transform2list
from copy import copy

from utils.pybullet_tools.darias_primitives import BodyConf
from utils.pybullet_tools.utils import LockRenderer, enable_gravity, step_simulation, WorldSaver, connect, set_pose, \
	Pose, Point, set_default_camera, stable_z, SINK_URDF, STOVE_URDF, load_model, \
	disconnect, TABLE_URDF, get_bodies, HideOutput, create_box, load_pybullet, Euler, get_movable_joints, \
	set_joint_positions, set_point, load_pybullet, step_simulation, Euler, get_links, get_link_info, \
	get_movable_joints, set_joint_positions, set_camera, get_center_extent, tform_from_pose, attach_viewcone, \
	LockRenderer, load_model, set_point, get_pose, get_link_name, set_collision_group_mask

from Tiago.tiago_utils import open_arm, close_arm, set_group_conf, get_initial_conf
import  json

class BuildWorldScenarioWalls(object):
	def __init__(self, path=None, fixed_object=False, random_config=None):
		with HideOutput():
			with LockRenderer():
				self.pos_table = [0, 0, 0.58]
				self.table_config = [1.5, 1, 0.58]

				""" Load Table in the simulation"""
				self.table = load_model('models/table_collision/table.urdf', fixed_base=True)

				""" Load floor to simulation """
				self.floor = load_model('models/short_floor.urdf', fixed_base=True)

				""" TIAGO ROBOT INIZIALIZATION """
				startPosition = self.load_start_position()
				#startOrientation = self.load_start_orientation()

				self.tiago = load_pybullet("../Tiago/tiago_description/tiago.urdf",
										   position=startPosition,
										   fixed_base=False)

				self.setStartPositionAndOrienation(self.tiago, [-100, -100, .02], (0, 0, 0, 0))

				initial_conf = get_initial_conf()

				# Configure Arm Position and Torso Position in the beginning of the simulation
				set_group_conf(self.tiago, 'arm', initial_conf)
				close_arm(self.tiago)
				open_arm(self.tiago)
				set_group_conf(self.tiago, 'torso', [0.20])

				mass = 1  # in kg


				self.walls = {
					"wall1": create_box(1.54, 0.25, 1.5, mass=10, color=(0.5, 0.5, 0.5, 1)),
					"wall2": create_box(1.04, 0.25, 1.5, mass=10, color=(0.5, 0.5, 0.5, 1))
				}

				positions = [(0, 0.645, 0.75), (-0.895, 0, 0.78)]
				orientations = [(0, 0, 0), (0, 0, np.pi / 2)]

				for body, pos, ori in zip(self.walls, positions, orientations):
					self.setStartPositionAndOrienation(self.walls[body], pos,
													   self.load_start_orientation(ori))
				self.static_object = {}
				if fixed_object == True:
					self.static_object = {
						"box1": create_box(.15, .15, .15, mass=2, color=(0.5, 0.5, 0.5, 1)),
						"box2": create_box(.2, .2, .2, mass=3, color=(0.5, 0.5, 0.5, 1)),
						"box3": create_box(.3, .3, .3, mass=5, color=(0.5, 0.5, 0.5, 1))
					}
					self.setBoxPositionAndOrientation()

				if path is not None:
					if random_config is None:
						num_lines = sum(1 for line in open(path)) - 1
						random_config = np.random.randint(0, num_lines)

					self.bd_body = {}
					with open(path) as file:
						for idx, line in enumerate(file):
							if idx == random_config:
								color, position, orientation = transform2list(list(line.rstrip('\n').split(" ")))
								for i, (c, pos, ori) in enumerate(zip(color, position, orientation)):
									self.bd_body["box" + str(i + 1)] = create_box(.07, .07, .1, mass=mass, color=c)
									self.setStartPositionAndOrienation(self.bd_body["box" + str(i + 1)], list(pos), ori)

				else:
					import Tiago.Environment.generator as gen
					self.static_object['underground'] = self.table
					#gen.random_generator('Environment/walls_environment_em.txt', self.static_object)
					gen.random_generator('Environment/walls_environment_hm.txt', self.static_object)

				if fixed_object == True:
					self.static_object.update(dict((self.static_object[k], k) for k in self.static_object))
				self.bd_body.update(dict((self.bd_body[k], k) for k in self.bd_body))
				self.walls.update(dict((self.walls[k], k) for k in self.walls))

				enable_gravity()

		self.movable_bodies = []
		for k in self.bd_body:
			if isinstance(k, str):
				self.movable_bodies.append(self.bd_body[k])

		if fixed_object == True:
			self.env_bodies = [self.floor]
			for k in self.static_object:
				if isinstance(k, str):
					self.env_bodies.append(self.static_object[k])

			for k in self.walls:
				if isinstance(k, str):
					self.env_bodies.append(self.walls[k])
		else:
			self.env_bodies = [self.floor]
			for k in self.walls:
				if isinstance(k, str):
					self.env_bodies.append(self.walls[k])

		self.regions = [self.table]

		self.all_bodies = list(
			set(self.movable_bodies) | set(self.env_bodies) | set(self.regions))  # all ids in model/body [0, 1, 2, ...]

		self.robots = [self.tiago]

		self.dic_body_info = { }

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
		for b in self.robots:
			set_collision_group_mask(b, int('0001', 2), int('0001', 2))

		set_camera(160, -35, 2.1, Point())
		self.saved_world = WorldSaver()


	def setBoxPositionAndOrientation(self):
		positions = [(0.4, 0.3, self.pos_table[2] + 0.2 / 2), (0.4, -0.2, self.pos_table[2] + 0.2 / 2), (-.4, -.15, self.pos_table[2] + 0.15)]
		orientations = [(0, 0, 0), (0, 0, np.pi / 4), (0, 0, np.pi / 8)]

		for body, pos, ori in zip(self.static_object, positions, orientations):
			self.setStartPositionAndOrienation(self.static_object[body], pos,
											   p.getQuaternionFromEuler(ori))


	def load_random_box_position(self):
		x = np.random.uniform(-self.table_config[0] / 2, self.table_config[0] / 2)
		y = np.random.uniform(-self.table_config[1] / 2, self.table_config[1] / 2)
		z = self.pos_table[2] + 0.1 / 2

		while (x > 0.45 and y > 0.3):
			x = np.random.uniform(-self.table_config[0] / 2 + 0.1, self.table_config[0] / 2 - 0.1)
			y = np.random.uniform(-self.table_config[1] / 2 + 0.1, self.table_config[1] / 2 - 0.1)

		return [x, y, z]


	def load_start_orientation(self, ori):
		w = [0, 0, np.random.uniform(0, 2 * np.pi)]
		return p.getQuaternionFromEuler(ori)


	def load_start_position(self):
		x = np.random.uniform(-5, 5)
		y = np.random.uniform(-5, 5)

		if (x < 1.5 and x > 0.0):
			x = x + 1
		elif (x < 0.0 and x > -1.5):
			x = x - 1

		if (y < 1.5 and y > 0.0):
			y = y + 1
		elif (y < 0.0 and y > -1.5):
			y = y - 1

		# print("Position: {}, {}".format(x, y))
		return [-100, -100, 0.02]


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
	import os
	#scn = BuildWorldScenarioStatic(os.path.dirname(os.path.abspath(__file__)) + "\\bd_box_static_config.txt", True)
	scn = BuildWorldScenarioWalls(None, False)
	scn.get_elements()

	for i in range(10000):
		step_simulation()
		time.sleep(0.5)

	disconnect()
	print('Finished.')
