import numpy as np
import pybullet as p
from utils.pybullet_tools.utils import create_box, body_collision, get_joint_position


def random_generator(path, static_objects):
	rand_obj = np.random.randint(5, 10)
	colors = [(1, 0, 0, 1), (0, 0, 1, 1), (0, 0, 1, 1), (0, 0, 1, 1), (0, 0, 1, 1), (0, 0, 1, 1), (0, 0, 1, 1), (0, 0, 1, 1), (0, 0, 1, 1), (0, 0, 1, 1)]

	bd_body = {}
	for i in range(rand_obj):
		color = colors[i]
		bd_body["box" + str(i + 1)] = create_box(.07, .07, .1, mass=1, color=color)
		pos = load_random_box_position()
		ori = load_random_orientation()
		p.resetBasePositionAndOrientation(bd_body["box" + str(i + 1)], pos, ori)

		for k in bd_body:
			if bd_body[k] == bd_body["box" + str(i + 1)]:
				continue
			while body_collision(bd_body["box" + str(i + 1)], bd_body[k]) !=0:
				pos = load_random_box_position()
				ori = load_random_orientation()
				p.resetBasePositionAndOrientation(bd_body["box" + str(i + 1)], pos, ori)

		for k in static_objects:
			obj = static_objects[k]
			while body_collision(obj, bd_body["box" + str(i + 1)]) != 0:
				if k == 'table':
					pos = load_random_box_position()
					ori = load_random_orientation()
				else:
					pos[2]= pos[2] + 0.01
				p.resetBasePositionAndOrientation(bd_body["box" + str(i + 1)], pos, ori)

	in_save = input("Konfiguration speichern? (y, n)")

	if in_save == 'y':
		input_file = [rand_obj]
		for i in range(rand_obj):
			input_file = input_file + list(colors[i])

		positions, orientations= (), ()
		for k in bd_body:
			pos, ori = p.getBasePositionAndOrientation(bd_body[k], physicsClientId=0)
			positions = list(positions) + list(pos)
			orientations = list(orientations) + list(ori)

		input_file = input_file + positions + orientations
		input_file = " ".join([str(i) for i in input_file])

		f = open(path, "a")
		f.write(input_file + "\n")
		f.close()



def random_generator2(path):
	rand_obj = np.random.randint(5, 10)
	colors = [(1, 0, 0, 1), (0, 0, 1, 1), (0, 0, 1, 1), (0, 0, 1, 1), (0, 0, 1, 1), (0, 0, 1, 1), (0, 0, 1, 1), (0, 0, 1, 1), (0, 0, 1, 1), (0, 0, 1, 1)]

	bd_body = {}
	for i in range(rand_obj):
		color = colors[i]
		bd_body["box" + str(i + 1)] = create_box(.07, .07, .1, mass=1, color=color)
		pos = load_random_box_position()
		ori = load_random_orientation()
		p.resetBasePositionAndOrientation(bd_body["box" + str(i + 1)], pos, ori)

		for k in bd_body:
			if bd_body[k] == bd_body["box" + str(i + 1)]:
				continue
			while body_collision(bd_body["box" + str(i + 1)], bd_body[k]) !=0:
				pos = load_random_box_position()
				ori = load_random_orientation()
				p.resetBasePositionAndOrientation(bd_body["box" + str(i + 1)], pos, ori)


	in_save = input("Konfiguration speichern? (y, n)")

	if in_save == 'y':
		input_file = [rand_obj]
		for i in range(rand_obj):
			input_file = input_file + list(colors[i])		#

		positions, orientations= (), ()
		for k in bd_body:
			pos, ori = p.getBasePositionAndOrientation(bd_body[k], physicsClientId=0)
			positions = list(positions) + list(pos)
			orientations = list(orientations) + list(ori)

		input_file = input_file + positions + orientations
		input_file = " ".join([str(i) for i in input_file])

		f = open(path, "a")
		f.write(input_file + "\n")
		f.close()


def load_random_orientation():
	w = np.random.uniform(0, np.pi / 2)
	ori = [0, 0, w]
	return p.getQuaternionFromEuler(ori)


def load_random_box_position():
	table_config = [1.5, 1, 0.58]
	x = np.random.uniform(-table_config[0] / 2 + 0.1, table_config[0] / 2 - 0.1)
	y = np.random.uniform(-table_config[1] / 2 + 0.1, table_config[1] / 2 - 0.1)
	z = table_config[2] + 0.1 / 2

	while (x > 0.45 and y > 0.3):
		x = np.random.uniform(-table_config[0] / 2 + 0.1, table_config[0] / 2 - 0.1)
		y = np.random.uniform(-table_config[1] / 2 + 0.1, table_config[1] / 2 - 0.1)

	return [x, y, z]



def transform2list(input_string):
	num_bodies = int(input_string[0])

	c = list(map(float, input_string[1:4*num_bodies+1]))
	p = list(map(float, input_string[4*num_bodies+1:4*num_bodies+3*num_bodies+1]))
	o = list(map(float, input_string[4*num_bodies+3*num_bodies+1:]))

	colors, positions, orientations = [], [], []
	for i in range(num_bodies):
		colors.append(tuple(c[i*4:i*4+4]))
		positions.append(tuple(p[i*3:i*3+3]))
		orientations.append(tuple(o[i*4:i*4+4]))

	return colors, positions, orientations