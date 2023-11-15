from copy import deepcopy, copy
import numpy as np
import scipy.spatial.distance as spdist
from collections import namedtuple
from .stream import is_active_arg
from .pddlstream.language.object import Object, OptimisticObject, EXE_Object, EXE_OptimisticObject, get_hash
from sklearn.metrics import pairwise_distances_argmin_min
from .topk_skeleton import EXE_Action, EXE_Stream, remap_action_args
from utils.pybullet_tools.utils import get_center_extent
from .decision_sampler import DecisionInfo, SamplerContinuous, SamplerDiscrete
from collections import defaultdict
import networkx as nx

# Continuous move: p1: seed_lower_bound, p2: seed_upper_bound
# Discrete move: p1: available_moves, p2: covariance matrix
# DecisionInfo = namedtuple('DecisionInfo', ['discrete', 'p1', 'p2', 'sampler', 'list_constraint'])
pose_generation_streams = ['sample-place', 'sample-stack']


def remap_action_with_uncertainty(action, mapping):
    concrete_action = remap_action_args(action, mapping)

    name, optms_args, optms_add_effects = action
    _, concrete_args, concrete_add_effects = concrete_action

    new_mapping = copy(mapping)
    if name == 'place':
        new_mapping[optms_args[1]] = concrete_args[1]

    realistic_action = remap_action_args(action, new_mapping)

    return realistic_action, new_mapping


def dist_m(m0, m1, weights):
    weights = np.array(weights)
    m0 = np.array([m0]) * weights
    m1 = np.array([m1]) * weights

    dist = spdist.cdist(m0, m1, 'sqeuclidean')[0][0]

    return dist


class SkeletonEnv(object):

    def __init__(self, skeleton_id, op_plan, update_env_reward_fn, stream_info, scn):
        self.skeleton_id = skeleton_id
        self.update_env_reward_fn = update_env_reward_fn
        self.stream_info = stream_info
        self.op_plan = op_plan  # [StreamResult_1.1, StreamResult_1.2, pAction_1, StreamResult_2.1, pAction_2,...]
        self.decision_steps = []  # wrt. op_plan
        self.action_steps = []
        self.env_reset_fn = scn.reset

        self.bd_body = scn.bd_body

        self.failure_log = {}

        self.depth_to_decision_info = {}
        self.depth_to_steps = {}
        self.step_to_victim_roles = defaultdict(list)  # step -> ctypes
        self.ctype_to_constraint = {}  # ctype -> an example constraint defined in current skeleton

        self.digraph = None  # the last constraint encountered
        self.bodyid_to_obj = {}

        depth = 0
        steps = []
        for step, op in enumerate(op_plan):
            if isinstance(op, EXE_Stream):
                op_info = self.get_op_info(op)
                """decision node"""
                if op_info.free_generator:
                    if steps:
                        self.depth_to_steps[depth] = tuple(steps)
                        depth += 1
                    self.decision_steps.append(step)

                    if op_info.discrete:
                        sampler = SamplerDiscrete(op_info.p1, op_info.p2)
                    else:
                        sampler = SamplerContinuous(op_info.p1, op_info.p2)

                    self.depth_to_decision_info[depth] = DecisionInfo(op.name, step, op_info.discrete,
                                                                      op_info.p1, op_info.p2,
                                                                      sampler)
                    self.depth_to_steps[depth] = (step,)
                    depth += 1
                    steps = []
                else:
                    steps.append(step)

            elif isinstance(op, EXE_Action):
                self.action_steps.append(step)
                steps.append(step)

        self.depth_to_steps[depth] = steps
        self.step_to_depth = {}
        for d, s_tuple in self.depth_to_steps.items():
            for s in s_tuple:
                self.step_to_depth[s] = d

        if not self.decision_steps:
            raise Warning('Invalid op_plan!')

        self.num_decision = len(self.decision_steps)
        self.op_length = len(self.op_plan)
        self.num_depth = len(self.depth_to_steps)

        self.dict_decision_info = {}
        self.dict_transition_info = {}

    def get_op_info(self, op):
        return self.stream_info[op.name]

    @property
    def start_with_transition_node(self):
        """ Not Needed Probably"""
        if not self.decision_steps:
            return True
        if self.decision_steps[0] != 0:
            return True
        return False

    def get_steps_ops(self, depth):
        if depth == self.num_depth:
            return (), ()

        steps = self.depth_to_steps[depth]

        return steps, tuple([self.op_plan[s] for s in steps])

    def apply_decision(self, depth, mapping, decision):
        """
        """
        assert depth in self.depth_to_decision_info
        steps, ops = self.get_steps_ops(depth)
        add_mapping = self._apply_stream(ops[0], mapping, decision)
        if add_mapping is None:
            return None, steps[0]  # updated_mapping, termial_step
        else:
            return add_mapping, None

    def apply_transition(self, depth, mapping):
        assert depth not in self.depth_to_decision_info

        steps, ops = self.get_steps_ops(depth)

        sum_action_reward = 0
        net_add_mapping = {}
        total_mapping = {}
        total_mapping.update(mapping)
        for step, op in zip(steps, ops):
            op = self.op_plan[step]
            if isinstance(op, EXE_Stream):
                assert not self.get_op_info(op).free_generator

                add_mapping = self._apply_stream(op, total_mapping)
                if add_mapping is None:
                    return None, step, sum_action_reward
                net_add_mapping.update(add_mapping)
                total_mapping.update(add_mapping)
            elif isinstance(op, EXE_Action):
                action_reward = self._simulate_action(op, total_mapping)
                if action_reward is None:
                    return None, step, sum_action_reward
                sum_action_reward += action_reward

        return net_add_mapping, None, sum_action_reward

    def get_digraph(self):
        list_node = list(self.digraph)
        set_node = set(list_node)
        assert len(list_node) == len(set_node)

        return self.digraph

    def _extend_digraph(self, vertex, digraph, step):
        """

        """
        list_op = self.op_plan[:step]
        list_op.reverse()
        if isinstance(vertex, EXE_Stream):
            for i, p in enumerate(vertex.inputs):
                digraph.add_edge(p, vertex, weight=1)  # duplicate edges will be ignored
                if not (i == 0 and vertex.name in pose_generation_streams):
                    self._extend_digraph(p, digraph, step)
            if vertex in list(digraph):
                step = self.op_plan.index(vertex)
                digraph.nodes[vertex]['step'] = step
        elif isinstance(vertex, EXE_OptimisticObject):
            for op in list_op:
                if isinstance(op, EXE_Stream):
                    if vertex in op.outputs:
                        step = self.op_plan.index(op)
                        digraph.add_edge(op, vertex, weight=1)
                        digraph.nodes[op]['step'] = step
                        self._extend_digraph(op, digraph, step)
                        s = digraph.nodes[op]['step']
                        break
        elif isinstance(vertex, EXE_Object):
            if isinstance(vertex.value, int):
                # predecessors = list(self.digraph.predecessors(vertex))
                in_edges = list(digraph.in_edges(vertex, data=False))
                if not in_edges:
                    find_parent = False
                    for op in list_op:
                        if isinstance(op, EXE_Stream):
                            if op.name in pose_generation_streams:
                                if vertex.value == op.inputs[0].value:
                                    digraph.add_edge(op.outputs[0], vertex, weight=1)
                                    step = self.op_plan.index(op)
                                    digraph.add_edge(op, op.outputs[0], weight=10)
                                    digraph.nodes[op]['step'] = step
                                    self._extend_digraph(op, digraph, step)
                                    find_parent = True
                                    break
                    if not find_parent:
                        obj_center, obj_extent = get_center_extent(vertex.value)
                        digraph.add_edge(tuple(obj_center), vertex, weight=1)

    def build_digraph(self, stream, seed_gen_fn):
        """
        Build a directional graph to indicate a constraint responsible for the failure in a stream evaluation.
        """
        self.digraph = nx.DiGraph()
        self._extend_digraph(stream, self.digraph, self.op_plan.index(stream))
        self.tmp_error_msg =None

        try:
            self.tmp_error_msg = seed_gen_fn.get_error_message()
            _, msg_obstacle, _ = self.tmp_error_msg
            if msg_obstacle:
                collision_obj = EXE_Object('temp_o' + str(msg_obstacle), msg_obstacle)
                self.digraph.add_edge(collision_obj, stream, weight=10)
                if stream.name in pose_generation_streams:
                    self.digraph.add_edge(collision_obj, stream.inputs[0], weight=1)
                self._extend_digraph(collision_obj, self.digraph, self.op_plan.index(stream))
        except AttributeError:
            pass

    def _apply_stream(self, stream, mapping, seed=None):
        """
        :param stream: EXE_Stream
        :param seed: the move(output of a stream generator) suggested by node.
        """
        self.digraph = None
        self.yg = None

        def mapping_fn(o):
            if is_active_arg(o):
                t = []
                for i in mapping.items():
                    t.append(i)

                return mapping[o]
            else:
                return o

        new_inputs = tuple(map(mapping_fn, stream.inputs))
        input_tuple = tuple(obj.value for obj in new_inputs)

        seed_gen_fn = self.get_op_info(stream).seed_gen_fn
        output_tuple = seed_gen_fn(input_tuple=input_tuple, seed=seed)  # tuple, can be None

        if not output_tuple:
            self.build_digraph(stream, seed_gen_fn)
            try:
                _, _, yg = seed_gen_fn.get_error_message()
            except:
                yg = 1
            self.yg = yg

            if stream.name not in self.failure_log:
                self.failure_log[stream.name] = 1
            else:
                self.failure_log[stream.name] += 1
            return None

        old_objects = stream.outputs  # tuple
        new_objects = None
        if output_tuple:
            new_objects = tuple(EXE_Object(o.pddl, v) for o, v in zip(old_objects, output_tuple))

        add_mapping = {}
        for v, c in zip(old_objects, new_objects):
            add_mapping[v] = c

        return add_mapping

    def _simulate_action(self, action, mapping):
        """
        update environment by the action, and return the execution result
        """

        concrete_action = remap_action_args(action, mapping)

        list_action = [concrete_action]

        action_reward = self.update_env_reward_fn(list_action)

        """Apply action uncertainties for specific actions."""
        if action.name == 'pick' and action.parameters[0].value == 1 and 0:
            if np.random.uniform() > 0.5:
                action_reward = None

        return action_reward

    def set_env_depth(self, depth, mapping):
        """
        set the environment as described by mapping
        """
        if depth in self.depth_to_decision_info:
            return
        """when depth is for transition node"""
        steps, ops = self.get_steps_ops(depth)

        mapping = copy(mapping)
        for step, op in zip(steps, ops):
            if isinstance(op, EXE_Action):
                concrete_action = remap_action_args(op, mapping)
                self.update_env_reward_fn([concrete_action])

    @property
    def problematic_streams(self):
        """Return streams that are most likely to fail"""
        self.failure_log = {k: v for k, v in
                            sorted(self.failure_log.items(),
                                   key=lambda item: item[1])}
        result = [s for s in self.failure_log]
        result.reverse()
        return result

    def __repr__(self):
        return '{}'.format(self.op_plan)


if __name__ == '__main__':
    A = tuple([1, 2, 3, 4])
    B = [A, A, A, A, A]
    list_eff = [[atom for atom in a] for a in B]
    list_eff = sum(list_eff, [])
    print(list_eff)

    d = {'a': 1, 'b': 2, 'c': 3, 'd': 4}

    for k, v in d.items():
        print(k, v)

    for i in range(4, 20 + 1):
        print(i)
