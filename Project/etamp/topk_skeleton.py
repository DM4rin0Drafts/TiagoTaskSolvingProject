import os
import shutil
import pickle
from datetime import datetime
from collections import defaultdict, namedtuple

from .pddlstream.language.object import Object, OptimisticObject, get_hash
from .pddlstream.utils import read, INF
# from .pddlstream.algorithms.downward import parse_lisp
from .pddlstream.language.constants import pAction, pA_info, pAtom
from .pddlstream.language.conversion import pddl2obj

from copy import deepcopy, copy

import itertools
# from symk.main import run_symk
from .stream import is_active_arg, StreamResult
from utils.pybullet_tools.utils import HideOutput

CONSTRAIN_STREAMS = True
CONSTRAIN_PLANS = True  # False  True
MAX_DEPTH = INF  # 1 | INF

PDDL_DIR = 'pddl'

PREDICATE_ORDER = "_applicable"
PARA_PROPO = '_p'
TYPE_PROPO = 'propo_action'

PREDICATE_UNUSED = "_unused"
PARA_UNUSED = '_s'
TYPE_UNUSED = 'propo_stream'

NT_Domain = namedtuple('NT_Domain',
                       ['name', 'requirements', 'types', 'type_to_constants',
                        'predicates', 'predicate_to_typeList', 'derived', 'action', 'functions'])
NT_Problem = namedtuple('NT_Problem',
                        ['name', 'domain', 'type_to_objects', 'init', 'goal', 'metric'])
NT_Action = namedtuple('NT_Action', ['name', 'parameters', 'precondition', 'effect'])
NT_Stream = namedtuple('NT_Stream',
                       ['name', 'inputs', 'domain', 'outputs', 'certified', 'input_type_list', 'output_type_list'])


# EXE_Action = namedtuple('EXE_Action', ['name', 'parameters', 'add_effects'])  # objects
# EXE_Stream = namedtuple('EXE_Stream', ['name', 'inputs', 'outputs'])  # objects

class EXE_Action(object):
    def __init__(self, name, parameters, add_effects):
        self.name = name
        self.parameters = parameters
        self.add_effects = add_effects

    def __iter__(self):
        for o in [self.name, self.parameters, self.add_effects]:
            yield o

    def __repr__(self):
        return 'A-{}: {}'.format(self.name, self.parameters)

    def __eq__(self, other):
        if not isinstance(other, EXE_Action):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return self.name == other.name and self.parameters == other.parameters and self.add_effects == other.add_effects

    def __hash__(self):
        return get_hash(self.name + str(self.parameters) + str(self.add_effects))


class EXE_Stream(object):
    def __init__(self, name, inputs, outputs):
        self.name = name
        self.inputs = inputs
        self.outputs = outputs

    def __iter__(self):
        for o in [self.name, self.inputs, self.outputs]:
            yield o

    def __repr__(self):
        return 'S-{}: {} -> {}'.format(self.name, self.inputs, self.outputs)

    def __eq__(self, other):
        if not isinstance(other, EXE_Stream):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return self.name == other.name and self.inputs == other.inputs and self.outputs == other.outputs

    def __hash__(self):
        return get_hash(self.name + str(self.inputs) + str(self.outputs))


def remap_action_args(action, mapping, all_para=False):
    def mapping_fn(o):
        if (is_active_arg(o) or all_para) and o in mapping:
            return mapping[o]
        else:
            return o

    # mapping_fn = lambda o: mapping[o]
    name, optms_args, add_effects = action.name, action.parameters, action.add_effects
    new_args = tuple(map(mapping_fn, optms_args))

    new_add_effects = []
    for patom in add_effects:
        new_patom = pAtom(patom.name, tuple(map(mapping_fn, patom.args)))
        new_add_effects.append(new_patom)

    return EXE_Action(name, new_args, new_add_effects)


def remap_stream_args(stream, mapping, all_para=False):
    def mapping_fn(o):
        if (is_active_arg(o) or all_para) and o in mapping:
            return mapping[o]
        else:
            return o

    new_inputs = tuple(map(mapping_fn, stream.inputs))
    new_outputs = tuple(map(mapping_fn, stream.outputs))

    return EXE_Stream(stream.name, new_inputs, new_outputs)


def get_original_name(propo_name):
    return propo_name[propo_name.index('_') + 1:]


def create_pAtom(atom):
    return pAtom(atom.predicate, [pddl2obj(a) for a in atom.args])


def get_pAtom_exe(patom):
    return pAtom(patom.name, tuple([o.get_EXE() for o in patom.args]))


def substitute_alist(alist, mapping):
    if isinstance(alist, list):
        return [substitute_alist(a, mapping) for a in alist]
    else:
        return mapping.get(alist, alist)


def alist_to_str(alist):
    if isinstance(alist, list):
        return '(' + ' '.join([alist_to_str(a) for a in alist]) + ')'
    else:
        return str(alist)


def remove_types(alist):
    new_list = []
    for i, a in enumerate(alist):
        if alist[i] == '-' or (i > 0 and alist[i - 1] == '-'):
            continue
        new_list.append(a)
    return new_list


