from copy import deepcopy, copy
import numpy as np
import scipy.spatial.distance as spdist
from sklearn.metrics import pairwise_distances_argmin_min
from .topk_skeleton import EXE_Action, EXE_Stream, remap_action_args

# Continuous move: p1: seed_lower_bound, p2: seed_upper_bound
# Discrete move: p1: available_moves, p2: covariance matrix
# DecisionInfo = namedtuple('DecisionInfo', ['discrete', 'p1', 'p2', 'sampler', 'list_constraint'])
pose_generation_streams = ['sample-place', 'sample-stack']


class DecisionInfo(object):
    def __init__(self, op_name, step, discrete, p1, p2, sampler, roles_as_culprit=None):
        self.op_name = op_name
        self.step = step
        self.discrete = discrete
        self.p1 = p1
        self.p2 = p2
        self.sampler = sampler
        if roles_as_culprit is None:
            self.roles_as_culprit = []

    def add_culprit_role(self, ctype, decision_id):
        role = (ctype, decision_id)
        if role not in self.roles_as_culprit:
            self.roles_as_culprit.append(role)

    def __repr__(self):
        return f'{self.step}_{self.op_name}_c={len(self.roles_as_culprit)}'


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


class SamplerContinuous(object):
    lower = 0
    upper = 1

    def __init__(self, weights, exporation_std, number_sample=10):
        self.weights = np.array(weights)
        self.exporation_std = np.array(exporation_std)
        self.dim = self.exporation_std.shape[0]
        self.number_sample = number_sample

    def __call__(self, existing_decisions, list_constraint=None):
        if list_constraint is None:
            list_constraint = []

        for c in list_constraint:
            ctype, decision_code = c
            constraints = Constraint.ctype_to_constraints[ctype]
            for cst in constraints:
                pass

        candidates = [np.random.uniform(0, 1, self.dim)
                      for i in range(self.number_sample * self.dim)]
        candidates = np.array(candidates)

        if len(existing_decisions) == 0:
            return candidates[0]

        existing_decisions = np.array(existing_decisions)

        _, list_min_dist = pairwise_distances_argmin_min(candidates, existing_decisions)

        best_idx = np.argmax(list_min_dist)

        return candidates[best_idx]


class SamplerDiscrete(object):

    def __init__(self, all_choice, list_p):
        self.all_choice = all_choice
        self.list_p = np.array(list_p)
        self.list_p = self.list_p / sum(list_p)

        self.num_decisions = len(list_p)

    def __call__(self, existing_decisions, list_constraint=None):
        if list_constraint is None:
            list_constraint = []
        candidates = [c for c in self.all_choice if c not in existing_decisions]
        assert candidates
        list_p = [self.list_p[self.all_choice.index(c)] for c in candidates]
        list_p = np.array(list_p)
        list_p = list_p / sum(list_p)
        choice = np.random.choice(candidates, p=list_p)
        return choice
