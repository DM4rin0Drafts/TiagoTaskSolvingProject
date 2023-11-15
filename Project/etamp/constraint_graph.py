from .pddlstream.language.object import Object, OptimisticObject, EXE_Object, EXE_OptimisticObject, get_hash
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash as ghash
from .topk_skeleton import EXE_Action, EXE_Stream, remap_action_args
from networkx.algorithms import isomorphism
from networkx.drawing.nx_pydot import graphviz_layout
from copy import copy, deepcopy
import numpy as np


def update_constraint_dict(new_c):
    existing_constraints = Constraint.ctype_to_constraints[new_c.ctype]
    if existing_constraints:
        """ctype exists"""
        Constraint.ctype_to_constraints[new_c.ctype].append(new_c)
        # same_decision_exist = False
        # for c in existing_constraints:
        #     if list(new_c.decision_vertices) == list(c.decision_vertices):
        #         assert np.sign(new_c.yg) == np.sign(c.yg)
        #         same_decision_exist = True
        # if not same_decision_exist:
        #     Constraint.ctype_to_constraints[new_c.ctype].append(new_c)
    else:
        assert new_c.yg >= 0
        Constraint.ctype_to_constraints[new_c.ctype].append(new_c)


def get_op_to_decision(env, node):
    op_to_decision = {}
    cur_node = node
    while True:
        if cur_node.is_root:
            break
        if cur_node.parent.is_decision_node:
            _, ops = env.get_steps_ops(cur_node.parent.depth)
            try:
                decision = tuple(cur_node.decision)
            except:
                decision = (cur_node.decision,)
            op_to_decision[ops[0]] = decision
        if cur_node.is_root:
            break
        else:
            cur_node = cur_node.parent

    return op_to_decision


def rename_stream(env, op_to_decision):
    new_op_to_decision = {}
    for v, value in op_to_decision.items():
        assert isinstance(v, EXE_Stream)
        arg_mapping = {}
        for p in v.inputs:
            if isinstance(p, EXE_Object):
                if isinstance(p.value, int):
                    arg_mapping[p] = EXE_Object(p.pddl, env.bd_body[p.value])
        if arg_mapping:
            new_op = EXE_Stream(v.name, tuple([arg_mapping.get(a, a) for a in v.inputs]), v.outputs)
        else:
            new_op = v
        new_op_to_decision[new_op] = value

    return new_op_to_decision


class Constraint(object):
    """
    A constraint based on directional graph
    """
    ctype_to_constraints = defaultdict(list)

    def __init__(self, raw_digraph, node, env, yg):

        # self.digraph = nx.DiGraph()
        # self.digraph.add_nodes_from(digraph.nodes)
        # self.digraph.add_edges_from(digraph.edges)
        assert isinstance(raw_digraph, nx.DiGraph)
        self.skeleton_id = env.skeleton_id
        self.digraph = raw_digraph.copy()
        self.yg = yg

        op_to_decision = get_op_to_decision(env, node)
        """Find the victim(stream)"""
        self.victim = (None, None)  # (step,op)
        for v in self.digraph:
            successors = list(self.digraph.successors(v))
            if not successors:
                self.victim = (self.digraph.nodes[v]['step'], v)
                break
        """Find culprits, and append their decision vertices"""
        self.culprits = {}  # a victim can itself be a culprit.
        self.decision_vertices = []
        vertices = list(self.digraph)
        for v in vertices:
            if isinstance(v, EXE_Stream):
                op_info = env.get_op_info(v)
                if op_info.free_generator:
                    decision = op_to_decision[v]
                    self.decision_vertices.append(decision)
                    self.culprits[v] = (self.digraph.nodes[v]['step'], decision)
                    self.digraph.add_edge(decision, v, weight=10)
        """Rename the vertices"""
        vertices = list(self.digraph)
        for v in vertices:
            if isinstance(v, EXE_Stream):
                arg_mapping = {}
                for p in v.inputs:
                    if isinstance(p, EXE_Object):
                        if isinstance(p.value, int):
                            arg_mapping[p] = EXE_Object(p.pddl, env.bd_body[p.value])
                if arg_mapping:
                    new_stream = EXE_Stream(v.name, tuple([arg_mapping.get(a, a) for a in v.inputs]), v.outputs)
                    mapping = {v: new_stream}
                    self.digraph = nx.relabel_nodes(self.digraph, mapping)
                    if v == self.victim[1]:
                        self.victim = (self.victim[0], new_stream)
                    if v in self.culprits:
                        self.culprits[new_stream] = self.culprits[v]
                        del self.culprits[v]
            if isinstance(v, EXE_Object):
                if isinstance(v.value, int):
                    mapping = {v: EXE_Object(v.pddl, env.bd_body[v.value])}
                    self.digraph = nx.relabel_nodes(self.digraph, mapping)
        """Extract decisions and conditions related to the current constraint"""
        self.dict_decision = {}  # decision_id to op
        self.dict_condition = {}  # condition_id to op
        vertices = list(self.digraph)
        for v in vertices:
            if v in self.culprits:
                predecessors = list(self.digraph.predecessors(v))
                for p in predecessors:
                    if isinstance(p, tuple):
                        shortest_path = nx.shortest_path(self.digraph, source=v, target=self.victim[1])
                        decision_id = v.name + '_' + str(len(shortest_path) - 1)
                        self.dict_decision[decision_id] = p
                        self.culprits[v] = (*self.culprits[v], decision_id)
            if isinstance(v, EXE_Object):
                if isinstance(v.value, str):
                    predecessors = list(self.digraph.predecessors(v))
                    for p in predecessors:
                        if isinstance(p, tuple):
                            shortest_path = nx.shortest_path(self.digraph, source=v, target=self.victim[1])
                            self.dict_condition[v.value + '_' + str(len(shortest_path) - 1)] = p

        description = str(self.victim[1].name)
        for p in self.victim[1].inputs:
            if isinstance(p, EXE_Object):
                if isinstance(p.value, str):
                    description += p.value  # body_name
        dlist = list(self.dict_condition)
        dlist.sort()
        for s in dlist:
            description += str(s)
        dlist = list(self.dict_decision)
        dlist.sort()
        for s in dlist:
            description += str(s)

        self.ctype = get_hash(description + ghash(self.digraph))

        assert self.ctype != 77569319

        update_constraint_dict(self)

        return

    @property
    def nodes(self):
        return list(self.digraph)

    @property
    def edges(self):
        return list(self.digraph.edges)

    def show(self):
        list_color = []
        lable_map = {}

        plt.rcParams["figure.figsize"] = (16, 8)
        plt.clf()

        for v in self.digraph:

            if v == self.victim[1]:
                list_color.append('coral')
                lable_map[v] = str(v)
            elif isinstance(v, EXE_Object) or isinstance(v, EXE_OptimisticObject):
                list_color.append('yellowgreen')
                lable_map[v] = str(v)
            elif isinstance(v, tuple):
                if v in self.decision_vertices:
                    list_color.append('cornflowerblue')
                else:
                    list_color.append('lightblue')
                lable_map[v] = '(' + ','.join([f'{i:.5f}' for i in v]) + ')'
            elif v in self.culprits:
                list_color.append('mediumorchid')
                lable_map[v] = str(v)
            else:
                list_color.append('yellowgreen')  # cornflowerblue
                lable_map[v] = str(v)

        dot_pos = graphviz_layout(self.digraph, prog="twopi")  # twopi sfdp circo
        # dot_pos = nx.spring_layout(self.digraph, scale=2)
        nx.draw(self.digraph, dot_pos, node_color=list_color, with_labels=False)
        nx.draw_networkx_labels(self.digraph, dot_pos, lable_map)
        plt.show()

        # plt.draw()
        # plt.pause(30)

    def __repr__(self):
        return f'{len(self.dict_condition)}_{len(self.dict_decision)}_{self.victim[1].name}{self.victim[1].inputs}_' + (
            f'{self.yg:.3f}' if self.yg > 0 else 'f')

    def get_copy(self):
        return deepcopy(self)


if __name__ == '__main__':

    edges = [(('o_collision',), ('inv',)),
             (('o_target',), ('inv',)),
             (('p_target',), ('inv',)),
             (('g_target',), ('inv',)),
             (('o_collision',), ('o_target',)),
             (('p_collision',), ('o_collision',)),
             (('p_target',), ('o_target',)),
             (('o_target',), ('sample-grasp',)),
             (('p_target',), ('sample-grasp',)),
             (('dir_target',), ('sample-grasp',)),
             (('sample-grasp',), ('g_target',)),
             (('o_target',), ('sample-grasp-dir',)),
             (('p_target',), ('sample-grasp-dir',)),
             (('sample-grasp-dir',), ('dir_target',)),
             ]
    # edges = [(('a',), ('b',)),
    #          (('b',), ('c',)),
    #          (('c',), ('a',)),
    #          ]

    graph1 = nx.DiGraph()
    for edge in edges:
        graph1.add_edge(*edge, weight=10)

    nx.set_node_attributes(graph1, ('o_collision',), 'tag')

    label_map = {}
    for node_id in graph1:
        label = ''
        for i in node_id:
            label += str(i)

        label_map[node_id] = label

    v = ('o_collision',)
    nx.set_node_attributes(graph1, v, 'collision')

    edges = [(('o_drawer',), ('inv',)),
             (('p_red', 0), ('inv',)),
             (('o_red',), ('inv',)),
             (('g_red',), ('inv',)),
             (('o_drawer',), ('o_red',)),
             (('o_drawer',), ('o_red',)),
             (('p_drawer', 0), ('o_drawer',)),
             (('p_red', 0), ('o_red',)),
             (('o_red',), ('sample-grasp',)),
             (('dir_red',), ('sample-grasp',)),
             (('p_red', 0), ('sample-grasp',)),
             (('sample-grasp',), ('g_red',)),
             (('o_red',), ('sample-grasp-dir',)),
             (('p_red', 0), ('sample-grasp-dir',)),
             (('p_red', 0), ('sample-grasp-dir',)),
             (('sample-grasp-dir',), ('dir_red',)),
             ]
    # edges = [(('c1',), ('a1',)),
    #          (('a1',), ('b1',)),
    #          (('b1',), ('c1',)),
    #          ]
    graph2 = nx.DiGraph()
    for edge in edges:
        graph2.add_edge(*edge, weight=1)

    """Hashes are identical for isomorphic graphs and strong guarantees 
    that non-isomorphic graphs will get different hashes. """
    hg1 = ghash(graph1)
    hg2 = ghash(graph2)

    nodes1 = set(graph1.nodes)
    nodes2 = set(graph2.nodes)

    # nodes1 = {'1', '2', '3', '4'}
    # nodes2 = {'1', '2', '3', '4'}

    graph_match = isomorphism.GraphMatcher(graph1, graph2)

    print(hg1 == hg2)
    print(graph_match.is_isomorphic())
    print(graph_match.mapping)

    # print(nodes1)
    # print(nodes2)
    # print(nodes1 == nodes2)

    #
    nx.draw(graph1, with_labels=True)
    plt.show()

    a = [1, 2, 3, 4]

    a.reverse()

    print(str(a[:2]))
