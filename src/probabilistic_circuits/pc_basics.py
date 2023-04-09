import itertools
import numpy as np
import collections
from functools import reduce
from operator import add, mul
from typing import Type, Callable
from probabilistic_circuits.pc_nodes import PCNode, PCInnerNode, PCSum, PCProduct, PCLeaf, OffsetLeaf


def apply(pc: PCNode, func: Callable[[PCNode], None], node_type: Type[PCNode] = PCNode):
    """Applies a function on all nodes of a specific type in the given circuit."""
    queue = collections.deque([pc])
    while queue:
        node = queue.popleft()
        if isinstance(node, node_type):
            func(node)


def get_nodes_by_type(pc: PCNode, node_type: Type[PCNode] = PCNode):
    """Returns all nodes of a specific type of circuit (including replicates)."""
    nodes = []
    def get_nodes(node: PCNode):
        nodes.append(node)
    apply(pc, get_nodes, node_type=node_type)
    return nodes


def update_scope(pc: PCNode):
    """Updates the scopes of all inner nodes in the given circuit (inplace)."""
    if isinstance(pc, PCInnerNode):
        pc.scope = set.union(*[update_scope(child) for child in pc.children])
        return pc.scope
    return pc.scope


def check_validity(pc: PCNode):
    """
    Checks the validity of the given circuit. Iterates over inner nodes and checks if their class variables have the
    correct type and that they do not have any children which are None. For product nodes decomposability is checked.
    This method does NOT check the smoothness for sum nodes.
    """
    def _check_validity(node: PCNode):
        if isinstance(node, PCProduct):
            assert (len(node.children) > 0)
            assert (isinstance(node.children, list))
            for j, child in enumerate(node.children):
                if child is None:
                    raise Exception("{} has a child which is None (node number {} of {} in total)!".format(node, j, len(node.children)))
                if not child.scope.issubset(node.scope):
                    raise Exception("{} is not decomposable! Scope of child {} is not a subset of its parents scope!".format(node, child))
                if len(node.scope) - len(child.scope) != len(node.scope - child.scope):
                    raise Exception("{} is not decomposable! Children have overlapping scopes!".format(node))

        elif isinstance(node, PCSum):
            assert (np.isclose(sum(node.weights), 1))
            assert (isinstance(node.children, list))
            assert (isinstance(node.weights, list))
            assert (len(node.children) == len(node.weights))
            assert (len(node.children) > 0)
            for j, child in enumerate(node.children):
                if child is None:
                    raise Exception("{} has a child which is None (node number {} of {} in total)!".format(node, j, len(node.children)))
        else:
            raise Exception("Unknown node: " + str(node))

    apply(pc, _check_validity, node_type=PCInnerNode)
