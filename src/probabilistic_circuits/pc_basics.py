import itertools
import collections
import numpy as np
from typing import Type, Callable
from probabilistic_circuits.pc_nodes import PCNode, PCInnerNode, PCSum, PCProduct, PCLeaf, OffsetLeaf


def apply(pc: PCNode, func: Callable[[PCNode], None], node_type: Type[PCNode] = PCNode) -> None:
    """Applies a function on all nodes of a specific type in the given circuit."""
    queue = collections.deque([pc])
    while queue:
        node = queue.popleft()
        if isinstance(node, node_type):
            func(node)
        if isinstance(node, PCInnerNode):
            for child in node.children:
                queue.append(child)


def get_nodes(pc: PCNode, node_type: Type[PCNode] = PCNode) -> list[PCNode]:
    """Returns all nodes of a specific type of circuit (including replicates)."""
    nodes = []

    def _get_nodes(node: PCNode):
        nodes.append(node)

    apply(pc, _get_nodes, node_type=node_type)
    return nodes


def update_scope(pc: PCNode) -> set[object]:
    """Updates the scopes of all inner nodes in the given circuit (inplace)."""
    if isinstance(pc, PCInnerNode):
        pc.scope = set.union(*[update_scope(child) for child in pc.children])
        return pc.scope
    return pc.scope


def check_validity(pc: PCNode) -> None:
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
                assert (child is not None)
                assert (isinstance(child, PCNode))
                assert (child.scope.issubset(node.scope))
                assert (len(node.scope) - len(child.scope) == len(node.scope - child.scope))
            assert (node.scope == set.union(*[child.scope for child in node.children]))

        elif isinstance(node, PCSum):
            assert (np.isclose(sum(node.weights), 1))
            assert (isinstance(node.children, list))
            assert (isinstance(node.weights, list))
            assert (len(node.children) == len(node.weights))
            assert (len(node.children) > 0)
            for j, child in enumerate(node.children):
                assert (child is not None)
                assert (isinstance(child, PCNode))
            assert (node.scope == set.union(*[child.scope for child in node.children]))
        else:
            raise Exception("Unknown node: " + str(node))

    apply(pc, _check_validity, node_type=PCInnerNode)


def get_populations(pc: PCNode, min_population_size: float = 0.0) -> list[tuple[float, set[PCLeaf]]]:
    """Returns all populations and their size of the circuit which satisfy the specified minimum population size."""
    assert (0.0 <= min_population_size <= 1.0)

    def _get_populations_recursive(cur_node: PCNode, cur_population_size: float) -> list[tuple[float, set[PCLeaf]]]:
        if cur_population_size < min_population_size:
            return []

        if isinstance(cur_node, PCLeaf):
            if isinstance(cur_node, OffsetLeaf):
                return [(1.0, set())]
            return [(1.0, {cur_node})]

        if isinstance(cur_node, PCSum):
            collected_subs = []
            for i, child in enumerate(cur_node.children):
                weight = cur_node.weights[i]
                retrieved_subs = _get_populations_recursive(child, cur_population_size * weight)
                for probability, nodes in retrieved_subs:
                    new_probability = weight * probability
                    if min_population_size <= new_probability:
                        collected_subs.append((new_probability, nodes))
            return collected_subs

        if isinstance(cur_node, PCProduct):
            results = [_get_populations_recursive(child, cur_population_size) for child in cur_node.children]
            collected_subs = []
            for combo in list(itertools.product(*results)):
                new_probability = 1.0
                new_nodes = set()
                for [p, nodes] in combo:
                    new_probability *= p
                    new_nodes.update(nodes)
                if min_population_size <= new_probability:
                    collected_subs.append((new_probability, new_nodes))
            return collected_subs

        raise Exception("Invalid node: " + str(cur_node))

    return _get_populations_recursive(pc, 1.0)
