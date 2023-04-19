import numpy as np
from collections import ChainMap
from probabilistic_circuits.pc_nodes import PCNode, PCSum, PCProduct, PCLeaf


def inference(pc: PCNode, instance: dict) -> float:
    """Computes the probability/density for a given instance using the given circuit."""

    def _inference(node: PCNode, cur_scope: set):
        if len(cur_scope) == 0:
            return 1.0
        if isinstance(node, PCProduct):
            return np.prod([_inference(child, cur_scope.intersection(child.scope)) for child in node.children])
        elif isinstance(node, PCSum):
            return np.sum([weight * _inference(child, cur_scope.intersection(child.scope))
                           for weight, child in zip(node.weights, node.children) if cur_scope.issubset(child.scope)])
        elif isinstance(node, PCLeaf):
            return node.inference(instance)
        else:
            raise Exception("Unknown node: {}".format(node))

    return _inference(pc, set(instance.keys()))


def sample(pc: PCNode, n: int = 1) -> list[dict[object, object]]:
    """Returns a list of n samples generated on the given circuit."""

    def _sample(node: PCNode) -> dict[object, object]:
        if isinstance(node, PCProduct):
            return dict(ChainMap(*[_sample(child) for child in node.children]))
        elif isinstance(node, PCSum):
            return _sample(node.children[np.random.choice(np.arange(len(node.children)), p=node.weights)])
        elif isinstance(node, PCLeaf):
            return node.sample()
        else:
            raise Exception("Unknown node: {}".format(node))

    return [_sample(pc) for _ in range(n)]


def mpe(pc: PCNode, randomized: bool = False) -> dict[object, object]:
    """
    Computes the most probable estimate.
    If randomized is set to True and two mpe's compete with the same probability then a random mpe is selected
    If randomized is set to False and two mpe's compete with the same probability then always the first one is selected.
    """

    def _maximize(node: PCNode, mem: dict[PCNode, any]):
        if isinstance(node, PCProduct):
            return np.prod([_maximize(child, mem) for child in node.children])
        elif isinstance(node, PCSum):
            probabilities = [weight * _maximize(child, mem) for weight, child in zip(node.weights, node.children)]
            if randomized:
                indices = np.argwhere(probabilities == np.amax(probabilities)).flatten()
                index = np.random.choice(indices)
            else:
                index = np.argmax(probabilities)
            mem[node] = index
            return probabilities[index]
        elif isinstance(node, PCLeaf):
            return node.max_inference()
        else:
            raise Exception("Unknown node: {}".format(node))

    def _obtain_assignments(node: PCNode, mem: dict[PCNode, any]):
        if isinstance(node, PCProduct):
            return dict(ChainMap(*[_obtain_assignments(child, mem) for child in node.children]))
        elif isinstance(node, PCSum):
            return _obtain_assignments(node.children[mem[node]], mem)
        elif isinstance(node, PCLeaf):
            return node.mpe()
        else:
            raise Exception("Unknown node: {}".format(node))

    max_dict = {}
    _maximize(pc, max_dict)
    return _obtain_assignments(pc, max_dict)
