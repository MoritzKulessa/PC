import numpy as np
from collections import ChainMap
from probabilistic_circuits import pc_prune
from probabilistic_circuits.pc_nodes import PCNode, PCSum, PCProduct, PCLeaf


def inference(pc: PCNode, instance: dict) -> float:
    """Computes the probability/density for a given instance using the given circuit."""
    def _inference(node: PCNode, cur_scope: set):
        if len(cur_scope) == 0:
            return 1.0
        if isinstance(node, PCProduct):
            return np.prod([_inference(child, cur_scope.intersection(child.scope)) for child in node.children])
        elif isinstance(node, PCSum):
            return np.sum([weight * _inference(child, cur_scope.intersection(child.scope)) for weight, child in zip(node.weights, node.children) if cur_scope.issubset(child.scope)])
        elif isinstance(node, PCLeaf):
            return node.inference(instance)
        else:
            raise Exception("Unknown node: {}".format(node))
    return _inference(pc, set(instance.keys()))


def sample(pc: PCNode, evidence: dict = None, n: int = 1) -> list[dict[object, object]] | None:
    """
    Generates a list of n samples for the given circuit.
    If evidence is given, the circuit will be conditioned on the evidence before the samples are generated.
    This method will return None if no samples can be generated for the given evidence.
    """
    def _sample(node: PCNode) -> dict[object, object]:
        if isinstance(node, PCProduct):
            return dict(ChainMap(*[_sample(child) for child in node.children]))
        elif isinstance(node, PCSum):
            return _sample(node.children[np.random.choice(np.arange(len(node.children)), p=node.weights)])
        elif isinstance(node, PCLeaf):
            return node.sample()
        else:
            raise Exception("Unknown node: {}".format(node))

    cond_prob, sample_structure = pc_prune.condition(pc, evidence, remove_conditioned_nodes=False)
    if cond_prob is None:
        return None
    return [_sample(sample_structure) for _ in range(n)]
