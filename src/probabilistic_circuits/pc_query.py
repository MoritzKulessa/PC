import itertools
import numpy as np
from collections import Counter, ChainMap
from probabilistic_circuits import pc_prune
from probabilistic_circuits.pc_nodes import PCSum, PCProduct, PCLeaf

import logging
logger = logging.getLogger(__name__)


def probability(pc, inst):
    def _prob(node, cur_scope):
        if isinstance(node, PCProduct):
            results = [_prob(child, cur_scope.intersection(child.scope)) for child in node.children]
            if None in results:
                return None
            return np.prod(results)
        elif isinstance(node, PCSum):
            if len(cur_scope) > 0:
                results = [(weight, _prob(child, cur_scope.intersection(child.scope))) for weight, child in zip(node.weights, node.children) if cur_scope.issubset(child.scope)]
                results = [weight * prob for weight, prob in results if prob is not None]
                if len(results) == 0:
                    return None
                return np.sum(results)
            else:
                # marginalize
                return 1.0
        elif isinstance(node, PCLeaf):
            if len(cur_scope) > 0 and cur_scope.issubset(node.scope):
                assert (len(node.scope) == 1)  # Currently multivariate leaves are not supported
                return node.prob(inst)
            else:
                # marginalize
                return 1.0
        else:
            raise Exception("Unknown node: {}".format(node))

    return _prob(pc, set(inst.keys()))


def sample(pc, inst=None, n_samples=1):
    def _sample(node):
        if isinstance(node, PCProduct):
            return dict(ChainMap(*[_sample(child) for child in node.children]))
        elif isinstance(node, PCSum):
            return _sample(node.children[np.random.choice(np.arange(len(node.children)), p=node.weights)])
        elif isinstance(node, PCLeaf):
            return node.sample()
        else:
            raise Exception("Unknown node: {}".format(node))

    cond_prob, sample_structure = pc_prune.condition(pc, inst, remove_cond_nodes=False)
    if cond_prob is None:
        return None
    return [_sample(sample_structure) for _ in range(n_samples)]


def relate(pc, inst):
    def _relate(node):
        if isinstance(node, PCProduct):
            return sum([_relate(child) for child in node.children], Counter())
        elif isinstance(node, PCSum):
            results = [Counter({k: weight * v for k, v in _relate(child).items()}) for weight, child in zip(node.weights, node.children)]
            return sum(results, Counter())
        elif isinstance(node, PCLeaf):
            return Counter(node.relate())
        else:
            raise Exception("Unknown node: {}".format(node))

    # Could also be computed with remove nodes but this requires to incorporate cond_prob into the result
    cond_prob, relate_structure = pc_prune.condition(pc, inst, remove_cond_nodes=False)
    if cond_prob is None:
        return {}
    return dict(_relate(relate_structure))


def mpe(pc, inst=None):
    def _maximize(node, mem):
        if isinstance(node, PCProduct):
            return np.prod([_maximize(child, mem) for child in node.children])
        elif isinstance(node, PCSum):
            probabilities = [weight * _maximize(child, mem) for weight, child in zip(node.weights, node.children)]
            index = np.argmax(probabilities)
            mem[node] = index
            return probabilities[index]
        elif isinstance(node, PCLeaf):
            return node.max_prob()
        else:
            raise Exception("Unknown node: {}".format(node))

    def _obtain_assignments(node, mem):
        if isinstance(node, PCProduct):
            return dict(ChainMap(*[_obtain_assignments(child, mem) for child in node.children]))
        elif isinstance(node, PCSum):
            return _obtain_assignments(node.children[mem[node]], mem)
        elif isinstance(node, PCLeaf):
            return node.max_value()
        else:
            raise Exception("Unknown node: {}".format(node))

    cond_prob, mpe_structure = pc_prune.condition(pc, inst, remove_cond_nodes=False)
    if cond_prob is None:
        return None

    max_dict = {}
    _maximize(mpe_structure, max_dict)
    return _obtain_assignments(mpe_structure, max_dict)


def mpe_all_assignments(pc, inst=None):
    def _maximize(node, mem):
        if isinstance(node, PCProduct):
            return np.prod([_maximize(child, mem) for child in node.children])
        elif isinstance(node, PCSum):
            probabilities = [weight * _maximize(child, mem) for weight, child in zip(node.weights, node.children)]
            idxs = np.argwhere(probabilities == np.amax(probabilities)).flatten()
            mem[node] = idxs
            return probabilities[idxs[0]]
        elif isinstance(node, PCLeaf):
            return node.max_prob()
        else:
            raise Exception("Unknown node: {}".format(node))

    def _obtain_assignments(node, mem):
        if isinstance(node, PCProduct):
            results = [_obtain_assignments(child, mem) for child in node.children]
            idxs = [list(range(len(result))) for result in results]
            new_results = []
            for combo in itertools.product(*idxs):
                combo_result = dict(ChainMap(*[results[i][index] for i, index in enumerate(combo)]))
                new_results.append(combo_result)
            return new_results
        elif isinstance(node, PCSum):
            results = []
            for index in mem[node]:
                results += _obtain_assignments(node.children[index], mem)
            return results
        elif isinstance(node, PCLeaf):
            return [node.max_value()]
        else:
            raise Exception("Unknown node: {}".format(node))

    cond_prob, mpe_structure = pc_prune.condition(pc, inst, remove_cond_nodes=False)
    if cond_prob is None:
        return None

    max_dict = {}
    _maximize(mpe_structure, max_dict)
    mpe_instances = []
    for assignment in _obtain_assignments(mpe_structure, max_dict):
        tmp = list(assignment.items())
        indices = [list(range(len(values))) for _, values in tmp]
        mpe_instances += [{tmp[i][0]: tmp[i][1][index] for i, index in enumerate(combo)} for combo in itertools.product(*indices)]
    return mpe_instances
