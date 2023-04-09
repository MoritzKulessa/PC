import itertools
import numpy as np
import collections
from functools import reduce
from operator import add, mul
from probabilistic_circuits.pc_nodes import PCSum, PCProduct, PCLeaf,OffsetLeaf


def apply(pc, func, include_leaves=True, only_leaves=False):
    queue = collections.deque([pc])
    while queue:
        node = queue.popleft()
        if isinstance(node, PCLeaf):
            if include_leaves:
                func(node)
        else:
            if not only_leaves:
                func(node)
            for c in node.children:
                queue.append(c)


def check_validity(pc):
    def _check_validity(node):
        # check decomposable
        if isinstance(node, PCProduct):
            assert (len(node.children) > 0)
            assert (isinstance(node.children, list))
            scope = set(node.scope)
            tmp_scope = set(node.scope)
            for j, c in enumerate(node.children):
                if c is None:
                    raise Exception("{} has a child which is None (node number {} of {} in total)!".format(node, j, len(node.children)))

                c_scope = set(c.scope)
                if not c_scope.issubset(scope):
                    raise Exception(str(node) + " is not consistent! Scope of child " + str(c) + " not the a subset of its parents scope!")
                tmp_len = len(tmp_scope)
                tmp_scope -= c_scope
                if tmp_len-len(c_scope) != len(tmp_scope):
                    s = "Product node scope: " + str(node.scope) + "\n"
                    for i, c1 in enumerate(node.children):
                        s += "Child " + str(i) + " scope: " + str(c1.scope) + "\n"
                    raise Exception(str(node) + " is not consistent! Children have overlapping scopes!\n" + s)

        # check children are not None
        elif isinstance(node, PCSum):
            assert(np.isclose(sum(node.weights), 1))
            assert (isinstance(node.children, list))
            assert (isinstance(node.weights, list))
            assert(len(node.children) == len(node.weights))
            assert(len(node.children) > 0)

            for j, c in enumerate(node.children):
                if c is None:
                    raise Exception("{} has a child which is None (node number {} of {} in total)!".format(node, j, len(node.children)))
        else:
            raise Exception("Unknown node: " + str(node))

    apply(pc, _check_validity, include_leaves=False)


def get_nodes_by_type(pc, node_type=None):
    # Returns nodes (including replicates)
    nodes = []

    def get_nodes(node):
        if node_type is None:
            nodes.append(node)
        else:
            if isinstance(node, node_type):
                nodes.append(node)

    if node_type is not None and node_type == PCLeaf:
        apply(pc, get_nodes, include_leaves=True, only_leaves=True)
    elif (node_type is not None and (node_type == PCSum or node_type == PCProduct)) or (node_type is not None and (node_type == (PCSum, PCProduct) or node_type == (PCProduct, PCSum))):
        apply(pc, get_nodes, include_leaves=False, only_leaves=False)
    else:
        apply(pc, get_nodes, include_leaves=True, only_leaves=False)
    return nodes


def update_scope(node):
    if isinstance(node, PCLeaf):
        return node.scope
    node.scope = set.union(*[update_scope(child) for child in node.children])
    return node.scope


def get_scope_leaf_dict(leaves, use_ids=False):
    scope_leaves = {}
    for leaf_id in range(len(leaves)):
        if len(leaves[leaf_id].scope) == 0:
            continue
        assert (len(leaves[leaf_id].scope) == 1)
        scope, = leaves[leaf_id].scope
        if scope not in scope_leaves:
            scope_leaves[scope] = []
        if use_ids:
            scope_leaves[scope].append(leaf_id)
        else:
            scope_leaves[scope].append(leaves[leaf_id])
    return scope_leaves


def get_sub_circuits(node, min_population=0.0):

    if isinstance(node, PCLeaf):
        if isinstance(node, OffsetLeaf):
            return [[1.0, set()]]
        return [[1.0, {node}]]

    elif isinstance(node, PCSum):
        collected_subs = []
        for i, child in enumerate(node.children):
            weight = node.weights[i]
            retrieved_subs = get_sub_circuits(child, min_population=min_population)
            for [p, dists] in retrieved_subs:
                new_prob = weight * p
                if new_prob > min_population:
                    collected_subs.append([new_prob, dists])
        return collected_subs

    elif isinstance(node, PCProduct):
        results = [get_sub_circuits(child, min_population=min_population) for child in node.children]
        collected_subs = []
        for combo in list(itertools.product(*results)):
            new_prob = 1.0
            new_dists = set()
            for [p, dists] in combo:
                new_prob *= p
                new_dists.update(dists)
            if new_prob > min_population:
                collected_subs.append([new_prob, new_dists])
        return collected_subs

    else:
        raise Exception("Invalid node: " + str(node))


def get_n_sub_circuits(node):
    if isinstance(node, PCLeaf):
        return 1
    counts = [get_n_sub_circuits(child) for child in node.children]
    if isinstance(node, PCSum):
        return reduce(add, counts)
    elif isinstance(node, PCProduct):
        return reduce(mul, counts)
    else:
        raise Exception("Invalid node: " + str(node))


def get_branching_factor(pc):
    n_children = []

    def _count_splits(node):
        n_children.append(len(node.children))
    apply(pc, _count_splits, include_leaves=False, only_leaves=False)
    return np.mean(n_children)


def get_number_of_edges(node):
    return sum([len(c.children) for c in get_nodes_by_type(node, (PCSum, PCProduct))])


def get_depth(pc):
    node_depth = {}

    def count_layers(node):
        ndepth = node_depth.setdefault(node, 1)

        if hasattr(node, "children"):
            for c in node.children:
                node_depth.setdefault(c, ndepth + 1)

    apply(pc, count_layers)
    return max(node_depth.values())


def get_node_statistic_dict(pc, name=None):
    nodes = get_nodes_by_type(pc)
    leaves = get_nodes_by_type(pc, node_type=PCLeaf)
    n_sps = get_n_sub_circuits(pc)
    return {
        "name": name,
        "sub_populations": n_sps,
        "n_nodes": len(nodes),
        "n_nodes_distinct": len(set(nodes)),
        "n_leaves": len(leaves),
        "n_leaves_distinct": len(set(leaves)),
        "n_sums": len([node for node in nodes if isinstance(node, PCSum)]),
        "n_products": len([node for node in nodes if isinstance(node, PCProduct)]),
        "depth": get_depth(pc),
        "edges": get_number_of_edges(pc),
        "branching_factor": round(get_branching_factor(pc), 2),
        "n_scope": len(pc.scope)
    }


def get_node_statistics_string(spn):
    nodes_stats = get_node_statistic_dict(spn)
    s = "#nodes: " + str(nodes_stats["n_nodes"]) + "\t(distinct: " + str(nodes_stats["n_nodes_distinct"]) + ")"
    s += "\t#leaves: " + str(nodes_stats["n_leaves"]) + "\t(distinct: " + str(nodes_stats["n_leaves_distinct"]) + ")"
    s += "\t#sums: " + str(nodes_stats["n_sums"])
    s += "\t#products: " + str(nodes_stats["n_products"])
    s += "\t#depth: " + str(nodes_stats["depth"])
    s += "\t#edges: " + str(nodes_stats["edges"])
    s += "\tbranching factor: " + str(nodes_stats["branching_factor"])
    return s
