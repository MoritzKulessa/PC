from functools import reduce
from operator import add, mul
from probabilistic_circuits import pc_basics
from probabilistic_circuits.pc_nodes import PCNode, PCInnerNode, PCSum, PCProduct, PCLeaf


def get_n_populations(pc: PCNode) -> int:
    """Returns the number of populations."""
    if isinstance(pc, PCLeaf):
        return 1
    counts = [get_n_populations(child) for child in pc.children]
    if isinstance(pc, PCSum):
        return reduce(add, counts)
    elif isinstance(pc, PCProduct):
        return reduce(mul, counts)
    else:
        raise Exception("Unknown node: " + str(pc))


def get_branching_factor(pc: PCNode) -> float:
    """Returns the mean branching factor of the inner nodes."""
    n_children = []

    def _count_splits(node: PCNode) -> None:
        n_children.append(len(node.children))
    pc_basics.apply(pc, _count_splits, node_type=PCInnerNode)
    return sum(n_children)/len(n_children)


def get_number_of_edges(pc: PCNode) -> int:
    """Returns the number of edges"""
    return sum([len(c.children) for c in pc_basics.get_nodes(pc, PCInnerNode)])


def get_depth(pc: PCNode) -> int:
    """Returns the maximum depth."""
    node_depth = {}

    def _count_layers(node):
        depth = node_depth.setdefault(node, 1)
        if hasattr(node, "children"):
            for c in node.children:
                node_depth.setdefault(c, depth + 1)

    pc_basics.apply(pc, _count_layers)
    return max(node_depth.values())


def get_stats(pc: PCNode) -> dict[str, float | int]:
    """Returns a dictionary with statistics about the circuit."""
    nodes = pc_basics.get_nodes(pc, node_type=PCNode)
    leaves = pc_basics.get_nodes(pc, node_type=PCLeaf)
    n_populations = get_n_populations(pc)
    return {
        "n_populations": n_populations,
        "n_nodes": len(nodes),
        "n_nodes_distinct": len(set(nodes)),
        "n_leaves": len(leaves),
        "n_leaves_distinct": len(set(leaves)),
        "n_sums": len([node for node in nodes if isinstance(node, PCSum)]),
        "n_products": len([node for node in nodes if isinstance(node, PCProduct)]),
        "depth": get_depth(pc),
        "edges": get_number_of_edges(pc),
        "branching_factor": get_branching_factor(pc),
        "n_scope": len(pc.scope)
    }


def get_stats_string(pc: PCNode) -> str:
    """Returns a string with statistics about the circuit."""
    nodes_stats = get_stats(pc)
    s = "#nodes: " + str(nodes_stats["n_nodes"]) + "\t(distinct: " + str(nodes_stats["n_nodes_distinct"]) + ")"
    s += "\t#leaves: " + str(nodes_stats["n_leaves"]) + "\t(distinct: " + str(nodes_stats["n_leaves_distinct"]) + ")"
    s += "\t#sums: " + str(nodes_stats["n_sums"])
    s += "\t#products: " + str(nodes_stats["n_products"])
    s += "\t#depth: " + str(nodes_stats["depth"])
    s += "\t#edges: " + str(nodes_stats["edges"])
    s += "\tbranching factor: " + str(round(nodes_stats["branching_factor"], 2))
    s += "\t#scope: " + str(nodes_stats["n_scope"])
    s += "\t#populations: " + str(nodes_stats["n_populations"])
    return s
