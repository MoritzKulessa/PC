import numpy as np
from probabilistic_circuits.pc_nodes import PCSum, PCProduct, PCLeaf, offset_leaf
from probabilistic_circuits import pc_basics, pc_prune


def rebuild_structured_decomposable(pc, scope_order=None):
    def _rebuild_recursive(circuits, cur_order):

        if len(cur_order) == 0:
            children = circuits[0][1]
            if len(children) == 0:
                return offset_leaf
            else:
                return PCProduct(children=children)

        splits = {}
        for weight, pop in circuits:
            assigned_child = None
            for leaf in pop:
                if cur_order[0] in leaf.scope:
                    assigned_child = leaf
                    break
            if assigned_child not in splits:
                splits[assigned_child] = []
            if assigned_child is not None:
                pop.discard(assigned_child)
            splits[assigned_child].append([weight, pop])

        children = []
        weights = []
        for leaf, new_pops in splits.items():
            weights.append(np.sum([weight for weight, _ in new_pops]))
            if leaf is None:
                children.append(_rebuild_recursive(new_pops, cur_order[1:]))
            else:
                children.append(PCProduct(children=[leaf, _rebuild_recursive(new_pops, cur_order[1:])]))
        return PCSum(children=children, weights=list(np.array(weights)/np.sum(weights)))

    if scope_order is None:
        leaves = pc_basics.get_nodes_by_type(pc, node_type=PCLeaf)
        scope_order = [scope for scope, _ in sorted(pc_basics.get_scope_leaf_dict(leaves).items(), key=lambda x: len(x[1]), reverse=True)]
    sub_cs = pc_basics.get_sub_circuits(pc)
    new_pc = _rebuild_recursive(sub_cs, scope_order)
    pc_basics.update_scope(new_pc)
    new_pc = pc_prune.contract(new_pc)
    return new_pc
