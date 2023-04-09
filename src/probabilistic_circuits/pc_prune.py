import numpy as np
from collections import Counter
from probabilistic_circuits.pc_nodes import PCSum, PCProduct, PCLeaf, OffsetLeaf, CategoricalLeaf
from probabilistic_circuits import pc_compare, pc_basics

import logging
logger = logging.getLogger(__name__)


#todo rebuild, straight foreward combine by selecting subset with most overlap
#todo Product node identification with counts ... all choices
#todo all them will not result in optimal structures, but counts may be interesting


def _get_node_prune_dict(pc):

    # trade inner nodes with multiple parents as leafs, these leaves can be pruned by itself
    # the question is, when is it worth to open the break the inner node... todo


    # First approach: Break inner node dependencies by transforming to a tree
    nodes = pc_basics.get_nodes_by_type(pc, node_type=(PCSum, PCProduct))
    rep_nodes = set([node for node, count in Counter(nodes).items() if count > 1])
    if len(rep_nodes) > 0:
        # Only implemented for trees
        raise Exception("Inner nodes have multiple parents: " + str(rep_nodes))

    node_dict = {}

    def _fill_dict(node):
        if isinstance(node, (PCSum, PCProduct)):
            results = sum([_fill_dict(child) for child in node.children], Counter())
            node_dict[node] = results
            return results
        elif isinstance(node, PCLeaf):
            return Counter({node: 1})
        else:
            raise Exception("Unknown node: {}".format(node))
    _fill_dict(pc)
    return node_dict


def _remove_nodes(node, remove_nodes):
    if node in remove_nodes:
        return None
    if isinstance(node, PCLeaf):
        return node
    i = 0
    while i < len(node.children):
        result = _remove_nodes(node.children[i], remove_nodes)
        if result is None:
            del node.children[i]
            if isinstance(node, PCSum):
                del node.weights[i]
                sum_weights = sum(node.weights)
                node.weights = [weight / sum_weights for weight in node.weights]
            continue
        i += 1
    if len(node.children) == 0:
        return None
    if len(node.children) == 1:
        return node.children[0]
    return node


def _replace_children(pc, replace_dict):
    # Prune the nodes in the PC by exchanging them with their replacement
    def _update_children(node):
        for i in range(len(node.children)):
            if node.children[i] in replace_dict:
                node.children[i] = replace_dict[node.children[i]]
        if len(set(node.children)) != len(node.children):
            # Duplicate children can only happen for Sum nodes (product nodes are decomposable)
            assert (isinstance(node, PCSum))

            # Identify the indexes of unique children
            child_indexes = {}
            for i, child in enumerate(node.children):
                if child not in child_indexes:
                    child_indexes[child] = []
                child_indexes[child].append(i)

            # Combine duplicate children
            new_children = []
            new_weights = []
            for child, indexes in child_indexes.items():
                new_children.append(child)
                new_weights.append(sum([node.weights[index] for index in indexes]))
            node.children = new_children
            node.weights = new_weights

    pc_basics.apply(pc, _update_children, include_leaves=False, only_leaves=False)


def prune_nodes(pc, node_type=None):

    # Obtain nodes
    nodes = pc_basics.get_nodes_by_type(pc, node_type=node_type)

    # Group leaves by scope
    scope_nodes = {}
    for i, node in enumerate(nodes):
        scope = frozenset(node.scope)
        if scope not in scope_nodes:
            scope_nodes[scope] = []
        scope_nodes[scope].append(node)

    # Compare nodes with same scope and determine which leaves can be pruned
    replace_dict = {}
    for scope, nodes in scope_nodes.items():
        similarity_dict = {}
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes):
                # Currently assumes to use symmetric distance measures
                if j <= i:
                    continue
                # Compute tue distance between the two nodes
                if pc_compare._is_similar(node1, node2):
                    if node1 not in similarity_dict:
                        similarity_dict[node1] = set()
                    if node2 not in similarity_dict:
                        similarity_dict[node2] = set()
                    similarity_dict[node1].add(node2)
                    similarity_dict[node2].add(node1)

        # Determine which leaves can be merged together by using the similarity sets. The leaves
        # which can be merged with more leaves will be preferred. Remember that if a leaf is already
        # been pruned by another leaf, it cannot be pruned by other leaves.
        # Generates a dictionary which key is the leaf to be removed and the value is its replacement.
        pruned_leaves = set()
        leaf_sets_sorted = sorted([(leaf, similar_leaves) for leaf, similar_leaves in similarity_dict.items()], reverse=True, key=lambda x: len(x[1]))
        for leaf, similar_leaves in leaf_sets_sorted:
            if leaf not in pruned_leaves:
                similar_leaves_diff = similar_leaves.difference(pruned_leaves)
                if len(similar_leaves_diff) > 0:
                    pruned_leaves.update(similar_leaves_diff)
                    # Here one could also compute the mean distribution over the similar leaves
                    for similar_leaf in similar_leaves:
                        replace_dict[similar_leaf] = leaf

    # Replace the leaves in the spn
    _replace_children(pc, replace_dict)

    # Update the node statistics in the spn
    pc_basics.update_scope(pc)


def contract(pc):
    def _contract_recursive(node, parent_node=None, index=None):
        if isinstance(node, PCLeaf):
            # Do not remove node
            return False
        else:
            if len(node.children) == 0:
                # Remove node
                return True
            elif len(node.children) == 1:
                # Remove current node by returning its child. In case the child has also
                # been pruned, return their result.
                rem = _contract_recursive(node.children[0], node, 0)
                if rem:
                    return True
                else:
                    if parent_node is None:
                        return node.children[0]
                    else:
                        parent_node.children[index] = node.children[0]
                    return False
            else:
                is_sum = isinstance(node, PCSum)
                new_children = []
                new_weights = []
                i = 0
                while i < len(node.children):

                    # Contract child
                    rem = _contract_recursive(node.children[i], node, i)

                    # In case of a product node, check for offset-leaves to remove
                    if not rem and not is_sum:
                        if isinstance(node.children[i], OffsetLeaf):
                            rem = True
                    if rem:
                        # In case the child node need to be removed
                        del node.children[i]
                        if is_sum:
                            del node.weights[i]
                        continue
                    else:
                        if is_sum:
                            if isinstance(node.children[i], PCSum):
                                # Combine sum nodes and remove child
                                new_children += [child for child in node.children[i].children]
                                new_weights += [weight * node.weights[i] for weight in node.children[i].weights]
                                del node.children[i]
                                del node.weights[i]
                                continue
                        else:
                            if isinstance(node.children[i], PCProduct):
                                # Combine product nodes and remove child
                                new_children += [child for child in node.children[i].children]
                                del node.children[i]
                                continue
                    i += 1

            # Update children and weights
            if len(new_children) > 0:
                node.children += new_children
                if len(new_weights) > 0:
                    node.weights += new_weights

            # Remove node if no children exist
            if len(node.children) == 0:
                return True
            if len(node.children) == 1:
                if parent_node is None:
                    return node.children[0]
                else:
                    parent_node.children[index] = node.children[0]
            return False

    new_root = _contract_recursive(pc)
    if isinstance(new_root, (PCProduct, PCSum, PCLeaf)):
        return new_root
    return pc


def condition(pc, inst, remove_cond_nodes=True):
    def _condition(node, cur_scope):
        if isinstance(node, PCProduct):
            results = [_condition(child, cur_scope.intersection(child.scope)) for child in node.children]
            probs, new_children = list(map(list, zip(*results)))
            if None in probs:
                return None, None
            new_children = [new_child for new_child in new_children if new_child is not None]
            if len(new_children) == 0 or np.prod(probs) == 0.0:
                return np.prod(probs), None
            return np.prod(probs), PCProduct(children=new_children)

        elif isinstance(node, PCSum):
            if len(cur_scope) > 0:
                probs = []
                new_children = []
                new_weights = []
                for weight, child in zip(node.weights, node.children):
                    if cur_scope.issubset(child.scope):
                        prob, new_child = _condition(child, cur_scope.intersection(child.scope))
                        if prob is None:
                            continue
                        if new_child is not None and prob > 0.0:
                            new_children.append(new_child)
                            new_weights.append(prob * weight)
                        if prob > 0.0:
                            probs.append(prob * weight)
                if len(new_children) == 0:
                    if len(probs) == 0:
                        return None, None
                    return np.sum(probs), None
                return np.sum(probs), PCSum(children=new_children, weights=list(np.array(new_weights) / np.sum(new_weights)))
            else:
                return 1.0, node

        elif isinstance(node, PCLeaf):
            if len(cur_scope) > 0 and cur_scope.issubset(node.scope):
                prob = node.prob(inst)
                if prob is None:
                    return None, None
                return prob, None if remove_cond_nodes else CategoricalLeaf(scope=node.scope, val_prob_dict={inst[s]: 1.0 for s in cur_scope})
            else:
                return 1.0, node
        else:
            raise Exception("Unknown node: {}".format(node))

    if pc is None:
        logger.warning("Conditioning on empty structure: {}".format(inst))
        return None, None

    if inst is None or len(inst) == 0:
        return 1.0, pc

    cond_prob, cond_structure = _condition(pc, set(inst.keys()))

    if cond_prob is None:
        logger.warning("Conditioning on unknown evidence: {}".format(inst))
        return None, None
    if cond_prob == 0.0:
        logger.warning("Conditioned on evidence which has 0 probability!")
    if cond_structure is None:
        logger.warning("Conditioning resulted in an empty structure!")
    else:
        pc_basics.update_scope(cond_structure)
        cond_structure = contract(cond_structure)
    return cond_prob, cond_structure


'''
def _deduplicate_inner_nodes(node):
    if isinstance(node, PCLeaf):
        return node
    elif isinstance(node, PCProduct):
        return PCProduct(children=[_deduplicate_inner_nodes(c) for c in node.children])
    elif isinstance(node, PCSum):
        return PCSum(weights=list(node.weights), children=[_deduplicate_inner_nodes(c) for c in node.children])


def product_node_injection(structure):

    # First approach: Break inner node dependencies by transforming to a tree
    nodes = pc_basics.get_nodes_by_type(structure, node_type=(PCSum, PCProduct))
    dup_nodes = set([node for node, count in Counter(nodes).items() if count > 1])
    if len(dup_nodes) > 0:
        structure = _deduplicate_inner_nodes(structure)
        logger.warning("Inner structure of the PC have been unfolded for pruning!")

    nodes = pc_basics.get_nodes_by_type(structure, node_type=PCLeaf)
    print(Counter(nodes))


    exit()

    def _injection(node, rem_leaves, c):
        if isinstance(node, PCProduct):
            new_children = []
            for child in node.children:
                new_node = _injection(child, rem_leaves, c+1)
                if new_node is not None:
                    new_children.append(new_node)
            if len(new_children) == 0:
                return None
            return PCProduct(children=new_children)
        elif isinstance(node, PCSum):

            leaf_sets = []
            for i, child in enumerate(node.children):
                leaves = [leaf for leaf in pc_basics.get_nodes_by_type(child, PCLeaf) if leaf not in rem_leaves]

                # Generate dictionary which collects the leaves for each scope
                scope_dict = {}
                for leaf in leaves:
                    # Currently only support leaves with single scope
                    assert (len(leaf.scope) == 1)
                    scope, = leaf.scope
                    if scope not in scope_dict:
                        scope_dict[scope] = []
                    scope_dict[scope].append(leaf)

                # Generate the leaf sets from the scope dictionary. We are only interested in leaves which are
                # unique in the sub-spn (e.g. for the same scope only one type leaf can exist).
                leaf_set = set()
                for scope, nodes in scope_dict.items():
                    if len(set(nodes)) == 1:
                        leaf_set.add(nodes[0])
                leaf_sets.append(leaf_set)

            # Try to select the children such that the most common leaf can be pruned
            counts = sum([Counter(leaf_set) for leaf_set in leaf_sets], Counter())

            print("HEHEHHEHEHE")
            print(counts)
            print(rem_leaves)

            if len(counts) > 0:
                # All most common
                most_common = counts.most_common()
                sel_leaf, count = [leaf for leaf, count in most_common if count == max_count]
                if count > 1:
                    logger.debug("Combine " + str(count) + " (of " + str(len(node.children)) + ") children to prune one leaf!")

                    mask = np.array([sel_leaf in leaf_set for leaf_set in leaf_sets])
                    children = np.array(node.children)
                    weights = np.array(node.weights)
                    prod_children = [sel_leaf]

                    #print(rem_leaves)
                    #print(rem_leaves.union({sel_leaf}))

                    sel_sum = PCSum(weights=list(weights[mask]/np.sum(weights[mask])), children=list(children[mask]))
                    new_node = _injection(sel_sum, rem_leaves.union({sel_leaf}), c+1)
                    if new_node is not None:
                        prod_children.append(new_node)

                    inj_node = PCProduct(children=prod_children)
                    if sum(mask) < len(node.children):
                        new_children = list(children[~mask]) + [inj_node]
                        new_weights = list(weights[~mask]) + [np.sum(weights[mask])]
                        inj_node = PCSum(weights=new_weights, children=new_children)

                    #return inj_node
                    #print(c)
                    #new_node = _injection(inj_node, rem_leaves, c+1)

                    #print(c)
                    #print(new_node)
                    #print(leaf_sets)
                    #print(rem_leaves)
                    #print(rem_leaves.union({sel_leaf}))
                    #print()


                    #print("dsads " + str(new_node.children))
                    if new_node is None:
                        return None
                    return new_node

            new_children = []
            new_weights = []
            for weight, child in zip(node.weights, node.children):
                new_node = _injection(child, rem_leaves, c+1)
                if new_node is not None:
                    new_weights.append(weight)
                    new_children.append(new_node)
            if len(new_children) == 0:
                return None
            return PCSum(children=new_children, weights=list(np.array(new_weights)/np.sum(new_weights)))

        elif isinstance(node, PCLeaf):
            #if node in rem_leaves:
            #    return None
            return node
        else:
            raise Exception("Unknown node: {}".format(node))

    _get_node_prune_dict(structure)
    exit()

    new_structure = _injection(structure, set(), 0)

    pc_basics.update_scope(new_structure)
    return new_structure
    #return contract_nodes(new_structure)




def product_node_injection_old(structure):

    def _injection(node, rem_leaves, c):
        if isinstance(node, PCProduct):
            new_children = []
            for child in node.children:
                new_node = _injection(child, rem_leaves, c+1)
                if new_node is not None:
                    new_children.append(new_node)
            if len(new_children) == 0:
                return None
            return PCProduct(children=new_children)
        elif isinstance(node, PCSum):

            leaf_sets = []
            for i, child in enumerate(node.children):
                leaves = [leaf for leaf in pc_basics.get_nodes_by_type(child, PCLeaf) if leaf not in rem_leaves]

                # Generate dictionary which collects the leaves for each scope
                scope_dict = {}
                for leaf in leaves:
                    # Currently only support leaves with single scope
                    assert (len(leaf.scope) == 1)
                    scope, = leaf.scope
                    if scope not in scope_dict:
                        scope_dict[scope] = []
                    scope_dict[scope].append(leaf)

                # Generate the leaf sets from the scope dictionary. We are only interested in leaves which are
                # unique in the sub-spn (e.g. for the same scope only one type leaf can exist).
                leaf_set = set()
                for scope, nodes in scope_dict.items():
                    if len(set(nodes)) == 1:
                        leaf_set.add(nodes[0])
                leaf_sets.append(leaf_set)

            # Try to select the children such that the most common leaf can be pruned
            counts = sum([Counter(leaf_set) for leaf_set in leaf_sets], Counter())

            print("HEHEHHEHEHE")
            print(counts)
            print(rem_leaves)

            if len(counts) > 0:
                # All most common
                most_common = counts.most_common()
                sel_leaf, count = [leaf for leaf, count in most_common if count == max_count]
                if count > 1:
                    logger.debug("Combine " + str(count) + " (of " + str(len(node.children)) + ") children to prune one leaf!")

                    mask = np.array([sel_leaf in leaf_set for leaf_set in leaf_sets])
                    children = np.array(node.children)
                    weights = np.array(node.weights)
                    prod_children = [sel_leaf]

                    #print(rem_leaves)
                    #print(rem_leaves.union({sel_leaf}))

                    sel_sum = PCSum(weights=list(weights[mask]/np.sum(weights[mask])), children=list(children[mask]))
                    new_node = _injection(sel_sum, rem_leaves.union({sel_leaf}), c+1)
                    if new_node is not None:
                        prod_children.append(new_node)

                    inj_node = PCProduct(children=prod_children)
                    if sum(mask) < len(node.children):
                        new_children = list(children[~mask]) + [inj_node]
                        new_weights = list(weights[~mask]) + [np.sum(weights[mask])]
                        inj_node = PCSum(weights=new_weights, children=new_children)

                    #return inj_node
                    #print(c)
                    #new_node = _injection(inj_node, rem_leaves, c+1)

                    #print(c)
                    #print(new_node)
                    #print(leaf_sets)
                    #print(rem_leaves)
                    #print(rem_leaves.union({sel_leaf}))
                    #print()


                    #print("dsads " + str(new_node.children))
                    if new_node is None:
                        return None
                    return new_node

            new_children = []
            new_weights = []
            for weight, child in zip(node.weights, node.children):
                new_node = _injection(child, rem_leaves, c+1)
                if new_node is not None:
                    new_weights.append(weight)
                    new_children.append(new_node)
            if len(new_children) == 0:
                return None
            return PCSum(children=new_children, weights=list(np.array(new_weights)/np.sum(new_weights)))

        elif isinstance(node, PCLeaf):
            #if node in rem_leaves:
            #    return None
            return node
        else:
            raise Exception("Unknown node: {}".format(node))

    new_structure = _injection(structure, set(), 0)

    pc_basics.update_scope(new_structure)
    return new_structure
    #return contract_nodes(new_structure)
'''
