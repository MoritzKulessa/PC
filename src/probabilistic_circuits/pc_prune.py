import numpy as np
from probabilistic_circuits.pc_nodes import PCNode, PCInnerNode, PCSum, PCProduct, PCLeaf, ValueLeaf, OffsetLeaf
from probabilistic_circuits import pc_basics

import logging
logger = logging.getLogger(__name__)


def contract(pc: PCNode) -> PCNode:
    """Prunes nodes from the given circuit (inplace) without changing the probability distribution"""
    def _contract_recursive(node: PCNode, parent_node: PCInnerNode = None, index: int = None):
        if isinstance(node, PCInnerNode):
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
        else:
            # Do not prune nodes which are not inner nodes
            return False

    new_root = _contract_recursive(pc)
    if isinstance(new_root, (PCProduct, PCSum, PCLeaf)):
        return new_root
    return pc


def condition(pc: PCNode, evidence: dict, remove_conditioned_nodes: bool = True) -> tuple[float | None, PCNode | None]:
    """
    Creates a new circuit by conditioning on the given evidence. If remove_conditioned_nodes is set to False, the
    created circuit keeps the nodes it was conditioned on. If leaf nodes do not change, the created circuit will use
    the original leaf nodes.
    """
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
                prob = node.inference(evidence)
                if prob is None:
                    return None, None
                if remove_conditioned_nodes:
                    return prob, None
                assert (len(cur_scope))
                s, = cur_scope
                return prob, ValueLeaf(scope={s}, value=evidence[s])
            else:
                return 1.0, node
        else:
            raise Exception("Unknown node: {}".format(node))

    if pc is None:
        logger.warning("Given circuit is None!")
        return None, None

    if evidence is None or len(evidence) == 0:
        return 1.0, pc

    cond_prob, cond_structure = _condition(pc, set(evidence.keys()))

    if cond_prob is None:
        logger.warning("Conditioning on unknown evidence: {}".format(evidence))
        return None, None
    if cond_prob == 0.0:
        logger.warning("Conditioned on evidence which has 0 probability!")
    if cond_structure is None:
        logger.warning("Conditioning resulted in an empty structure!")
    else:
        pc_basics.update_scope(cond_structure)
        cond_structure = contract(cond_structure)
    return cond_prob, cond_structure
