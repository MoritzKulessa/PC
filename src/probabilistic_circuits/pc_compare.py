
def _is_similar(pc1, pc2):
    #todo exact match implementation
    assert (pc1.scope == pc2.scope)


    print("Helllo")

    # Just an experimental version
    if len(pc1.scope) > 1:
        from probabilistic_circuits import pc_query
        relate1 = pc_inference.relate(pc1, inst=None)
        relate2 = pc_inference.relate(pc2, inst=None)
        return relate1 == relate2
    else:
        if len(pc1.scope) == 0:
            return True  # Constant nodes are all the same
        else:
            from probabilistic_circuits.pc_nodes import CategoricalLeaf
            assert (isinstance(pc1, CategoricalLeaf))
            assert (isinstance(pc2, CategoricalLeaf))
            return pc1.val_prob_dict == pc2.val_prob_dict


def kl_divergence(pc1, pc2):
    # todo KL ivergence

    def _kl_divergence():
        pass


    pass



if __name__ == "__main__":

    from probabilistic_circuits.pc_nodes import PCSum, PCProduct, CategoricalLeaf

    l11 = CategoricalLeaf(scope={"a"}, val_prob_dict={True: 1.0})
    l12 = CategoricalLeaf(scope={"b"}, val_prob_dict={True: 1.0})
    pc1 = PCProduct(scope={"a", "b"},children=[l11, l12])

    #l11 = CategoricalLeaf(scope={"a"}, val_prob_dict={True: 1.0})
    #l12 = CategoricalLeaf(scope={"b"}, val_prob_dict={True: 1.0})
    pc2 = PCProduct(scope={"a", "b"}, children=[l11, l12])






