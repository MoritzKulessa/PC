from probabilistic_circuits import pc_learn, pc_query, pc_prune, pc_rebuild, pc_plot, pc_basics

# todo ideas
# Use parents in addition to children, and implemented the algorithms bottom up ... or with topological order, need depth for node to order


def check_sampling(pc):
    samples = []
    for s in pc_inference.sample(pc, inst={"d": True}, n_samples=1000):
        sample_str = ", ".join(sorted([str(k) + "=" + str(v) for k, v in s.items()]))
        samples.append(sample_str)

    count_dict = {}
    for sample in samples:
        if sample not in count_dict:
            count_dict[sample] = 0
        count_dict[sample] += 1

    for k, v in count_dict.items():
        print("{} : {}".format(v, k))



if __name__ == "__main__":

    import sys
    import logging

    # show logging
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    #use_case_text()
    #exit()

    train_instances = [
        {"a": True},
        {"a": True, "b": True},
        {"c": True, "d": True},
        {"c": True, "b": True},
        {"c": True, "b": True},
        {"a": True, "b": True, "c": True, },
        {"a": True, "b": True, "c": True, "d": True},
        {"e": True}
    ]
    train_instances2 = [
        {"b": True},
        {"d": True, "e": True},
        {"d": True, "f": True},
        {"d": True, "f": True},
        {"d": True, "f": True},
        {"d": True, "e": True, "f": True},
        {"d": True, "b": True, "a": True, "c": True},
    ]

    # Learn
    pc = pc_learn.learn_dict(instances=train_instances, min_instances_slice=5)
    #pc = pc_learn.learn_shallow(instances=train_instances)

    # Verify
    pc_basics.check_validity(pc)

    # Evaluate
    print(pc_inference.probability(pc, {"a": True, "b": True}))

    # Update
    new_pc = pc_learn.learn_dict_shallow(instances=train_instances2)
    pc = pc_learn.update(pc, len(train_instances), new_pc, len(train_instances2))

    # Verify
    pc_basics.check_validity(pc)

    # Evaluate
    print(pc_inference.probability(pc, {"a": True, "b": True}))

    # Sample
    check_sampling(pc)
