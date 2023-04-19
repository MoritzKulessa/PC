from probabilistic_circuits import pc_learn, pc_query, pc_basics


if __name__ == "__main__":
    import sys
    import logging

    # show logging
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

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

    # Verify
    pc_basics.check_validity(pc)

    # Evaluate
    print(pc_query.inference(pc, {"a": True, "b": True}))

    # Update
    new_pc = pc_learn.learn_dict_shallow(instances=train_instances2)
    pc = pc_learn.combine(pc, len(train_instances), new_pc, len(train_instances2))

    # Verify
    pc_basics.check_validity(pc)

    # Evaluate
    print(pc_query.inference(pc, {"a": True, "b": True}))

    # Check samples
    samples = []
    for s in pc_query.sample(pc, evidence={"d": True}, n=1000):
        sample_str = ", ".join(sorted([str(k) + "=" + str(v) for k, v in s.items()]))
        samples.append(sample_str)

    count_dict = {}
    for sample in samples:
        if sample not in count_dict:
            count_dict[sample] = 0
        count_dict[sample] += 1

    for k, v in count_dict.items():
        print("{} : {}".format(v, k))
