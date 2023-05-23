import numpy as np
import pandas as pd

from probabilistic_circuits import pc_learn, pc_query, pc_basics, pc_stats

if __name__ == "__main__":
    import sys
    import logging

    # show logging
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    instances_df = pd.read_csv("_data/car.data", header=None, names=["a", "b", "c", "d", "e", "f", "g"], dtype=str)
    instances_dict = instances_df.to_dict('records')

    # Learn
    pc = pc_learn.learn(instances=instances_df, min_population_size=1.0)

    # Evaluate
    ll_score = np.sum([np.log(pc_query.inference(pc, d)) for d in instances_dict])
    print(ll_score)
    print(pc_stats.get_stats_string(pc))

    for i in range(20):
        print(i)
        pc = pc_learn.update(pc,
                             instances=instances_df,
                             learning_rate=0.3,
                             extract_min_population_size=0.0001,
                             learn_min_population_size=0.01)
        print(pc_stats.get_stats_string(pc))

    print("Evaluate")
    ll_score = np.sum([np.log(pc_query.inference(pc, d)) for d in instances_dict])
    print(ll_score)
    print(pc_stats.get_stats_string(pc))




    '''
    # Update
    new_pc = pc_learn.learn_shallow(instances=train_instances2)
    pc = pc_learn.combine(pc, len(train_instances), new_pc, len(train_instances2))

    # Verify
    pc_basics.check_validity(pc)

    # Evaluate
    print(pc_query.inference(pc, {"a": True, "b": True}))

    # Check samples
    samples = []
    for s in pc_query.sample(pc, n=1000):
        sample_str = ", ".join(sorted([str(k) + "=" + str(v) for k, v in s.items()]))
        samples.append(sample_str)

    count_dict = {}
    for sample in samples:
        if sample not in count_dict:
            count_dict[sample] = 0
        count_dict[sample] += 1

    for k, v in count_dict.items():
        print("{} : {}".format(v, k))
    '''