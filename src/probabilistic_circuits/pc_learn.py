import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from probabilistic_circuits import pc_basics, pc_prune
from probabilistic_circuits.pc_nodes import PCNode, PCSum, PCProduct, ValueLeaf, OffsetLeaf

import logging

logger = logging.getLogger(__name__)


def _check_for_independent_rows(data: np.ndarray, nan_value: object) -> tuple[list[list[int]], list: int]:
    """
    Checks if the given matrix contains rows or groups of rows which are independent of each other.
    Returns a tuple. The first item of the tuple is a list containing a list of row-indices. Each list of row-indices
    represent a group rows which are independent all other rows. The second item is a list of columns representing which
    columns are associated with respective independent group.
    """
    assert (len(data.shape) == 2)

    matrix = data != nan_value
    groups = [set(np.arange(matrix.shape[0])[matrix[:, 0]])]
    col_ids = [{0}]
    for col_id in range(1, matrix.shape[1]):
        indexes = set(np.arange(matrix.shape[0])[matrix[:, col_id]])
        is_intersect = False
        for i in range(len(groups)):
            un = groups[i].union(indexes)
            if len(un) < (len(indexes) + len(groups[i])):
                groups[i] = un
                col_ids[i].add(col_id)
                is_intersect = True
                break

        if not is_intersect:
            groups.append(indexes)
            col_ids.append({col_id})

        for group in groups:
            if len(group) == matrix.shape[0]:
                return None, None

    rem_groups = set()
    for i in range(len(groups)):
        if i in rem_groups:
            continue
        for j in range(len(groups)):
            if i <= j or j in rem_groups:
                continue
            un = groups[i].union(groups[j])
            if len(un) < (len(groups[j]) + len(groups[i])):
                groups[i] = un
                col_ids[i] = col_ids[i].union(col_ids[j])
                rem_groups.add(j)

    rem_groups = sorted(rem_groups, reverse=True)
    for group_id in sorted(rem_groups, reverse=True):
        del groups[group_id]
        del col_ids[group_id]

    if len(groups) == 1:
        return None, None

    # Check if valid, can be removed
    assert (len(set.union(*groups)) == matrix.shape[0])
    assert (sum([len(group) for group in groups]) == matrix.shape[0])
    assert (len(set.intersection(*groups)) == 0)
    return [list(group) for group in groups], [list(col_id) for col_id in col_ids]


def _check_columns(data, nan_value) -> tuple[list[int], list[int], list[int]]:
    """
    Checks the values of the columns. Returns a triple. The first item of the triple contains the columns-indices for
    the columns which only contain the nan_value. The second item of the triple contains the column-indices which only
    contain a unique value (but not the nan_value). The third item of the triple contains all other columns.
    """
    nan_cols = []
    naive_split = []
    other_cols = []
    for col_id in range(data.shape[1]):
        unique_values = list(set(data[:, col_id]))
        if len(unique_values) == 1:
            if unique_values[0] == nan_value:
                nan_cols.append(col_id)
            else:
                naive_split.append(col_id)
        else:
            other_cols.append(col_id)
    return nan_cols, naive_split, other_cols


def _learn(data: np.ndarray,
           columns: np.array,
           weights: np.array,
           min_population_size: float,
           categorical_columns: np.array,
           nan_value: object) -> PCNode:
    """
    Learns the structure of a circuit recursively using the given matrix and names for the columns. The growing of the
    structure will be stopped if less than min_instances_slice instances are available. The nan_value is the placeholder
    for values which are unknown.
    """
    assert (len(data.shape) == 2)
    assert (data.shape[0] == len(weights))
    assert (data.shape[1] == len(columns))
    assert (0.0 <= min_population_size <= 1.0)

    def _learn_recursive(cur_data: np.ndarray, cur_columns: np.array, cur_weights: np.array):
        assert (len(cur_data.shape) == 2)
        assert (cur_data.shape[0] > 0)
        assert (cur_data.shape[1] > 0)
        assert (cur_data.shape[1] == len(cur_columns))

        # Finalize, if only one feature is available
        if cur_data.shape[1] == 1:
            logger.debug("Finalize: {}".format(cur_data.shape))
            unique_values = list(set(list(cur_data.flatten())))
            if len(unique_values) > 1:
                sum_children, sum_weights = [], []
                for value in unique_values:
                    indices = cur_data[:, 0] == value
                    sum_children.append(_learn_recursive(cur_data[indices], cur_columns, cur_weights[indices]))
                    sum_weights.append(np.sum(cur_weights[indices]))
                return PCSum(scope=set(cur_columns),
                             children=sum_children,
                             weights=list(sum_weights / np.sum(sum_weights)))
            if unique_values[0] == nan_value:
                return OffsetLeaf()
            return ValueLeaf(scope=set(cur_columns), value=unique_values[0])

        # Check the minimum population size
        if np.sum(cur_weights) <= min_population_size:
            logger.debug("Population size reached: {}".format(np.sum(cur_weights)))
            product_children = []
            for i in range(cur_data.shape[1]):
                product_children.append(
                    _learn_recursive(cur_data[:, i].reshape(-1, 1), cur_columns[i: i + 1], cur_weights))
            return PCProduct(scope=set(cur_columns), children=product_children)

        # Check for independent populations
        split_rows, split_cols = _check_for_independent_rows(cur_data, nan_value=nan_value)
        if split_rows is not None:
            logger.debug("Independent subpopulations: {}".format(len(split_rows)))
            sum_children, sum_weights = [], []
            for split_row, split_col in zip(split_rows, split_cols):
                data_slice = cur_data[split_row, :]
                data_slice = data_slice[:, split_col]
                sum_children.append(_learn_recursive(data_slice, cur_columns[split_col], cur_weights[split_row]))
                sum_weights.append(np.sum(cur_weights[split_row]))
            return PCSum(scope=set(cur_columns), children=sum_children, weights=list(sum_weights / np.sum(sum_weights)))

        # Naive factorization
        nan_cols, naive_cols, other_cols = _check_columns(cur_data, nan_value)
        if len(naive_cols) > 0:
            logger.debug("Naive factorization: {}".format(len(naive_cols)))

            # Compute the circuit for the columns with naive values
            product_children = []
            for col in naive_cols:
                product_children.append(_learn_recursive(cur_data[:, col:col + 1],
                                                         cur_columns[col:col + 1],
                                                         cur_weights))

            # Compute the circuit for the remaining columns
            if len(other_cols) > 0:
                data_slice = cur_data[:, other_cols]
                nan_row_mask = np.array([np.all(row == nan_value) for row in data_slice])
                if np.sum(nan_row_mask) < data_slice.shape[0]:
                    data_slice = data_slice[~nan_row_mask, :]
                    child = _learn_recursive(data_slice, cur_columns[other_cols], cur_weights[~nan_row_mask])
                    sum_children = [OffsetLeaf(), child]
                    sum_weights = [np.sum(cur_weights[nan_row_mask]), np.sum(cur_weights[~nan_row_mask])]
                    product_children.append(
                        PCSum(children=sum_children, weights=list(sum_weights / np.sum(sum_weights))))

            return PCProduct(scope=set(cur_columns[other_cols]), children=product_children)

        # Prepare data for clustering (OHE for categorical attributes)
        cur_categorical_columns = set.intersection(set(cur_columns), set(categorical_columns))
        if len(cur_categorical_columns) > 0:
            mask = [col in cur_categorical_columns for col in cur_columns]
            ohc_data = []
            for col in range(cur_data.shape[1]):
                if mask[col]:
                    unique, inverse = np.unique(cur_data[:, col].astype(str), return_inverse=True)
                    ohc_data.append(np.eye(unique.shape[0])[inverse])
            if np.sum(mask) < cur_data.shape[1]:
                ohc_data.append(cur_data[~mask])
            cluster_data = np.concatenate(ohc_data, axis=1)
        else:
            cluster_data = cur_data

        # Perform clustering
        clusters = KMeans(n_clusters=2, n_init='auto').fit_predict(cluster_data, sample_weight=cur_weights)
        cluster_indices = [np.arange(len(clusters))[clusters == i] for i in range(2)]
        logger.debug("Cluster: {}".format([len(ind) for ind in cluster_indices]))

        sum_children = []
        sum_weights = []
        for ind in cluster_indices:
            cluster_slice = cur_data[ind]
            cluster_weights = cur_weights[ind]
            _, naive_cols, other_cols = _check_columns(cluster_slice, nan_value)
            cluster_slice = cluster_slice[:, naive_cols + other_cols]
            sum_children.append(_learn_recursive(cluster_slice, cur_columns[naive_cols + other_cols], cluster_weights))
            sum_weights.append(np.sum(cluster_weights))

        return PCSum(scope=set(cur_columns), children=sum_children, weights=list(sum_weights / np.sum(sum_weights)))

    return _learn_recursive(cur_data=data, cur_columns=columns, cur_weights=weights)


def learn(instances: list[dict[object, object]] | np.ndarray | pd.DataFrame,
          columns: list[object] = None,
          weights: list[float] = None,
          min_population_size: float = 0.01,
          nan_value: object = 10000000) -> PCNode:
    """
    Learns a circuit. If a numpy matrix is given, the columns need to be specified. The growing of the structure will be
    stopped if less than min_population_size is available. The nan_value is the placeholder for values which are
    unknown.
    """
    if isinstance(instances, (np.matrix, np.ndarray)):
        assert (columns is not None)
        df = pd.DataFrame(instances, columns=columns)
    else:
        assert (columns is None)
        df = pd.DataFrame(instances)
    if nan_value in df.values:
        raise Exception("Nan-value exists in data. Use a different one.")
    df = df.fillna(nan_value)
    if weights is None:
        weights = np.array([1 / len(instances) for _ in range(len(instances))])
    categorical_columns = list(df.select_dtypes(exclude=["number", "bool_"]).columns)
    pc = _learn(data=df.to_numpy(),
                columns=np.array(df.columns),
                weights=np.array(weights),
                min_population_size=min_population_size,
                categorical_columns=categorical_columns,
                nan_value=nan_value)
    pc_basics.update_scope(pc)
    return pc_prune.contract(pc)


def learn_shallow(instances: list[dict]) -> PCNode:
    """
    Learns a shallow circuit from a list of dictionaries. Each instance will be represented by a product node with the
    assigned values as ValueLeaf nodes. All product nodes will be combined by a sum node with equal weights.
    """
    leaf_dict = {}
    s_children = []
    for inst in instances:
        p_children = []
        for k, v in inst.items():
            if (k, v) not in leaf_dict:
                leaf_dict[(k, v)] = ValueLeaf(scope={k}, value=v)
            p_children.append(leaf_dict[(k, v)])
        s_children.append(PCProduct(children=p_children))
    pc = PCSum(children=s_children, weights=list(np.full(len(instances), fill_value=1 / len(instances))))
    pc_basics.update_scope(pc)
    return pc_prune.contract(pc)


def combine(pc1: PCNode, size1: float, pc2: PCNode, size2: float) -> PCSum:
    """Combines two circuits by a sum node. The weights of the sum node is relative to their sizes."""
    return PCSum(
        children=[pc1, pc2],
        weights=[size1 / (size1 + size2), size2 / (size1 + size2)],
        scope=set.union(pc1.scope, pc2.scope)
    )


def relearn(pc: PCNode, extract_min_population_size: float = 0.01, learn_min_population_size: float = 0.01) -> PCNode:
    """Relearns the structure of the circuit by first extracting the population and then learn over the population."""
    populations = pc_basics.get_populations(pc, min_population_size=extract_min_population_size)

    weights, instances = [], []
    for population_size, leaves in populations:
        weights.append(population_size)
        d = {}
        for leaf in leaves:
            d |= leaf.mpe()
        instances.append(d)

    logger.info("Forgot {}% of populations.".format(round((1 - np.sum(weights)) * 100, 4)))

    return learn(instances, weights=weights, min_population_size=learn_min_population_size)


def update(pc: PCNode,
           instances: list[dict[object, object]] | np.ndarray | pd.DataFrame,
           columns: list[object] = None,
           weights: list[float] = None,
           learning_rate: float = 0.05,
           extract_min_population_size: float = 0.001,
           learn_min_population_size: float = 0.01) -> PCNode:
    """
    Updates the structure of the circuit by first extracting the population, adding the instances and then relearn
    the circuit.
    """
    assert (0.0 < learning_rate)

    # Parse instances into a dataframe
    if isinstance(instances, (np.matrix, np.ndarray)):
        assert (columns is not None)
        df = pd.DataFrame(instances, columns=columns)
    else:
        assert (columns is None)
        df = pd.DataFrame(instances)

    # Extract populations
    populations = pc_basics.get_populations(pc, min_population_size=extract_min_population_size)
    pc_weights, pc_instances = [], []
    for population_size, leaves in populations:
        pc_weights.append(population_size)
        d = {}
        for leaf in leaves:
            d |= leaf.mpe()
        pc_instances.append(d)

    logger.info("Forgot {}% of populations.".format(round((1 - np.sum(pc_weights)) * 100, 4)))

    # Concat the populations with the instances
    df = pd.concat([pd.DataFrame(pc_instances), df], ignore_index=True)

    # Concat weights
    sum_pc_weights = np.sum(pc_weights)
    instance_weight = (sum_pc_weights * learning_rate) / len(instances)
    if weights is None:
        weights = [instance_weight] * len(instances)
    else:
        weights = list((weights / np.sum(weights)) * instance_weight)
    weights = pc_weights + weights

    # Learn over the combined dataset
    return learn(df, weights=weights, min_population_size=learn_min_population_size)
