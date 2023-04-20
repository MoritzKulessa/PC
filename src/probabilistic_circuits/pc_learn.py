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


def _check_columns(data, nan_value) -> tuple[list[int], list[int]]:
    """
    Checks the values of the columns. Returns a tuple. The first item of the tuple contains the columns-indices for the
    columns which only contain the nan_value. The second item of the tuple contains the column-indices which only
    contain a unique value (but not the nan_value).
    """
    nan_cols = []
    naive_split = []
    for col in range(data.shape[1]):
        unique_values = np.unique(data[:, col])
        if len(unique_values) == 1:
            if unique_values[0] == nan_value:
                nan_cols.append(col)
            else:
                naive_split.append(col)
    return nan_cols, naive_split


def _learn(data: np.ndarray,
           columns: np.array,
           weights: np.array,
           min_population_size: float,
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
            unique_values, unique_counts = np.unique(cur_data, return_counts=True)
            logger.info("Finalize {} Unique values = {}".format(cur_data.shape, len(unique_values)))
            if len(unique_values) > 1:
                sum_children, sum_weights = [], []
                for value in unique_values:
                    indices = cur_data[:, 0] == value
                    sum_children.append(_learn_recursive(cur_data[indices], cur_columns, cur_weights[indices]))
                    sum_weights.append(np.sum(cur_weights[indices]))
                return PCSum(scope=set(cur_columns),
                             children=sum_children,
                             weights=list(sum_weights/np.sum(sum_weights)))
            if unique_values[0] == nan_value:
                return OffsetLeaf()
            return ValueLeaf(scope=set(cur_columns), value=unique_values[0])

        # Check the minimum number of instances
        if np.sum(cur_weights) <= min_population_size:
            logger.info("Less than min instances slice: {}".format(len(cur_data)))
            product_children = []
            for i in range(cur_data.shape[1]):
                product_children.append(_learn_recursive(cur_data[:, i].reshape(-1, 1), cur_columns[i], cur_weights))
            return PCProduct(scope=set(cur_columns), children=product_children)

        # Check for independent populations
        split_rows, split_cols = _check_for_independent_rows(cur_data, nan_value=nan_value)
        if split_rows is not None:
            logger.info("Split independent subpopulations: {} groups".format(len(split_rows)))
            sum_children, sum_weights = [], []
            for split_row, split_col in zip(split_rows, split_cols):
                data_slice = cur_data[split_row, :]
                data_slice = data_slice[:, split_col]
                sum_children.append(_learn_recursive(data_slice, cur_columns[split_col], cur_weights[split_row]))
                sum_weights.append(np.sum(cur_weights[split_row]))
            return PCSum(scope=set(cur_columns), children=sum_children, weights=list(sum_weights/np.sum(sum_weights)))

        # Naive factorization
        nan_cols, naive_split = _check_columns(cur_data, nan_value)
        if len(naive_split) > 0:
            logger.info("Found {} constant attributes".format(naive_split))

            # Compute the circuit for the columns with naive values
            product_children = []
            for col in naive_split:
                product_children.append(_learn_recursive(cur_data[:, col:col + 1],
                                                         cur_columns[col:col + 1],
                                                         cur_weights))

            # Compute the circuit for the remaining columns
            remaining_cols = [i for i in range(cur_data.shape[1]) if i not in nan_cols and i not in naive_split]
            if len(remaining_cols) > 0:
                data_slice = cur_data[:, remaining_cols]
                nan_row_mask = np.array([np.all(row == nan_value) for row in data_slice])
                if np.sum(nan_row_mask) < data_slice.shape[0]:
                    data_slice = data_slice[~nan_row_mask, :]
                    child = _learn_recursive(data_slice, cur_columns[remaining_cols], cur_weights[~nan_row_mask])
                    sum_children = [OffsetLeaf(), child]
                    sum_weights = [np.sum(cur_weights[nan_row_mask]), np.sum(cur_weights[~nan_row_mask])]
                    product_children.append(PCSum(children=sum_children, weights=list(sum_weights/np.sum(sum_weights))))

            return PCProduct(scope=set(cur_columns[remaining_cols]), children=product_children)

        # Perform clustering
        clusters = KMeans(n_clusters=2, n_init='auto').fit_predict(cur_data, sample_weight=cur_weights)
        cluster_indices = [np.arange(len(clusters))[clusters == i] for i in range(2)]
        logger.info("Found clusters: {}".format(len(cluster_indices)))

        sum_children = []
        sum_weights = []
        for ind in cluster_indices:
            cluster_data = cur_data[ind]
            cluster_weights = cur_weights[ind]
            nan_cols, _ = _check_columns(cluster_data, nan_value)
            remaining_cols = [i for i in range(cur_data.shape[1]) if i not in nan_cols]
            cluster_data = cluster_data[:, remaining_cols]
            sum_children.append(_learn_recursive(cluster_data, cur_columns[remaining_cols], cluster_weights))
            sum_weights.append(np.sum(cluster_weights))

        return PCSum(scope=set(cur_columns), children=sum_children, weights=list(sum_weights/np.sum(sum_weights)))

    return _learn_recursive(cur_data=data, cur_columns=columns, cur_weights=weights)


def learn_matrix(instances: np.ndarray,
                 columns: list,
                 weights: list[float] = None,
                 min_population_size: float = 0.01,
                 nan_value: object = 1000000) -> PCNode:
    """
    Learns a circuit from a structured dataset. The growing of the structure will be stopped if less than
    min_population_size is available. The nan_value is the placeholder for values which are unknown.
    """
    df = pd.DataFrame(instances)
    if nan_value in df.values:
        raise Exception("Nan-value exists in data. Use a different one.")
    df = df.fillna(nan_value)
    if weights is None:
        weights = np.array([1/len(instances) for _ in range(len(instances))])
    pc = _learn(df.to_numpy(), np.array(columns), np.array(weights), min_population_size, nan_value)
    pc_basics.update_scope(pc)
    return pc_prune.contract(pc)


def learn_dict(instances: list[dict],
               weights: list[float] = None,
               min_population_size: float = 0.01,
               nan_value: object = 10000000) -> PCNode:
    """
    Learns a circuit from a list of dictionaries. The growing of the structure will be stopped if less than
    min_population_size is available. The nan_value is the placeholder for values which are unknown.
    """
    df = pd.DataFrame(instances)
    if nan_value in df.values:
        raise Exception("Nan-value exists in data. Use a different one.")
    df = df.fillna(nan_value)
    if weights is None:
        weights = np.array([1/len(instances) for _ in range(len(instances))])
    pc = _learn(df.to_numpy(), np.array(df.columns), np.array(weights), min_population_size, nan_value)
    pc_basics.update_scope(pc)
    return pc_prune.contract(pc)


def learn_dict_shallow(instances: list[dict]) -> PCNode:
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


def relearn(pc, min_population_size: float = 0.01, min_instances_slice: int = 1):
    """Relearns the structure of the circuit."""
    #todo
    pass
