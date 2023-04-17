import unittest

import numpy as np

from probabilistic_circuits import pc_learn, pc_query, pc_basics
from probabilistic_circuits.pc_nodes import PCSum, PCProduct, PCLeaf


def get_data():
    """
    Returns an example dataset with the following probability distribution:

    P(a) = 0.6
    P(b) = 0.6
    P(c) = 0.6
    P(d) = 0.2
    P(e) = 0.1
    P(a,b) = 0.4
    P(a,c) = 0.3
    P(a,d) = 0.1
    P(b,c) = 0.5
    P(c,d) = 0.2
    P(a,b,c) = 0.3
    P(a,b,c,d) = 0.1
    """
    return [
        {"a": True},
        {"a": True},
        {"a": True, "b": True},
        {"b": True, "c": True},
        {"b": True, "c": True},
        {"c": True, "d": True},
        {"a": True, "b": True, "c": True},
        {"a": True, "b": True, "c": True},
        {"a": True, "b": True, "c": True, "d": True},
        {"e": True},
    ]


class TestLearning(unittest.TestCase):
    def test_learn_dict_shallow(self):
        """ Tests for the method learn_matrix(...)"""

        # Get data
        instances = get_data()

        # Learn circuit
        pc = pc_learn.learn_dict_shallow(instances)

        # Check structure
        self.assertTrue(pc is not None)
        pc_basics.check_validity(pc)
        assert (isinstance(pc, PCSum))
        self.assertEqual(10, len(pc.children))
        self.assertTrue(all([isinstance(child, (PCProduct, PCLeaf)) for child in pc.children]))

        # Check probabilities
        self.assertAlmostEqual(0.6, pc_query.inference(pc, {"a": True}))
        self.assertAlmostEqual(0.6, pc_query.inference(pc, {"b": True}))
        self.assertAlmostEqual(0.6, pc_query.inference(pc, {"c": True}))
        self.assertAlmostEqual(0.2, pc_query.inference(pc, {"d": True}))
        self.assertAlmostEqual(0.1, pc_query.inference(pc, {"e": True}))
        self.assertAlmostEqual(0.4, pc_query.inference(pc, {"a": True, "b": True}))
        self.assertAlmostEqual(0.3, pc_query.inference(pc, {"a": True, "c": True}))
        self.assertAlmostEqual(0.1, pc_query.inference(pc, {"a": True, "d": True}))
        self.assertAlmostEqual(0.5, pc_query.inference(pc, {"b": True, "c": True}))
        self.assertAlmostEqual(0.2, pc_query.inference(pc, {"c": True, "d": True}))
        self.assertAlmostEqual(0.3, pc_query.inference(pc, {"a": True, "b": True, "c": True}))
        self.assertAlmostEqual(0.1, pc_query.inference(pc, {"a": True, "b": True, "c": True, "d": True}))

    def test_learn_dict(self):
        """ Tests for the method learn_matrix(...)"""

        # Get data
        instances = get_data()

        # Check for parameter minimum instances slice equals 1
        pc = pc_learn.learn_dict(instances, min_instances_slice=1)
        # Check structure
        self.assertTrue(pc is not None)
        pc_basics.check_validity(pc)
        # Check all probabilities
        self.assertAlmostEqual(0.6, pc_query.inference(pc, {"a": True}))
        self.assertAlmostEqual(0.6, pc_query.inference(pc, {"b": True}))
        self.assertAlmostEqual(0.6, pc_query.inference(pc, {"c": True}))
        self.assertAlmostEqual(0.2, pc_query.inference(pc, {"d": True}))
        self.assertAlmostEqual(0.1, pc_query.inference(pc, {"e": True}))
        self.assertAlmostEqual(0.4, pc_query.inference(pc, {"a": True, "b": True}))
        self.assertAlmostEqual(0.3, pc_query.inference(pc, {"a": True, "c": True}))
        self.assertAlmostEqual(0.1, pc_query.inference(pc, {"a": True, "d": True}))
        self.assertAlmostEqual(0.5, pc_query.inference(pc, {"b": True, "c": True}))
        self.assertAlmostEqual(0.2, pc_query.inference(pc, {"c": True, "d": True}))
        self.assertAlmostEqual(0.3, pc_query.inference(pc, {"a": True, "b": True, "c": True}))
        self.assertAlmostEqual(0.1, pc_query.inference(pc, {"a": True, "b": True, "c": True, "d": True}))

        # Check for parameter minimum instances slice equals 2-10
        for i in range(2, 11):
            pc = pc_learn.learn_dict(instances, min_instances_slice=i)
            # Check structure
            self.assertTrue(pc is not None)
            pc_basics.check_validity(pc)
            # Ensure that single probabilities are correct
            self.assertAlmostEqual(0.6, pc_query.inference(pc, {"a": True}))
            self.assertAlmostEqual(0.6, pc_query.inference(pc, {"b": True}))
            self.assertAlmostEqual(0.6, pc_query.inference(pc, {"c": True}))
            self.assertAlmostEqual(0.2, pc_query.inference(pc, {"d": True}))
            self.assertAlmostEqual(0.1, pc_query.inference(pc, {"e": True}))

        # Check for parameter minimum instances slice equals the dataset size plus one
        pc = pc_learn.learn_dict(instances, min_instances_slice=11)
        # Check structure
        self.assertTrue(pc is not None)
        pc_basics.check_validity(pc)

        # Check all probabilities
        self.assertAlmostEqual(0.6, pc_query.inference(pc, {"a": True}))
        self.assertAlmostEqual(0.6, pc_query.inference(pc, {"b": True}))
        self.assertAlmostEqual(0.6, pc_query.inference(pc, {"c": True}))
        self.assertAlmostEqual(0.2, pc_query.inference(pc, {"d": True}))
        self.assertAlmostEqual(0.1, pc_query.inference(pc, {"e": True}))
        self.assertAlmostEqual(0.36, pc_query.inference(pc, {"a": True, "b": True}))
        self.assertAlmostEqual(0.36, pc_query.inference(pc, {"a": True, "c": True}))
        self.assertAlmostEqual(0.12, pc_query.inference(pc, {"a": True, "d": True}))
        self.assertAlmostEqual(0.36, pc_query.inference(pc, {"b": True, "c": True}))
        self.assertAlmostEqual(0.12, pc_query.inference(pc, {"c": True, "d": True}))
        self.assertAlmostEqual(0.216, pc_query.inference(pc, {"a": True, "b": True, "c": True}))
        self.assertAlmostEqual(0.0432, pc_query.inference(pc, {"a": True, "b": True, "c": True, "d": True}))

    def test_learn_matrix(self):
        """ Tests for the method learn_matrix(...)"""

        # Get data
        instances = get_data()
        # Transform data to matrix
        attributes = list(set.union(*[set(inst.keys()) for inst in instances]))
        matrix = np.full((len(instances), len(attributes)), fill_value=0)
        for i, inst in enumerate(instances):
            for j, attribute in enumerate(attributes):
                if attribute in inst:
                    matrix[i][j] = inst[attribute]

        # Check for parameter minimum instances slice equals 1
        pc = pc_learn.learn_matrix(matrix, columns=attributes, min_instances_slice=1)
        # Check structure
        self.assertTrue(pc is not None)
        pc_basics.check_validity(pc)
        # Check all probabilities
        self.assertAlmostEqual(0.6, pc_query.inference(pc, {"a": True}))
        self.assertAlmostEqual(0.6, pc_query.inference(pc, {"b": True}))
        self.assertAlmostEqual(0.6, pc_query.inference(pc, {"c": True}))
        self.assertAlmostEqual(0.2, pc_query.inference(pc, {"d": True}))
        self.assertAlmostEqual(0.1, pc_query.inference(pc, {"e": True}))
        self.assertAlmostEqual(0.4, pc_query.inference(pc, {"a": True, "b": True}))
        self.assertAlmostEqual(0.3, pc_query.inference(pc, {"a": True, "c": True}))
        self.assertAlmostEqual(0.1, pc_query.inference(pc, {"a": True, "d": True}))
        self.assertAlmostEqual(0.5, pc_query.inference(pc, {"b": True, "c": True}))
        self.assertAlmostEqual(0.2, pc_query.inference(pc, {"c": True, "d": True}))
        self.assertAlmostEqual(0.3, pc_query.inference(pc, {"a": True, "b": True, "c": True}))
        self.assertAlmostEqual(0.1, pc_query.inference(pc, {"a": True, "b": True, "c": True, "d": True}))

        # Check for parameter minimum instances slice equals 2-10
        for i in range(2, 11):
            pc = pc_learn.learn_matrix(matrix, columns=attributes, min_instances_slice=i)
            # Check structure
            self.assertTrue(pc is not None)
            pc_basics.check_validity(pc)
            # Ensure that single probabilities are correct
            self.assertAlmostEqual(0.6, pc_query.inference(pc, {"a": True}))
            self.assertAlmostEqual(0.6, pc_query.inference(pc, {"b": True}))
            self.assertAlmostEqual(0.6, pc_query.inference(pc, {"c": True}))
            self.assertAlmostEqual(0.2, pc_query.inference(pc, {"d": True}))
            self.assertAlmostEqual(0.1, pc_query.inference(pc, {"e": True}))

        # Check for parameter minimum instances slice equals the dataset size plus one
        pc = pc_learn.learn_matrix(matrix, columns=attributes, min_instances_slice=11)
        # Check structure
        self.assertTrue(pc is not None)
        pc_basics.check_validity(pc)

        # Check all probabilities
        self.assertAlmostEqual(0.6, pc_query.inference(pc, {"a": True}))
        self.assertAlmostEqual(0.6, pc_query.inference(pc, {"b": True}))
        self.assertAlmostEqual(0.6, pc_query.inference(pc, {"c": True}))
        self.assertAlmostEqual(0.2, pc_query.inference(pc, {"d": True}))
        self.assertAlmostEqual(0.1, pc_query.inference(pc, {"e": True}))
        self.assertAlmostEqual(0.36, pc_query.inference(pc, {"a": True, "b": True}))
        self.assertAlmostEqual(0.36, pc_query.inference(pc, {"a": True, "c": True}))
        self.assertAlmostEqual(0.12, pc_query.inference(pc, {"a": True, "d": True}))
        self.assertAlmostEqual(0.36, pc_query.inference(pc, {"b": True, "c": True}))
        self.assertAlmostEqual(0.12, pc_query.inference(pc, {"c": True, "d": True}))
        self.assertAlmostEqual(0.216, pc_query.inference(pc, {"a": True, "b": True, "c": True}))
        self.assertAlmostEqual(0.0432, pc_query.inference(pc, {"a": True, "b": True, "c": True, "d": True}))


if __name__ == '__main__':
    unittest.main()
