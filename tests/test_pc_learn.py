import unittest
from probabilistic_circuits import pc_learn, pc_query, pc_basics
from probabilistic_circuits.pc_nodes import PCSum, PCProduct, PCLeaf, ValueLeaf, OffsetLeaf


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


def get_example_pc():
    """
    Returns an example of a circuit. This circuit represents the joint probability distribution over the attribute's
    car, airplane and equipment with the following probabilities (all other probabilities are zero):

    P(car="BMW") = 0.15
    P(car="Mercedes") = 0.2
    P(car="VW") = 0.15
    P(airplane="Airbus") = 0.25
    P(airplane="Boing") = 025
    P(equipment="radio") = 0.75

    P(car="BMW", equipment="radio") = 0.075
    P(car="Mercedes", equipment="radio") = 0.1
    P(car="VW", equipment="radio") = 0.075
    P(airplane="Airbus", equipment="radio") = 0.25
    P(airplane="Boing", equipment="radio") = 025
    """

    # Multi-nominal probability distribution for car [BMW(30%), Mercedes(40%), VM(30%)]
    car1 = ValueLeaf(scope={"car"}, value="BMW")
    car2 = ValueLeaf(scope={"car"}, value="Mercedes")
    car3 = ValueLeaf(scope={"car"}, value="VW")
    s1 = PCSum(scope={"car"}, children=[car1, car2, car3], weights=[0.3, 0.4, 0.3])

    # Probability distribution for equipment (including an offset) [Radio(50%), OFFSET(50%)]
    offset = OffsetLeaf()
    equipment1 = ValueLeaf(scope={"equipment"}, value="radio")
    s2 = PCSum(scope={"equipment"}, children=[equipment1, offset], weights=[0.5, 0.5])

    # Joint probability distribution over car and equipment. Half of the cars have a radio.
    p1 = PCProduct(scope={"car", "equipment"}, children=[s1, s2])

    # Multi-nominal probability distribution for airplane [Airbus(50%), Boing(50%)]
    airplane1 = ValueLeaf(scope={"airplane"}, value="Airbus")
    airplane2 = ValueLeaf(scope={"airplane"}, value="Boing")
    s3 = PCSum(scope={"airplane"}, children=[airplane1, airplane2], weights=[0.5, 0.5])

    # Joint probability distribution over airplane and equipment. All airplanes have a radio.
    equipment1 = ValueLeaf(scope={"equipment"}, value="radio")
    p2 = PCProduct(scope={"airplane", "equipment"}, children=[s3, equipment1])

    # Joint probability distribution over car, airplane and equipment by combining the two joint probability
    # distributions which are defined above. There are 50% airplanes and 50% cars.
    return PCSum(scope={"car", "airplane", "equipment"}, children=[p1, p2], weights=[0.5, 0.5])


class TestLearning(unittest.TestCase):
    def test_learn_dict_shallow(self):
        """ Tests for the method learn_matrix(...)"""

        # Get data
        instances = get_data()

        # Learn circuit
        pc = pc_learn.learn_shallow(instances)

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

    def test_learn(self):
        """ Tests for the method learn(...)"""

        # Get data
        instances = get_data()

        # Check for parameter min_population_size equals 0.01
        pc = pc_learn.learn(instances, min_population_size=0.01)
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

        # Check for parameter min_population_size equals 0.2 - 1.0
        for i in range(2, 11):
            pc = pc_learn.learn(instances, min_population_size=i/10)
            # Check structure
            self.assertTrue(pc is not None)
            pc_basics.check_validity(pc)
            # Ensure that single probabilities are correct
            self.assertAlmostEqual(0.6, pc_query.inference(pc, {"a": True}))
            self.assertAlmostEqual(0.6, pc_query.inference(pc, {"b": True}))
            self.assertAlmostEqual(0.6, pc_query.inference(pc, {"c": True}))
            self.assertAlmostEqual(0.2, pc_query.inference(pc, {"d": True}))
            self.assertAlmostEqual(0.1, pc_query.inference(pc, {"e": True}))

        # Check for parameter min_population_size equals the dataset size
        pc = pc_learn.learn(instances, min_population_size=1.0)
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

        # Check learning with weights
        pc = pc_learn.learn(instances, min_population_size=0.01, weights=[0.09]*9 + [0.19])
        # Check structure
        self.assertTrue(pc is not None)
        pc_basics.check_validity(pc)

        # Check all probabilities
        self.assertAlmostEqual(0.54, pc_query.inference(pc, {"a": True}))
        self.assertAlmostEqual(0.54, pc_query.inference(pc, {"b": True}))
        self.assertAlmostEqual(0.54, pc_query.inference(pc, {"c": True}))
        self.assertAlmostEqual(0.18, pc_query.inference(pc, {"d": True}))
        self.assertAlmostEqual(0.19, pc_query.inference(pc, {"e": True}))
        self.assertAlmostEqual(0.36, pc_query.inference(pc, {"a": True, "b": True}))
        self.assertAlmostEqual(0.27, pc_query.inference(pc, {"a": True, "c": True}))
        self.assertAlmostEqual(0.09, pc_query.inference(pc, {"a": True, "d": True}))
        self.assertAlmostEqual(0.45, pc_query.inference(pc, {"b": True, "c": True}))
        self.assertAlmostEqual(0.18, pc_query.inference(pc, {"c": True, "d": True}))
        self.assertAlmostEqual(0.27, pc_query.inference(pc, {"a": True, "b": True, "c": True}))
        self.assertAlmostEqual(0.09, pc_query.inference(pc, {"a": True, "b": True, "c": True, "d": True}))

        # Check for parameter min_population_size equals 0.2 - 1.0
        for i in range(2, 11):
            pc = pc_learn.learn(instances, min_population_size=i / 10, weights=[0.09]*9 + [0.19])
            # Check structure
            self.assertTrue(pc is not None)
            pc_basics.check_validity(pc)
            # Ensure that single probabilities are correct
            self.assertAlmostEqual(0.54, pc_query.inference(pc, {"a": True}))
            self.assertAlmostEqual(0.54, pc_query.inference(pc, {"b": True}))
            self.assertAlmostEqual(0.54, pc_query.inference(pc, {"c": True}))
            self.assertAlmostEqual(0.18, pc_query.inference(pc, {"d": True}))
            self.assertAlmostEqual(0.19, pc_query.inference(pc, {"e": True}))

        # Check for categorical attributes
        instances = [
            {"v1": "a", "v2": "a", "v3": "a"},
            {"v1": "a", "v2": "a"},
            {"v2": "a", "v3": "a"},
            {"v1": "a", "v3": "a"},
        ]
        pc = pc_learn.learn(instances, min_population_size=0.01)
        self.assertAlmostEqual(0.75, pc_query.inference(pc, {"v1": "a"}))
        self.assertAlmostEqual(0.75, pc_query.inference(pc, {"v2": "a"}))
        self.assertAlmostEqual(0.75, pc_query.inference(pc, {"v3": "a"}))
        self.assertAlmostEqual(0.50, pc_query.inference(pc, {"v1": "a", "v2": "a"}))
        self.assertAlmostEqual(0.50, pc_query.inference(pc, {"v2": "a", "v3": "a"}))
        self.assertAlmostEqual(0.50, pc_query.inference(pc, {"v1": "a", "v3": "a"}))
        self.assertAlmostEqual(0.25, pc_query.inference(pc, {"v1": "a", "v2": "a", "v3": "a"}))


if __name__ == '__main__':
    unittest.main()
