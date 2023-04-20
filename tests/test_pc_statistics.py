import unittest
from probabilistic_circuits import pc_stats
from probabilistic_circuits.pc_nodes import PCSum, PCProduct, ValueLeaf, OffsetLeaf


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


class TestStatistics(unittest.TestCase):
    def test_get_n_populations(self):
        """Tests the method get_n_populations"""
        pc = get_example_pc()
        self.assertEqual(8, pc_stats.get_n_populations(pc))

    def test_get_depth(self):
        """Tests the method get_depth"""
        pc = get_example_pc()
        self.assertEqual(4, pc_stats.get_depth(pc))

    def test_get_branching_factor(self):
        """Tests the method get_branching_factor"""
        pc = get_example_pc()
        self.assertEqual(13/6, pc_stats.get_branching_factor(pc))

    def test_get_number_of_edges(self):
        """Tests the method get_number_of_edges"""
        pc = get_example_pc()
        self.assertEqual(13, pc_stats.get_number_of_edges(pc))

    def test_get_statistics(self):
        """Tests the method get_statistics"""
        pc = get_example_pc()
        stats = pc_stats.get_stats(pc)
        self.assertEqual(8, stats["n_populations"])
        self.assertEqual(14, stats["n_nodes"])
        self.assertEqual(14, stats["n_nodes_distinct"])
        self.assertEqual(8, stats["n_leaves"])
        self.assertEqual(8, stats["n_leaves_distinct"])
        self.assertEqual(4, stats["n_sums"])
        self.assertEqual(2, stats["n_products"])
        self.assertEqual(4, stats["depth"])
        self.assertEqual(13, stats["edges"])
        self.assertEqual(13/6, stats["branching_factor"])
        self.assertEqual(3, stats["n_scope"])


if __name__ == '__main__':
    unittest.main()
