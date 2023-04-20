import unittest
from probabilistic_circuits import pc_prune, pc_basics, pc_query
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


class TestPruning(unittest.TestCase):
    def test_contract(self):
        """Tests for the method contract(...)"""

        # Check circuit which cannot be pruned
        pc = get_example_pc()
        pruned_pc = pc_prune.contract(pc)
        self.assertEqual(pc, pruned_pc)
        self.assertEqual(len(pc_basics.get_nodes(pc)), len(pc_basics.get_nodes(pruned_pc)))
        pc_basics.check_validity(pruned_pc)

        # Data for testing
        leaf1 = ValueLeaf(scope={"a"}, value="a")
        leaf2 = ValueLeaf(scope={"b"}, value="b")
        leaf3 = ValueLeaf(scope={"c"}, value="c")

        # Test removing sum/product node which does not have a child
        s1 = PCSum(scope=set(), children=[], weights=[])
        s2 = PCSum(scope={"a", "b", "c"}, children=[leaf1, leaf2, s1, leaf3], weights=[0.3, 0.4, 124423, 0.3])
        pruned_pc = pc_prune.contract(s2)
        assert (isinstance(pruned_pc, PCSum))
        self.assertEqual(3, len(pruned_pc.children))
        self.assertEqual(3, len(pruned_pc.weights))
        self.assertTrue({leaf1, leaf2, leaf3} == set(pruned_pc.children))
        p1 = PCProduct(scope=set(), children=[])
        s2 = PCSum(scope={"a", "b", "c"}, children=[leaf1, leaf2, p1, leaf3], weights=[0.3, 0.4, 124423, 0.3])
        pruned_pc = pc_prune.contract(s2)
        assert (isinstance(pruned_pc, PCSum))
        self.assertEqual(3, len(s2.children))
        self.assertEqual(3, len(s2.weights))
        self.assertTrue({leaf1, leaf2, leaf3} == set(pruned_pc.children))
        s1 = PCSum(scope=set(), children=[], weights=[])
        p1 = PCProduct(scope={"a", "b", "c"}, children=[leaf1, leaf2, s1, leaf3])
        pruned_pc = pc_prune.contract(p1)
        assert (isinstance(pruned_pc, PCProduct))
        self.assertEqual(3, len(p1.children))
        self.assertTrue({leaf1, leaf2, leaf3} == set(pruned_pc.children))
        p1 = PCProduct(scope=set(), children=[])
        p2 = PCProduct(scope={"a", "b", "c"}, children=[leaf1, leaf2, p1, leaf3])
        pruned_pc = pc_prune.contract(p2)
        assert (isinstance(pruned_pc, PCProduct))
        self.assertEqual(3, len(p2.children))
        self.assertTrue({leaf1, leaf2, leaf3} == set(pruned_pc.children))

        # Test removing sum/product node which has single child
        s1 = PCSum(scope={"a", "b", "c"}, children=[leaf1, leaf2, leaf3], weights=[0.3, 0.4, 0.3])
        s2 = PCSum(scope={"a", "b", "c"}, children=[s1], weights=[1.0])
        pruned_pc = pc_prune.contract(s2)
        self.assertEqual(s1, pruned_pc)
        self.assertNotEqual(s2, pruned_pc)
        s1 = PCSum(scope={"a", "b", "c"}, children=[leaf1, leaf2, leaf3], weights=[0.3, 0.4, 0.3])
        p1 = PCProduct(scope={"a", "b", "c"}, children=[s1])
        pruned_pc = pc_prune.contract(p1)
        assert (isinstance(pruned_pc, PCSum))
        self.assertEqual(s1, pruned_pc)
        self.assertNotEqual(p1, pruned_pc)

        # Test removing double sum/product node which does not have a child
        s1 = PCSum(scope={"a", "b", "c"}, children=[leaf1, leaf2, leaf3], weights=[0.3, 0.4, 0.3])
        s2 = PCSum(scope={"a", "b", "c"}, children=[s1], weights=[1.0])
        s3 = PCSum(scope={"a", "b", "c"}, children=[s2], weights=[1.0])
        pruned_pc = pc_prune.contract(s3)
        self.assertEqual(s1, pruned_pc)
        self.assertNotEqual(s2, pruned_pc)
        self.assertNotEqual(s3, pruned_pc)
        s1 = PCSum(scope={"a", "b", "c"}, children=[leaf1, leaf2, leaf3], weights=[0.3, 0.4, 0.3])
        p1 = PCProduct(scope={"a", "b", "c"}, children=[s1])
        p2 = PCProduct(scope={"a", "b", "c"}, children=[p1])
        pruned_pc = pc_prune.contract(p2)
        self.assertEqual(s1, pruned_pc)
        self.assertNotEqual(p1, pruned_pc)
        self.assertNotEqual(p2, pruned_pc)

        # Test combine sum nodes
        s1 = PCSum(scope={"a", "b"}, children=[leaf1, leaf2], weights=[0.5, 0.5])
        s2 = PCSum(scope={"a", "b", "c"}, children=[s1, leaf3], weights=[0.5, 0.5])
        pruned_pc = pc_prune.contract(s2)
        assert (isinstance(pruned_pc, PCSum))
        self.assertEqual(3, len(pruned_pc.children))
        self.assertTrue({leaf1, leaf2, leaf3} == set(pruned_pc.children))
        self.assertEqual(3, len(pruned_pc.weights))
        s1 = PCSum(scope={"a"}, children=[leaf1], weights=[0.5, 0.5])
        s2 = PCSum(scope={"a", "b"}, children=[s1, leaf2], weights=[0.5, 0.5])
        s3 = PCSum(scope={"a", "b", "c"}, children=[s2, leaf3], weights=[0.5, 0.5])
        pruned_pc = pc_prune.contract(s3)
        assert (isinstance(pruned_pc, PCSum))
        self.assertEqual(3, len(pruned_pc.children))
        self.assertTrue({leaf1, leaf2, leaf3} == set(pruned_pc.children))
        self.assertEqual(3, len(pruned_pc.weights))

        # Test combine product nodes
        p1 = PCProduct(scope={"a", "b"}, children=[leaf1, leaf2])
        p2 = PCProduct(scope={"a", "b", "c"}, children=[p1, leaf3])
        pruned_pc = pc_prune.contract(p2)
        assert (isinstance(pruned_pc, PCProduct))
        self.assertEqual(3, len(pruned_pc.children))
        self.assertTrue({leaf1, leaf2, leaf3} == set(pruned_pc.children))
        p1 = PCProduct(scope={"a"}, children=[leaf1])
        p2 = PCProduct(scope={"a", "b"}, children=[p1, leaf2])
        p3 = PCProduct(scope={"a", "b", "c"}, children=[p2, leaf3])
        pruned_pc = pc_prune.contract(p3)
        assert (isinstance(pruned_pc, PCProduct))
        self.assertEqual(3, len(pruned_pc.children))
        self.assertTrue({leaf1, leaf2, leaf3} == set(pruned_pc.children))

    def test_condition(self):
        """Tests for the method condition(...)"""
        pc = get_example_pc()

        # Check empty evidence
        cond_prob, cond_pc = pc_prune.condition(pc, evidence={}, remove_conditioned_nodes=True)
        self.assertEqual(1.0, cond_prob)
        self.assertEqual(pc, cond_pc)
        self.assertEqual(len(pc_basics.get_nodes(pc)), len(pc_basics.get_nodes(cond_pc)))
        pc_basics.check_validity(pc)
        self.assertAlmostEqual(0.150, pc_query.inference(cond_pc, {"car": "BMW"}))
        self.assertAlmostEqual(0.200, pc_query.inference(cond_pc, {"car": "Mercedes"}))
        self.assertAlmostEqual(0.150, pc_query.inference(cond_pc, {"car": "VW"}))
        self.assertAlmostEqual(0.250, pc_query.inference(cond_pc, {"airplane": "Airbus"}))
        self.assertAlmostEqual(0.250, pc_query.inference(cond_pc, {"airplane": "Boing"}))
        self.assertAlmostEqual(0.750, pc_query.inference(cond_pc, {"equipment": "radio"}))
        self.assertAlmostEqual(0.075, pc_query.inference(cond_pc, {"car": "BMW", "equipment": "radio"}))
        self.assertAlmostEqual(0.100, pc_query.inference(cond_pc, {"car": "Mercedes", "equipment": "radio"}))
        self.assertAlmostEqual(0.075, pc_query.inference(cond_pc, {"car": "VW", "equipment": "radio"}))
        self.assertAlmostEqual(0.250, pc_query.inference(cond_pc, {"airplane": "Airbus", "equipment": "radio"}))
        self.assertAlmostEqual(0.250, pc_query.inference(cond_pc, {"airplane": "Boing", "equipment": "radio"}))

        # Check invalid evidence
        cond_prob, cond_pc = pc_prune.condition(pc, evidence={"car": "Toyota"}, remove_conditioned_nodes=True)
        self.assertTrue(cond_prob is None)
        self.assertTrue(cond_pc is None)

        # Check valid evidence 1
        cond_prob, cond_pc = pc_prune.condition(pc, evidence={"equipment": "radio"}, remove_conditioned_nodes=True)
        self.assertAlmostEqual(0.75, cond_prob)
        self.assertNotEqual(pc, cond_pc)
        self.assertTrue(len(pc_basics.get_nodes(pc)) > len(pc_basics.get_nodes(cond_pc)))
        pc_basics.check_validity(pc)
        self.assertAlmostEqual(0.3 * 1 / 3, pc_query.inference(cond_pc, {"car": "BMW"}))
        self.assertAlmostEqual(0.4 * 1 / 3, pc_query.inference(cond_pc, {"car": "Mercedes"}))
        self.assertAlmostEqual(0.3 * 1 / 3, pc_query.inference(cond_pc, {"car": "VW"}))
        self.assertAlmostEqual(1 / 3, pc_query.inference(cond_pc, {"airplane": "Airbus"}))
        self.assertAlmostEqual(1 / 3, pc_query.inference(cond_pc, {"airplane": "Boing"}))
        self.assertAlmostEqual(0.0, pc_query.inference(cond_pc, {"equipment": "radio"}))

        # Check without removing conditioned nodes 1
        cond_prob, cond_pc = pc_prune.condition(pc, evidence={"equipment": "radio"}, remove_conditioned_nodes=False)
        self.assertAlmostEqual(0.3 * 1 / 3, pc_query.inference(cond_pc, {"car": "BMW"}))
        self.assertAlmostEqual(0.4 * 1 / 3, pc_query.inference(cond_pc, {"car": "Mercedes"}))
        self.assertAlmostEqual(0.3 * 1 / 3, pc_query.inference(cond_pc, {"car": "VW"}))
        self.assertAlmostEqual(1 / 3, pc_query.inference(cond_pc, {"airplane": "Airbus"}))
        self.assertAlmostEqual(1 / 3, pc_query.inference(cond_pc, {"airplane": "Boing"}))
        self.assertAlmostEqual(0.3 * 1 / 3, pc_query.inference(cond_pc, {"car": "BMW", "equipment": "radio"}))
        self.assertAlmostEqual(0.4 * 1 / 3, pc_query.inference(cond_pc, {"car": "Mercedes", "equipment": "radio"}))
        self.assertAlmostEqual(0.3 * 1 / 3, pc_query.inference(cond_pc, {"car": "VW", "equipment": "radio"}))
        self.assertAlmostEqual(1 / 3, pc_query.inference(cond_pc, {"airplane": "Airbus", "equipment": "radio"}))
        self.assertAlmostEqual(1 / 3, pc_query.inference(cond_pc, {"airplane": "Boing", "equipment": "radio"}))
        self.assertAlmostEqual(1.0, pc_query.inference(cond_pc, {"equipment": "radio"}))

        # Check valid evidence 2
        cond_prob, cond_pc = pc_prune.condition(pc, evidence={"car": "VW"}, remove_conditioned_nodes=True)
        self.assertAlmostEqual(0.15, cond_prob)
        self.assertNotEqual(pc, cond_pc)
        self.assertTrue(len(pc_basics.get_nodes(pc)) > len(pc_basics.get_nodes(cond_pc)))
        pc_basics.check_validity(pc)
        self.assertAlmostEqual(0.5, pc_query.inference(cond_pc, {"equipment": "radio"}))

        # Check without removing conditioned nodes 1
        cond_prob, cond_pc = pc_prune.condition(pc, evidence={"car": "VW"}, remove_conditioned_nodes=False)
        self.assertAlmostEqual(0.15, cond_prob)
        self.assertNotEqual(pc, cond_pc)
        self.assertTrue(len(pc_basics.get_nodes(pc)) > len(pc_basics.get_nodes(cond_pc)))
        pc_basics.check_validity(pc)
        self.assertAlmostEqual(1.0, pc_query.inference(cond_pc, {"car": "VW"}))
        self.assertAlmostEqual(0.5, pc_query.inference(cond_pc, {"equipment": "radio"}))
        self.assertAlmostEqual(0.5, pc_query.inference(cond_pc, {"car": "VW", "equipment": "radio"}))

    def test_prune_leaves(self):
        """Tests for the method prune_leaves(...)"""

        # Check for the example circuit
        pc = get_example_pc()
        pc_prune.prune_leaves(pc)
        self.assertEqual(pc.children[0].children[1].children[0], pc.children[1].children[1])

        # Dummy data
        leaf1 = ValueLeaf(scope={"a"}, value="a")
        leaf2 = ValueLeaf(scope={"a"}, value="a")
        leaf3 = ValueLeaf(scope={"a"}, value="b")
        leaf4 = ValueLeaf(scope={"c"}, value="c")

        # Check single sum node with same leaves
        pc = PCSum(scope={"a", "c"}, children=[leaf1, leaf2, leaf3, leaf4], weights=[0.5, 0.25, 0.2, 0.05])
        pc_prune.prune_leaves(pc)
        self.assertEqual(pc.children[0], pc.children[1])
        self.assertNotEqual(pc.children[0], pc.children[2])
        self.assertNotEqual(pc.children[0], pc.children[3])
        self.assertNotEqual(pc.children[1], pc.children[2])
        self.assertNotEqual(pc.children[1], pc.children[3])
        self.assertNotEqual(pc.children[2], pc.children[3])

        # Check single sum node with same leaves
        s1 = PCSum(scope={"a"}, children=[leaf1, leaf3], weights=[0.5, 0.5])
        pc = PCSum(scope={"a"}, children=[s1, leaf2], weights=[0.5, 0.5])
        pc_prune.prune_leaves(pc)
        self.assertEqual(pc.children[1], pc.children[0].children[0])


if __name__ == '__main__':
    unittest.main()
