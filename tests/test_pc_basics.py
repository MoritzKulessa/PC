import unittest
from probabilistic_circuits import pc_basics
from probabilistic_circuits.pc_nodes import PCNode, PCSum, PCProduct, PCLeaf, ValueLeaf, OffsetLeaf


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


class TestBasics(unittest.TestCase):

    def test_apply(self):
        """ Tests for the method apply(...)"""
        pc = get_example_pc()

        def _func1(node: PCNode):
            node.scope.add("test")

        pc_basics.apply(pc, func=_func1)

        self.assertTrue("test" in pc.scope)
        self.assertTrue("test" in pc.children[0].scope)
        self.assertTrue("test" in pc.children[0].children[0].scope)
        self.assertTrue("test" in pc.children[0].children[0].children[0].scope)
        self.assertTrue("test" in pc.children[0].children[0].children[1].scope)
        self.assertTrue("test" in pc.children[0].children[0].children[2].scope)
        self.assertTrue("test" in pc.children[0].children[1].scope)
        self.assertTrue("test" in pc.children[0].children[1].children[0].scope)
        self.assertTrue("test" in pc.children[0].children[1].children[1].scope)

        self.assertTrue("test" in pc.children[1].scope)
        self.assertTrue("test" in pc.children[1].children[0].scope)
        self.assertTrue("test" in pc.children[1].children[0].children[0].scope)
        self.assertTrue("test" in pc.children[1].children[0].children[1].scope)
        self.assertTrue("test" in pc.children[1].children[1].scope)

    def test_get_nodes(self):
        """ Tests for the method get_nodes(...)."""
        pc = get_example_pc()

        self.assertEqual(14, len(pc_basics.get_nodes(pc)))
        self.assertEqual(14, len(pc_basics.get_nodes(pc, node_type=PCNode)))
        self.assertTrue(all([isinstance(node, PCNode) for node in pc_basics.get_nodes(pc, node_type=PCNode)]))

        self.assertEqual(8, len(pc_basics.get_nodes(pc, node_type=PCLeaf)))
        self.assertTrue(all([isinstance(node, PCLeaf) for node in pc_basics.get_nodes(pc, node_type=PCLeaf)]))

        self.assertEqual(4, len(pc_basics.get_nodes(pc, node_type=PCSum)))
        self.assertTrue(all([isinstance(node, PCSum) for node in pc_basics.get_nodes(pc, node_type=PCSum)]))

        self.assertEqual(2, len(pc_basics.get_nodes(pc, node_type=PCProduct)))
        self.assertTrue(all([isinstance(node, PCProduct) for node in pc_basics.get_nodes(pc, node_type=PCProduct)]))

        self.assertEqual(1, len(pc_basics.get_nodes(pc, node_type=OffsetLeaf)))
        self.assertTrue(all([isinstance(node, OffsetLeaf) for node in pc_basics.get_nodes(pc, node_type=OffsetLeaf)]))

    def test_update_scope(self):
        """ Tests for the method update_scope(...).."""
        leaf1 = ValueLeaf(scope={"a"}, value="a")
        leaf2 = ValueLeaf(scope={"b"}, value="b")

        s1 = PCSum(children=[leaf1, leaf2], weights=[0.5, 0.5])
        pc_basics.update_scope(s1)
        self.assertTrue({"a", "b"} == s1.scope)

        p1 = PCProduct(children=[leaf1, leaf2])
        pc_basics.update_scope(p1)
        self.assertTrue({"a", "b"} == p1.scope)

        pc = get_example_pc()
        pc.scope = set()
        pc.children[0].scope = set()
        pc.children[0].children[0].scope = set()
        pc.children[0].children[1].scope = set()
        pc.children[1].scope = set()
        pc.children[1].children[0].scope = set()
        pc_basics.update_scope(pc)
        self.assertEqual(pc.scope, {"airplane", "car", "equipment"})
        self.assertEqual(pc.children[0].scope, {"car", "equipment"})
        self.assertEqual(pc.children[0].children[0].scope, {"car"})
        self.assertEqual(pc.children[0].children[1].scope, {"equipment"})
        self.assertEqual(pc.children[1].scope, {"airplane", "equipment"})
        self.assertEqual(pc.children[1].children[0].scope, {"airplane"})

    def test_check_validity(self):
        """Tests for the method check_validity(...)."""

        offset_leaf = OffsetLeaf()

        # Check product node
        self.assertRaises(AssertionError, pc_basics.check_validity, pc=PCProduct(children=[None]))
        self.assertRaises(AssertionError, pc_basics.check_validity, pc=PCProduct(children=["a"]))
        pc = PCProduct(
            scope={"a"},
            children=[ValueLeaf(scope={"a"}, value="a"), ValueLeaf(scope={"b"}, value="b")]
        )
        self.assertRaises(AssertionError, pc_basics.check_validity, pc=pc)
        pc = PCProduct(
            scope={"c"},
            children=[ValueLeaf(scope={"a"}, value="a"), ValueLeaf(scope={"b"}, value="b")]
        )
        self.assertRaises(AssertionError, pc_basics.check_validity, pc=pc)
        pc = PCProduct(
            scope={"a", "b"},
            children=[ValueLeaf(scope={"a"}, value="a"), ValueLeaf(scope={"b"}, value="b")]
        )
        # Assert no AssertionError
        pc_basics.check_validity(pc)

        # Check sum node
        self.assertRaises(AssertionError, pc_basics.check_validity, pc=PCSum(children=[None], weights=[1.0]))
        self.assertRaises(AssertionError, pc_basics.check_validity, pc=PCSum(children=["a"], weights=[1.0]))
        self.assertRaises(AssertionError, pc_basics.check_validity, pc=PCSum(children=[], weights=[]))
        self.assertRaises(AssertionError, pc_basics.check_validity, pc=PCSum(children=[offset_leaf], weights=[]))
        self.assertRaises(AssertionError, pc_basics.check_validity, pc=PCSum(children=[], weights=[1.0]))
        self.assertRaises(AssertionError, pc_basics.check_validity, pc=PCSum(children=[offset_leaf], weights=[0.5]))
        pc = PCSum(
            scope={"a"},
            children=[ValueLeaf(scope={"a"}, value="a"), ValueLeaf(scope={"b"}, value="b")],
            weights=[0.5, 0.5]
        )
        self.assertRaises(AssertionError, pc_basics.check_validity, pc=pc)
        pc = PCSum(
            scope={"c"},
            children=[ValueLeaf(scope={"a"}, value="a"), ValueLeaf(scope={"b"}, value="b")],
            weights=[0.5, 0.5]
        )
        self.assertRaises(AssertionError, pc_basics.check_validity, pc=pc)
        pc = PCSum(
            scope={"a", "b"},
            children=[ValueLeaf(scope={"a"}, value="a"), ValueLeaf(scope={"b"}, value="b")],
            weights=[0.5, 0.5]
        )
        # Assert no AssertionError
        pc_basics.check_validity(pc)

        # Check example circuit
        pc = get_example_pc()
        # Assert no AssertionError
        pc_basics.check_validity(pc)

    def test_get_populations(self):
        """Tests for the method get_populations(...)."""
        pc = get_example_pc()

        def _check_population(pops: list[tuple[float, set[PCNode]]], pop: dict, pop_size: float) -> bool:
            pop_string = ", ".join(sorted([str(k) + "=" + str(v) for k, v in pop.items()]))
            for cur_pop_size, cur_pop in pops:
                cur_pop_strings = []
                for node in cur_pop:
                    assert (isinstance(node, ValueLeaf))
                    cur_pop_strings.append(str(list(node.scope)[0]) + "=" + str(node.value))
                cur_pop_string = ", ".join(sorted(cur_pop_strings))
                if pop_string == cur_pop_string:
                    self.assertAlmostEqual(cur_pop_size, pop_size)
                    return True
            return False

        # Check min_population_size = 0.0
        populations = pc_basics.get_populations(pc, min_population_size=0.0)
        self.assertEqual(8, len(populations))
        self.assertTrue(_check_population(populations, {"car": "BMW"}, 0.075))
        self.assertTrue(_check_population(populations, {"car": "Mercedes"}, 0.1))
        self.assertTrue(_check_population(populations, {"car": "VW"}, 0.075))
        self.assertTrue(_check_population(populations, {"car": "BMW", "equipment": "radio"}, 0.075))
        self.assertTrue(_check_population(populations, {"car": "Mercedes", "equipment": "radio"}, 0.1))
        self.assertTrue(_check_population(populations, {"car": "VW", "equipment": "radio"}, 0.075))
        self.assertTrue(_check_population(populations, {"airplane": "Airbus", "equipment": "radio"}, 0.25))
        self.assertTrue(_check_population(populations, {"airplane": "Boing", "equipment": "radio"}, 0.25))

        # Check min_population_size = 0.08
        populations = pc_basics.get_populations(pc, min_population_size=0.08)
        self.assertEqual(4, len(populations))
        self.assertTrue(_check_population(populations, {"car": "Mercedes"}, 0.1))
        self.assertTrue(_check_population(populations, {"car": "Mercedes", "equipment": "radio"}, 0.1))
        self.assertTrue(_check_population(populations, {"airplane": "Airbus", "equipment": "radio"}, 0.25))
        self.assertTrue(_check_population(populations, {"airplane": "Boing", "equipment": "radio"}, 0.25))


if __name__ == '__main__':
    unittest.main()
