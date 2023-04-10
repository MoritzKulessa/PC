import unittest
from probabilistic_circuits.pc_nodes import OffsetLeaf, ValueLeaf


class TestLeafNodes(unittest.TestCase):
    def test_OffsetLeaf(self):
        """Tests if the implemented functions of OffsetLeaf work."""

        # Create leaf
        offset_leaf = OffsetLeaf()

        # Check inference
        self.assertEqual(1.0, offset_leaf.inference({"a": 100, "b": 200, 10: "c"}))
        self.assertEqual(1.0, offset_leaf.inference({}))

        # Check sample
        sample = offset_leaf.sample()
        self.assertTrue(len(sample) == 0 and isinstance(sample, dict))

        # Check the maximum probability/density
        self.assertEqual(1.0, offset_leaf.max_inference())

        # Check the most likely value
        mpe = offset_leaf.mpe()
        self.assertTrue(len(mpe) == 0 and isinstance(mpe, dict))

    def test_ValueLeaf(self):
        """Tests if the implemented functions of ValueLeaf work."""

        # Create leaf
        value_leaf = ValueLeaf(scope={"car"}, value="BMW")

        # Check inference
        self.assertEqual(1.0, value_leaf.inference({"a": 100, "car": "BMW", 10: "c"}))
        self.assertEqual(0.0, value_leaf.inference({"a": 100, "car": "Mercedes", 10: "c"}))
        self.assertEqual(0.0, value_leaf.inference({}))

        # Check sample
        sample = value_leaf.sample()
        self.assertTrue(len(sample) == 1 and isinstance(sample, dict) and sample["car"] == "BMW")

        # Check the maximum probability/density
        self.assertEqual(1.0, value_leaf.max_inference())

        # Check the most likely value
        mpe = value_leaf.mpe()
        self.assertTrue(
            len(mpe) == 1 and
            isinstance(mpe, dict) and
            len(mpe["car"]) == 1 and
            mpe["car"][0] == "BMW"
        )


if __name__ == '__main__':
    unittest.main()
