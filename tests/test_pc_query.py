import unittest
from scipy import stats
from collections import Counter
from probabilistic_circuits.pc_nodes import PCSum, PCProduct, ValueLeaf, OffsetLeaf
from probabilistic_circuits import pc_query

import logging
logger = logging.getLogger(__name__)


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


class TestQueries(unittest.TestCase):
    def test_inference(self):
        """Tests the functionality of the method inference."""
        pc = get_example_pc()

        self.assertEqual(1.0, pc_query.inference(pc, {}))
        self.assertEqual(0.15, pc_query.inference(pc, {"car": "BMW"}))
        self.assertEqual(0.20, pc_query.inference(pc, {"car": "Mercedes"}))
        self.assertEqual(0.15, pc_query.inference(pc, {"car": "VW"}))
        self.assertEqual(0.25, pc_query.inference(pc, {"airplane": "Airbus"}))
        self.assertEqual(0.25, pc_query.inference(pc, {"airplane": "Boing"}))
        self.assertEqual(0.75, pc_query.inference(pc, {"equipment": "radio"}))

        self.assertEqual(0.075, pc_query.inference(pc, {"car": "BMW", "equipment": "radio"}))
        self.assertEqual(0.100, pc_query.inference(pc, {"car": "Mercedes", "equipment": "radio"}))
        self.assertEqual(0.075, pc_query.inference(pc, {"car": "VW", "equipment": "radio"}))
        self.assertEqual(0.250, pc_query.inference(pc, {"airplane": "Airbus", "equipment": "radio"}))
        self.assertEqual(0.250, pc_query.inference(pc, {"airplane": "Boing", "equipment": "radio"}))

        self.assertEqual(0.0, pc_query.inference(pc, {"car": "Porsche"}))
        self.assertEqual(0.0, pc_query.inference(pc, {"airplane": "Cessna"}))
        self.assertEqual(0.0, pc_query.inference(pc, {"car": "BMW", "airplane": "Boing"}))
        self.assertEqual(0.0, pc_query.inference(pc, {"car": "BMW", "airplane": "Cessna"}))
        self.assertEqual(0.0, pc_query.inference(pc, {"car": "Porsche", "airplane": "Boing"}))

    def test_sample(self):
        """Tests the functionality of the method sample."""

        def _count_samples(samples: list[dict[object, object]]) -> Counter:
            sample_strings = [", ".join(sorted([str(k) + "=" + str(v) for k, v in s.items()])) for s in samples]
            return Counter(sample_strings)

        def _check_p_value(ground_truth: int, counter: Counter, sample_identifier: str, alpha: float = 0.01):
            sample_count = counter[sample_identifier]
            cdf_value = stats.poisson(mu=ground_truth).cdf(sample_count)
            p_value = min([cdf_value, 1 - cdf_value])
            if p_value < alpha / 2:
                logger.warning("Computed p-value {} for {} (expected count={}, observed count={})"
                                .format(round(p_value, 10), sample_identifier, ground_truth, sample_count))
            self.assertTrue(p_value > 1.0e-10)

        pc = get_example_pc()

        counts = _count_samples(pc_query.sample(pc, evidence={}, n=10000))
        _check_p_value(750,  counts, 'car=BMW')
        _check_p_value(1000, counts, 'car=Mercedes')
        _check_p_value(750,  counts, 'car=VW')
        _check_p_value(2500, counts, 'airplane=Airbus, equipment=radio')
        _check_p_value(2500, counts, 'airplane=Boing, equipment=radio')
        _check_p_value(750,  counts, 'car=BMW, equipment=radio')
        _check_p_value(1000, counts, 'car=Mercedes, equipment=radio')
        _check_p_value(750,  counts, 'car=VW, equipment=radio')

        counts = _count_samples(pc_query.sample(pc, evidence={'car': 'BMW'}, n=10000))
        self.assertTrue(all(['car=BMW' in sample_name for sample_name in counts.keys()]))
        _check_p_value(5000, counts, 'car=BMW')
        _check_p_value(5000, counts, 'car=BMW, equipment=radio')

        counts = _count_samples(pc_query.sample(pc, evidence={'airplane': 'Airbus'}, n=1000))
        self.assertTrue(all(['airplane=Airbus' in sample_name for sample_name in counts.keys()]))
        self.assertEqual(1000, counts['airplane=Airbus, equipment=radio'])

        counts = _count_samples(pc_query.sample(pc, evidence={'equipment': 'radio'}, n=1000))
        self.assertTrue(all(['equipment=radio' in sample_name for sample_name in counts.keys()]))


if __name__ == '__main__':
    unittest.main()
