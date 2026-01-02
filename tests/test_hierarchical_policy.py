import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import unittest
from bloodcancer.agents.tools import RoutingTool, _ensemble_combine

class DummyCfg:
    class_names = ["normal", "leukemia"]

class TestHierarchicalPolicy(unittest.TestCase):
    def test_routing_hierarchical_policy_builds_stage2_plan(self):
        rt = RoutingTool(DummyCfg())
        out = rt.run({"policy":"two_stage_hierarchical", "image_path":"x"})
        self.assertEqual(out.get("mode"), "two_stage")
        st2 = out.get("stage2_plan")
        self.assertIsInstance(st2, dict)
        self.assertEqual(st2.get("mode"), "hierarchical")
        self.assertTrue(isinstance(st2.get("families"), list))

    def test_best_per_class_combine(self):
        outputs = [
            {"_model_id":"m1","probs":{"a":0.9,"b":0.1}},
            {"_model_id":"m2","probs":{"a":0.2,"b":0.8}},
        ]
        probs, conf, dis = _ensemble_combine(outputs, ["a","b"], strategy="best_per_class", class_to_model={"a":"m1","b":"m2"})
        self.assertGreater(probs["a"], probs["b"])
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=5)

if __name__ == "__main__":
    unittest.main()
