import importlib.util
import unittest
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "scripts/build.py"
spec = importlib.util.spec_from_file_location("awesome_build", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class RankingTests(unittest.TestCase):
    def test_dangling_graph_preserves_probability(self):
        result = module.pagerank([1, 2, 3], [], {1: 1.0, 2: 2.0, 3: 1.0})
        self.assertAlmostEqual(sum(result.values()), 1.0)
        self.assertAlmostEqual(result[2], 0.5)
        self.assertAlmostEqual(result[1], result[3])

    def test_reference_transfers_rank_to_target(self):
        result = module.pagerank([1, 2], [(1, 2)], {1: 1.0, 2: 1.0})
        self.assertGreater(result[2], result[1])
        self.assertAlmostEqual(sum(result.values()), 1.0)

    def test_duplicate_self_and_unknown_edges_cannot_add_votes(self):
        ordinary = module.pagerank([1, 2], [(1, 2)], {1: 1.0, 2: 1.0})
        polluted = module.pagerank([2, 1, 1], [(1, 2), (1, 2), (1, 1), (1, 99), (99, 2)], {1: 1.0, 2: 1.0})
        self.assertEqual(ordinary, polluted)

    def test_held_source_cannot_vote_in_approved_graph(self):
        baseline = module.pagerank([1, 2], [(1, 2)], {1: 1.0, 2: 1.0})
        with_held = module.pagerank([1, 2], [(1, 2), (3, 1)], {1: 1.0, 2: 1.0, 3: 100.0})
        self.assertEqual(baseline, with_held)

    def test_quality_uses_health_and_original_completeness(self):
        record = {"link_status": "unchecked", "quality_components": {"tagline": True, "description_at_least_100_characters": True, "sources": False, "subcategory": False}}
        health, complete, q = module.quality(record)
        self.assertEqual((health, complete), (0.2, 0.5))
        self.assertAlmostEqual(q, 0.29)


if __name__ == "__main__":
    unittest.main()
