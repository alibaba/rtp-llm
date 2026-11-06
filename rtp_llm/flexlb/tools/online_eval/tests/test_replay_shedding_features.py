import array
import csv
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from replay_shedding_features import Feature, Request, Simulation, Worker, read_features


class FeatureReplayTest(unittest.TestCase):
    def test_prefix_graph_preserves_parent_relation_and_unique_suffix(self):
        fields = [
            "query_id",
            "best_prefix_query_id",
            "input_len",
            "output_len",
            "query_start_epoch_ms",
            "best_prefix_len_512_tokens",
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "features.csv"
            with path.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(fields)
                writer.writerow(["parent", "", 1536, 17, 0, 0])
                writer.writerow(["child", "parent", 2048, 31, 1000, 1024])
                writer.writerow(["missing", "unseen", 2048, 9, 2000, 1536])
            parent, child, missing = read_features(path)
            self.assertEqual(parent.keys[:2], child.keys[:2])
            self.assertNotEqual(parent.keys[2], child.keys[2])
            self.assertTrue(set(parent.keys).isdisjoint(missing.keys))
            self.assertEqual((child.input, child.output, child.at), (2048, 31, 1))

    def test_cache_is_worker_local_delayed_and_expires_at_twenty_minutes(self):
        feature = Feature(0, 1024, 10, array.array("q", [1, 2]), 2)
        request = Request(1, 0, feature, feature.keys, False)
        simulation = Simulation([feature], None, 1, 2, 1, "steady", pathlib.Path("."))
        worker = simulation.workers[0]
        worker.cache = {1: (0, 0.020), 2: (0, 0.020)}
        self.assertEqual(0, simulation.cache_hit(worker, request, 0.019))
        self.assertEqual(1024, simulation.cache_hit(worker, request, 0.020))
        self.assertEqual(0, simulation.cache_hit(simulation.workers[1], request, 1))
        self.assertEqual(1024, simulation.cache_hit(worker, request, 1199.999))
        self.assertEqual(0, simulation.cache_hit(worker, request, 1200))

    def test_refresh_preserves_visible_prefix_and_delays_only_new_suffix(self):
        feature = Feature(0, 1536, 10, array.array("q", [1, 2, 3]), 3)
        request = Request(1, 0, feature, feature.keys, False)
        simulation = Simulation([feature], None, 1, 1, 1, "steady", pathlib.Path("."))
        worker = simulation.workers[0]
        simulation.publish_cache(worker, feature.keys[:2], 0)
        self.assertEqual(1024, simulation.cache_hit(worker, request, 1))
        simulation.publish_cache(worker, feature.keys, 1)
        self.assertEqual(1024, simulation.cache_hit(worker, request, 1.019))
        self.assertEqual(1536, simulation.cache_hit(worker, request, 1.020))
        self.assertEqual(0, simulation.cache_hit(worker, request, 1201))
        simulation.publish_cache(worker, feature.keys, 1201)
        self.assertEqual(0, simulation.cache_hit(worker, request, 1201.019))
        self.assertEqual(1536, simulation.cache_hit(worker, request, 1201.020))

    def test_random_copy_step_only_removes_a_requests_and_preserves_identity(self):
        features = [
            Feature(i, 1024, 10, array.array("q", [1, i + 2]), 1) for i in range(200)
        ]

        def arrivals(scenario):
            sim = Simulation(
                features,
                None,
                17,
                2,
                2,
                scenario,
                pathlib.Path("."),
                base_qps=2,
                copy_step=100,
            )
            sim.plan()
            return {
                value[0].sample_id: (at, value[0].stream, list(value[2]))
                for at, _, kind, value in sim.events
                if kind == "arrival"
            }

        control, thinned = arrivals("mixed_control"), arrivals("mixed_thin")
        self.assertEqual(400, len(control))
        self.assertTrue(thinned.keys() < control.keys())
        for identity, value in control.items():
            at, stream, _ = value
            if at < 100 or stream == "b":
                self.assertEqual(value, thinned[identity])
            if identity in thinned:
                self.assertEqual(value, thinned[identity])
        self.assertTrue(set(control[1][2]).isdisjoint(control[201][2]))


if __name__ == "__main__":
    unittest.main()
