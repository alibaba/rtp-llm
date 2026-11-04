#!/usr/bin/env python3

import importlib.util
import pathlib
import unittest


MODULE_PATH = pathlib.Path(__file__).with_name("flexlb_event_baseline.py")
SPEC = importlib.util.spec_from_file_location("flexlb_event_baseline", MODULE_PATH)
baseline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(baseline)


PARAMETERS = [
    -957.8349869923576, 19.479887335226252, -0.21753212616946724,
    64.56275349888392, 46.031287293423574, 16.516869607425917,
    0.0, -0.3010100112650802, 3.7871819914802636,
    -0.20672180219383884, 0.0,
]


class FlexLBEventBaselineTest(unittest.TestCase):
    def make_config(self, node_count=1):
        return baseline.Config(
            traffic_log="unused", output="unused", node_count=node_count,
            formula_parameters=PARAMETERS, fixed_window_ms=50.0,
            block_size=1024, steady_discard_fraction=0.0)

    def test_node_local_windows_are_independent(self):
        simulator = baseline.FlexLBEventSimulator(self.make_config(node_count=2))
        requests = [
            baseline.Request(0, "a", 0.0, 1024, ("shared",)),
            baseline.Request(1, "b", 10.0, 1024, ("other",)),
        ]
        summary = simulator.run(requests)
        self.assertEqual(2, summary["result"]["request_count"])
        self.assertEqual(2, summary["result"]["batch_count"])
        self.assertEqual([1, 1], sorted(
            item["batch_size"] for item in summary["batches"]))
        self.assertEqual([50.0, 60.0], sorted(
            item["dispatch_ms"] for item in summary["batches"]))

    def test_batch_full_dispatches_without_waiting(self):
        config = self.make_config()
        config.batch_size_max = 2
        simulator = baseline.FlexLBEventSimulator(config)
        requests = [
            baseline.Request(0, "a", 0.0, 256, ()),
            baseline.Request(1, "b", 5.0, 256, ()),
        ]
        summary = simulator.run(requests)
        self.assertEqual(1, summary["result"]["batch_count"])
        self.assertEqual(2.0, summary["result"]["average_batch_size"])
        self.assertEqual(5.0, summary["batches"][0]["dispatch_ms"])
        self.assertEqual("batch_full", summary["batches"][0]["dispatch_reason"])

    def test_full_cache_hit_keeps_one_block_miss(self):
        self.assertEqual(1024, baseline.bounded_hit_tokens(2048, 2, 1024))
        self.assertEqual(1024, baseline.bounded_hit_tokens(2048, 1, 1024))

    def test_total_prompt_parameter_is_outside_nonlinear_term(self):
        request = baseline.Request(0, "a", 0.0, 2048, ())
        without_total = baseline.nonlinear_latency_ms([request], PARAMETERS)
        parameters = list(PARAMETERS)
        parameters[10] = 2.0
        with_total = baseline.nonlinear_latency_ms([request], parameters)
        self.assertAlmostEqual(4.0, with_total - without_total)

    def test_timestamp_ms_preserves_sub_millisecond_precision(self):
        self.assertEqual(
            1787475371864.089,
            baseline.record_timestamp_ms({"timestamp_ms": 1787475371864.089}))
        self.assertEqual(1500.0, baseline.record_timestamp_ms({"timestamp": 1.5}))

    def test_forced_node_bypasses_selector(self):
        simulator = baseline.FlexLBEventSimulator(self.make_config(node_count=2))
        request = baseline.Request(0, "a", 0.0, 1024, (), forced_node_index=1)
        summary = simulator.run([request])
        self.assertEqual([0, 1], summary["result"]["node_request_counts"])
        self.assertEqual(1, summary["configuration_audit"][
            "forced_node_request_count"])

    def test_per_node_parameters_override_cluster_default(self):
        config = self.make_config(node_count=2)
        config.formula_parameters_by_node = [
            PARAMETERS, [value + 1.0 for value in PARAMETERS]]
        simulator = baseline.FlexLBEventSimulator(config)
        self.assertEqual(PARAMETERS, simulator.node_parameters(0))
        self.assertEqual(config.formula_parameters_by_node[1],
                         simulator.node_parameters(1))

    def test_engine_dynamic_batches_requests_queued_behind_running_batch(self):
        config = self.make_config()
        config.batching_mode = "engine_dynamic"
        simulator = baseline.FlexLBEventSimulator(config)
        requests = [
            baseline.Request(0, "a", 0.0, 1024, ()),
            baseline.Request(1, "b", 1.0, 1024, ()),
            baseline.Request(2, "c", 2.0, 1024, ()),
        ]
        summary = simulator.run(requests)
        self.assertEqual(2, summary["result"]["batch_count"])
        self.assertEqual([1, 2], [
            item["batch_size"] for item in summary["batches"]])
        self.assertEqual("engine_dynamic", summary["batches"][1]["dispatch_reason"])

    def test_engine_dynamic_inflight_tracking_is_explicit(self):
        config = self.make_config()
        config.batching_mode = "engine_dynamic"
        simulator = baseline.FlexLBEventSimulator(config)
        request = baseline.Request(0, "a", 0.0, 1024, ())
        node = simulator.nodes[0]
        simulator.dispatch_dynamic_request(node, request, 0.0)
        self.assertEqual([], node.inflight)

        config.selector_inflight_tracking = True
        simulator = baseline.FlexLBEventSimulator(config)
        request = baseline.Request(0, "a", 0.0, 1024, ())
        node = simulator.nodes[0]
        simulator.dispatch_dynamic_request(node, request, 0.0)
        self.assertEqual(1, len(node.inflight))

    def test_cache_status_snapshot_does_not_invent_pending_feedback(self):
        config = self.make_config()
        config.batching_mode = "engine_dynamic"
        config.cache_status_interval_mode = "fixed"
        config.cache_status_initial_interval_ms = 50.0
        simulator = baseline.FlexLBEventSimulator(config)
        node = simulator.nodes[0]
        node.update_cache(("a",))
        node.execution_queue.append(baseline.Request(0, "a", 0.0, 1024, ("a",)))
        simulator.refresh_cache_status(0.0)
        self.assertEqual({"a"}, node.reported_cache)
        self.assertEqual(0, node.reported_pending_count)
        node.update_cache(("b",))
        simulator.refresh_cache_status(49.0)
        self.assertNotIn("b", node.reported_cache)
        simulator.refresh_cache_status(50.0)
        self.assertIn("b", node.reported_cache)

    def test_flexlb_dynamic_cache_interval_uses_diff_history(self):
        config = self.make_config()
        config.cache_status_interval_mode = "flexlb_dynamic"
        config.cache_status_initial_interval_ms = 100.0
        simulator = baseline.FlexLBEventSimulator(config)
        simulator.update_dynamic_cache_interval(0)
        simulator.update_dynamic_cache_interval(0)
        self.assertEqual(100.0, simulator.cache_status_interval_ms)
        simulator.update_dynamic_cache_interval(0)
        self.assertEqual(130, simulator.cache_status_interval_ms)


if __name__ == "__main__":
    unittest.main()
