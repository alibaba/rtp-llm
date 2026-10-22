import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from eplb_runtime_test_utils import FakeModel, fake_runtime_module

from rtp_llm.cpp.models.eplb.test.libth_eplb_py_wrapper_test import (
    ExpertBalancerStepTest,
)


class TestExpertBalancerStep(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available(), "requires the assigned CUDA GPU")
        inference_mode = torch.inference_mode()
        inference_mode.__enter__()
        self.addCleanup(inference_mode.__exit__, None, None, None)
        self.directory = tempfile.TemporaryDirectory(dir=os.environ.get("TEST_TMPDIR"))
        self.addCleanup(self.directory.cleanup)
        log_path = patch.dict(os.environ, {"LOG_PATH": self.directory.name})
        log_path.start()
        self.addCleanup(log_path.stop)
        runtime_module = fake_runtime_module()
        runtime_module.start()
        self.addCleanup(runtime_module.stop)

    def balancer(self, model=None, interval=2, model_id=0, **kwargs):
        balancer = ExpertBalancerStepTest(model, 100, interval, model_id, **kwargs)
        self.addCleanup(balancer.destroy)
        return balancer

    def snapshots(self):
        return sorted(
            (
                json.loads(path.read_text())
                for path in (Path(self.directory.name) / "eplb_snapshot").glob("*.json")
            ),
            key=lambda window: window["step_end"],
        )

    def test_disabled_balancer_has_no_counts_and_step_is_inert(self):
        balancer = self.balancer(interval=0)
        self.assertIsNone(balancer.counts)
        balancer.step(warmup=True)
        for _ in range(3):
            balancer.step()
        self.assertEqual(self.snapshots(), [])
        self.assertTrue(balancer.update_config(mode=0))
        self.assertFalse(balancer.update_config(mode=2))

    def test_dense_model_has_no_writer_and_reports_empty_window(self):
        balancer = self.balancer(moe=False)
        self.assertIsNone(balancer.counts)
        balancer.step()
        balancer.step()
        [window] = self.snapshots()
        self.assertEqual(window["rank_samples"], [[[0], [0]]])
        self.assertEqual(window["moe_lb"], 1)
        self.assertEqual(window["total_activations"], 0)

    def test_replicated_tp_only_rank_zero_exposes_writer(self):
        writer = self.balancer(tp_size=2, tp_rank=0)
        replica = self.balancer(tp_size=2, tp_rank=1)
        self.assertEqual(tuple(writer.counts.shape), (1, 2))
        self.assertIsNone(replica.counts)

    def test_runtime_has_physical_counts_with_monitoring_disabled(self):
        model = FakeModel()
        balancer = self.balancer(model, interval=0, model_id=1)
        counts = balancer.counts
        pointer = counts.data_ptr()
        self.assertEqual(counts.dtype, torch.int32)
        self.assertEqual(tuple(counts.shape), (1, 2))
        self.assertEqual(model.state.identity, (0, 1, 1))
        for value in (2, 5):
            counts.fill_(value)
            balancer.step()
            self.assertEqual(counts.tolist(), [[0, 0]])
        self.assertEqual(
            [x.tolist() for x in model.state.samples], [[[2, 2]], [[5, 5]]]
        )
        self.assertEqual(model.state.sample_pointers, [pointer, pointer])
        self.assertEqual(self.snapshots(), [])

    def test_runtime_reference_is_released_when_destroyed_without_gil(self):
        model = FakeModel()
        balancer = self.balancer(model, interval=0)
        reference = model.state.runtime
        self.assertIsNotNone(reference())
        balancer.destroy()
        self.assertIsNone(reference())
        self.assertTrue(model.state.destroyed)

    def test_initial_stats_samples_without_migration_then_enables_all(self):
        model = FakeModel()
        model.state.apply_on_calls = {1, 2, 3}
        balancer = self.balancer(model, interval=0, model_id=1, mode=1)
        self.assertEqual(model.state.options["eplb_mode"], 1)
        self.assertEqual(model.state.identity, (0, 1, 1))
        counts = balancer.counts
        pointer = counts.data_ptr()
        for value in (2, 5):
            counts.fill_(value)
            balancer.step()
            self.assertEqual(counts.tolist(), [[0, 0]])
        self.assertEqual(
            [sample.tolist() for sample in model.state.samples],
            [[[2, 2]], [[5, 5]]],
        )
        self.assertEqual(model.state.applied, 0)
        self.assertEqual(self.snapshots(), [])

        self.assertTrue(balancer.update_config(mode=3))
        counts.fill_(7)
        balancer.step()
        self.assertEqual(
            model.state.calls[-3:], [("update", 3, 1), "record", "balance"]
        )
        self.assertEqual(model.state.samples[-1].tolist(), [[7, 7]])
        self.assertEqual(model.state.applied, 1)
        self.assertEqual(counts.tolist(), [[0, 0]])
        self.assertEqual(balancer.counts.data_ptr(), pointer)
        reference = model.state.runtime
        balancer.destroy()
        self.assertIsNone(reference())
        self.assertTrue(model.state.destroyed)

    def test_monitor_only_records_once_and_resets_unwritten_next_step(self):
        balancer = self.balancer(model_id=1)
        self.assertTrue(balancer.update_config(mode=0))
        counts = balancer.counts
        pointer = counts.data_ptr()
        counts.copy_(torch.tensor([[2, 3]], device="cuda"))
        balancer.step()
        self.assertEqual(counts.tolist(), [[0, 0]])
        self.assertEqual(self.snapshots(), [])
        self.assertTrue(balancer.update_config(mode=0))
        balancer.step()

        [window] = self.snapshots()
        self.assertEqual(window["rank_samples"], [[[5], [0]]])
        self.assertEqual((window["step_begin"], window["step_end"]), (1, 2))
        self.assertEqual((window["model_id"], window["model_type"]), (1, "draft"))
        self.assertEqual(balancer.counts.data_ptr(), pointer)
        self.assertFalse(balancer.update_config(mode=2))

    def test_runtime_samples_before_monitor_clears_and_balance_observes_clear(self):
        model = FakeModel()
        balancer = self.balancer(model)
        counts = balancer.counts
        pointer = counts.data_ptr()
        counts.copy_(torch.tensor([[2, 3]], device="cuda"))
        balancer.step()
        counts.copy_(torch.tensor([[5, 7]], device="cuda"))
        balancer.step()
        self.assertEqual(
            model.state.calls, ["get_layers", "record", "balance", "record", "balance"]
        )
        self.assertEqual(
            [x.tolist() for x in model.state.samples], [[[2, 3]], [[5, 7]]]
        )
        self.assertEqual(
            [x.tolist() for x in model.state.balance_counts], [[[0, 0]], [[0, 0]]]
        )
        self.assertEqual(model.state.sample_pointers, [pointer, pointer])
        [window] = self.snapshots()
        self.assertEqual(window["rank_samples"], [[[5], [12]]])

    def test_warmup_discards_counts_without_runtime_or_monitor_progress(self):
        model = FakeModel()
        balancer = self.balancer(model)
        counts = balancer.counts
        counts.copy_(torch.tensor([[1, 2]], device="cuda"))
        balancer.step()
        counts.fill_(99)
        balancer.step(warmup=True)
        self.assertEqual(model.state.calls, ["get_layers", "record", "balance"])
        self.assertEqual(counts.tolist(), [[0, 0]])
        self.assertEqual(self.snapshots(), [])
        counts.copy_(torch.tensor([[3, 4]], device="cuda"))
        balancer.step()

        [window] = self.snapshots()
        self.assertEqual(window["rank_samples"], [[[3], [7]]])
        self.assertEqual((window["step_begin"], window["step_end"]), (1, 2))
        self.assertEqual(
            [x.tolist() for x in model.state.samples], [[[1, 2]], [[3, 4]]]
        )

    def test_cuda_graph_capture_discard_and_replay_record_once(self):
        model = FakeModel()
        balancer = self.balancer(model)
        counts = balancer.counts
        pointer = counts.data_ptr()
        increments = torch.tensor([[2, 3]], dtype=torch.int32, device="cuda")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            counts.add_(increments)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            counts.add_(increments)
        graph.replay()
        balancer.step(warmup=True)
        self.assertEqual(model.state.calls, ["get_layers"])
        self.assertEqual(self.snapshots(), [])
        self.assertEqual(counts.tolist(), [[0, 0]])

        for _ in range(2):
            graph.replay()
            balancer.step()
            self.assertEqual(counts.tolist(), [[0, 0]])
            self.assertEqual(balancer.counts.data_ptr(), pointer)
        [window] = self.snapshots()
        self.assertEqual(window["rank_samples"], [[[5], [5]]])
        self.assertEqual((window["step_begin"], window["step_end"]), (1, 2))
        self.assertEqual(
            [x.tolist() for x in model.state.samples], [[[2, 3]], [[2, 3]]]
        )

    def test_noop_balance_preserves_partial_monitor_window(self):
        model = FakeModel()
        balancer = self.balancer(model, interval=4)
        for value in range(1, 5):
            balancer.counts.fill_(value)
            balancer.step()
        [window] = self.snapshots()
        self.assertEqual(window["mapping_epoch"], 0)
        self.assertEqual(window["rank_samples"], [[[2], [4], [6], [8]]])

    def test_mapping_change_discards_partial_window_and_keeps_count_storage(self):
        model = FakeModel()
        model.state.apply_on_calls = {2}
        balancer = self.balancer(model, interval=4)
        pointer = balancer.counts.data_ptr()
        for value in (1, 2):
            balancer.counts.fill_(value)
            balancer.step()
        self.assertEqual(model.state.applied, 1)
        self.assertEqual(self.snapshots(), [])
        for value in range(3, 7):
            balancer.counts.fill_(value)
            balancer.step()
        [window] = self.snapshots()
        self.assertEqual(window["mapping_epoch"], 1)
        self.assertEqual((window["step_begin"], window["step_end"]), (3, 6))
        self.assertEqual(window["rank_samples"], [[[6], [8], [10], [12]]])
        self.assertEqual(balancer.counts.data_ptr(), pointer)
        self.assertEqual(model.state.sample_pointers, [pointer] * 6)

    def test_dynamic_disable_is_applied_before_sampling_and_monitor_keeps_running(self):
        model = FakeModel()
        model.state.apply_on_calls = {2}
        balancer = self.balancer(model)
        balancer.counts.fill_(1)
        balancer.step()
        self.assertTrue(balancer.update_config(mode=0, update_time=5))
        balancer.counts.fill_(8)
        balancer.step(warmup=True)
        self.assertNotIn(("update", 0, 5), model.state.calls)
        balancer.counts.fill_(3)
        balancer.step()
        self.assertEqual(
            model.state.calls[-3:], [("update", 0, 5), "record", "balance"]
        )
        self.assertEqual([x.tolist() for x in model.state.samples], [[[1, 1]]])
        self.assertEqual(model.state.applied, 0)
        [window] = self.snapshots()
        self.assertEqual(window["rank_samples"], [[[2], [6]]])
        self.assertEqual(window["mapping_epoch"], 0)

    def test_invalid_dynamic_configuration_is_rejected(self):
        model = FakeModel()
        balancer = self.balancer(model, interval=0)
        self.assertFalse(balancer.update_config(mode=2, update_time=0))
        self.assertFalse(balancer.update_config(mode=-1))
        self.assertFalse(balancer.update_config(mode=4))
        balancer.counts.fill_(2)
        balancer.step()
        self.assertEqual(model.state.calls, ["get_layers", "record", "balance"])


if __name__ == "__main__":
    unittest.main()
