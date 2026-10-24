import json
import os
import tempfile
import unittest
import weakref
from pathlib import Path
from unittest.mock import patch

import torch

from rtp_llm.cpp.models.eplb.test.libth_eplb_py_wrapper_test import (
    ExpertBalancerStepTest,
)


class FakePlanner:
    def __init__(self):
        self.plans = []
        self.loads = []

    def create_balance_plan(self, log_stats, gpu_loads):
        self.plans.append((log_stats.clone(), gpu_loads.clone()))
        return (
            torch.tensor([0], dtype=torch.int32),
            torch.ones(2, dtype=torch.int32),
            torch.tensor([[1], [0]], dtype=torch.int32),
            torch.tensor([1, 0], dtype=torch.int32),
        )

    def load_moe_weight(self, layer_id, ep_rank, ep_size, phy2log):
        self.loads.append((layer_id.item(), ep_rank, ep_size, phy2log.clone()))
        return (
            0,
            torch.full((2, 2, 1), 7, dtype=torch.bfloat16),
            torch.full((2, 1, 1), 11, dtype=torch.bfloat16),
            torch.empty(0),
            torch.empty(0),
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

    def balancer(self, planner=None, interval=2, model_id=0, **kwargs):
        return ExpertBalancerStepTest(planner, 100, interval, model_id, **kwargs)

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
        self.assertTrue(balancer.disable_planner())

    def test_dense_model_has_no_writer_and_reports_empty_window(self):
        balancer = self.balancer(moe=False)
        self.assertIsNone(balancer.counts)
        balancer.step()
        balancer.step()
        [window] = self.snapshots()
        self.assertEqual(window["rank_samples"], [[[0], [0]]])
        self.assertEqual(window["moe_lb"], 1)
        self.assertEqual(window["total_activations"], 0)

    def test_dense_afd_does_not_start_monitoring(self):
        balancer = self.balancer(moe=False, afd=True)
        self.assertIsNone(balancer.counts)
        balancer.step(warmup=True)
        for _ in range(3):
            balancer.step()
        self.assertEqual(self.snapshots(), [])

    def test_replicated_tp_only_rank_zero_exposes_writer(self):
        writer = self.balancer(tp_size=2, tp_rank=0)
        replica = self.balancer(tp_size=2, tp_rank=1)
        self.assertEqual(tuple(writer.counts.shape), (1, 2))
        self.assertIsNone(replica.counts)

    def test_planner_runs_with_monitoring_disabled(self):
        planner = FakePlanner()
        balancer = self.balancer(planner, interval=0)
        self.assertIsNone(balancer.counts)
        balancer.step(warmup=True)
        for _ in range(3):
            balancer.step()
        self.assertEqual((len(planner.plans), len(planner.loads)), (1, 1))
        self.assertEqual(balancer.gate_weights.tolist(), [[[7], [7]], [[7], [7]]])
        self.assertEqual(self.snapshots(), [])

    def test_planner_reference_is_released_when_destroyed_without_gil(self):
        planner = FakePlanner()
        planner_ref = weakref.ref(planner)
        balancer = self.balancer(planner, interval=0)
        del planner
        self.assertIsNotNone(planner_ref())
        balancer.destroy()
        self.assertIsNone(planner_ref())

    def test_monitor_only_records_once_and_resets_unwritten_next_step(self):
        balancer = self.balancer(model_id=1)
        counts = balancer.counts
        pointer = counts.data_ptr()
        counts.copy_(torch.tensor([[2, 3]], device="cuda"))
        balancer.step()
        self.assertEqual(counts.tolist(), [[0, 0]])
        self.assertEqual(self.snapshots(), [])
        balancer.step()

        [window] = self.snapshots()
        self.assertEqual(window["rank_samples"], [[[5], [0]]])
        self.assertEqual((window["step_begin"], window["step_end"]), (1, 2))
        self.assertEqual((window["model_id"], window["model_type"]), (1, "draft"))
        self.assertEqual(balancer.counts.data_ptr(), pointer)
        self.assertFalse(balancer.disable_planner())

    def test_warmup_discards_counts_without_advancing_window(self):
        balancer = self.balancer()
        counts = balancer.counts
        counts.copy_(torch.tensor([[1, 2]], device="cuda"))
        balancer.step()
        counts.fill_(99)
        balancer.step(warmup=True)
        self.assertEqual(counts.tolist(), [[0, 0]])
        self.assertEqual(self.snapshots(), [])
        counts.copy_(torch.tensor([[3, 4]], device="cuda"))
        balancer.step()

        [window] = self.snapshots()
        self.assertEqual(window["rank_samples"], [[[3], [7]]])
        self.assertEqual((window["step_begin"], window["step_end"]), (1, 2))

    def test_cuda_graph_capture_discard_and_replay_record_once(self):
        balancer = self.balancer()
        counts = balancer.counts
        pointer = counts.data_ptr()
        increments = torch.tensor([[2, 3]], dtype=torch.int64, device="cuda")
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

    def test_planner_runs_once_after_real_steps_and_mapping_discards_partial_window(
        self,
    ):
        planner = FakePlanner()
        balancer = self.balancer(planner, interval=4)
        counts_pointer = balancer.counts.data_ptr()
        for _ in range(2):
            balancer.counts.fill_(99)
            balancer.step(warmup=True)
        self.assertEqual((len(planner.plans), len(planner.loads)), (0, 0))
        balancer.counts.fill_(1)
        balancer.step()  # INIT -> PREPARING
        balancer.step(warmup=True)
        self.assertEqual((len(planner.plans), len(planner.loads)), (0, 0))
        balancer.counts.fill_(2)
        balancer.step()  # create and load one plan
        self.assertEqual((len(planner.plans), len(planner.loads)), (1, 1))
        self.assertEqual(planner.plans[0][0].tolist(), [[2, 3]])
        self.assertEqual(planner.plans[0][1].tolist(), [[5]])
        self.assertEqual(planner.loads[0][:3], (0, 0, 1))
        balancer.counts.fill_(3)
        balancer.step()  # apply weights, discard the old mapping's partial window
        self.assertEqual(balancer.gate_weights.tolist(), [[[7], [7]], [[7], [7]]])
        self.assertEqual(balancer.log2phy.tolist(), [[1], [0]])
        self.assertEqual(balancer.counts.data_ptr(), counts_pointer)
        self.assertEqual(self.snapshots(), [])

        self.assertTrue(balancer.disable_planner())
        for value in range(4, 8):
            balancer.counts.fill_(value)
            balancer.step()
        [window] = self.snapshots()
        self.assertEqual(window["mapping_epoch"], 1)
        self.assertEqual((window["step_begin"], window["step_end"]), (4, 7))
        self.assertEqual(window["rank_samples"], [[[8], [10], [12], [14]]])
        self.assertEqual((len(planner.plans), len(planner.loads)), (1, 1))


if __name__ == "__main__":
    unittest.main()
