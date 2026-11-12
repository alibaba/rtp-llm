import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.eplb.runtime import EplbLayer
from rtp_llm.eplb.test.test_support import make_layer
from rtp_llm.models_py.modules.glm5_mega_moe import mega_gate


class Glm5MegaGateTest(unittest.TestCase):
    def config(self):
        return SimpleNamespace(
            moe_k=8,
            moe_n_group=1,
            moe_topk_group=1,
            has_moe_norm=True,
            routed_scaling_factor=2.5,
        )

    def test_borrows_maps_and_output_storage(self):
        plan = make_layer()
        ids = torch.empty((1, 8), dtype=torch.int64)
        weights = torch.empty((1, 8))
        kernel = Mock()
        with patch.object(mega_gate, "_native_gate", return_value=kernel):
            actual = mega_gate.mega_gate(
                torch.empty(1, 256),
                torch.empty(4, 256),
                torch.empty(4),
                self.config(),
                plan,
                out=(ids, weights),
            )
        self.assertIs(actual[0], weights)
        self.assertIs(actual[1], ids)
        self.assertIs(
            kernel.call_args.kwargs["to_physical_map"], plan.logical_to_physical
        )
        self.assertIs(kernel.call_args.kwargs["logical_count"], plan.logical_count)
        self.assertEqual(kernel.call_args.kwargs["scoring_func"], "sigmoid")

    @torch.inference_mode()
    def test_native_eager_and_graph_mapping_matches_reference(self):
        self.assertTrue(torch.cuda.is_available())
        self.assertEqual(torch.cuda.get_device_capability()[0], 10)
        self.assertIsNotNone(mega_gate._native_gate())
        torch.manual_seed(739)
        config = self.config()
        plan = make_layer(
            tuple(range(256)) + tuple(range(32)) * 2, device="cuda", replica_offset=7
        )
        w = torch.randn(256, 6144, device="cuda", dtype=torch.bfloat16) * 0.01
        bias = torch.randn(256, device="cuda", dtype=torch.float32) * 0.01
        for rows in (1, 17, 256, 8193):
            with self.subTest(rows=rows):
                x = torch.randn(rows, 6144, device="cuda", dtype=torch.bfloat16)
                self.assertTrue(mega_gate.is_supported(x, w, bias, config, plan))
                logical_ids = torch.empty(rows, 8, dtype=torch.int64, device="cuda")
                reference_weights = torch.empty(rows, 8, device="cuda")
                mega_gate._native_gate()(
                    x,
                    w,
                    8,
                    use_shared_as_routed=False,
                    num_shared_experts=0,
                    routed_scaling_factor=2.5,
                    ep_rank=7,
                    scoring_func="sigmoid",
                    bias=bias,
                    to_physical_map=None,
                    logical_count=None,
                    out=(logical_ids, reference_weights),
                )
                weights, ids = mega_gate.mega_gate(x, w, bias, config, plan)
                torch.testing.assert_close(
                    ids, plan.map_ids(logical_ids), rtol=0, atol=0
                )
                torch.testing.assert_close(weights, reference_weights, rtol=0, atol=0)
                scores = torch.sigmoid(x.float() @ w.float().T)
                expected = scores.gather(1, logical_ids)
                expected = expected / expected.sum(1, keepdim=True) * 2.5
                torch.testing.assert_close(weights, expected, rtol=2e-3, atol=2e-4)
                graph = torch.cuda.CUDAGraph()
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    mega_gate.mega_gate(x, w, bias, config, plan, out=(ids, weights))
                torch.cuda.current_stream().wait_stream(stream)
                with torch.cuda.graph(graph, stream=stream):
                    mega_gate.mega_gate(x, w, bias, config, plan, out=(ids, weights))
                for offset in (3, 11):
                    plan.logical_to_physical.add_(offset)
                    graph.replay()
                    torch.testing.assert_close(
                        ids, plan.map_ids(logical_ids), rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        weights, reference_weights, rtol=0, atol=0
                    )
                graph.reset()

    @torch.inference_mode()
    def test_graph_reads_updated_replica_counts_across_ranks(self):
        self.assertTrue(torch.cuda.is_available())
        self.assertEqual(torch.cuda.get_device_capability()[0], 10)
        self.assertIsNotNone(mega_gate._native_gate())
        torch.manual_seed(20261028)
        experts, redundant = 384, 64
        layouts = [
            make_layer(
                tuple(
                    torch.cat(
                        (
                            torch.arange(experts),
                            experts - 1 - torch.arange(redundant) // copies,
                        )
                    )
                    .roll(shift)
                    .tolist()
                ),
                device="cuda",
            )
            for copies, shift in ((1, 7), (4, 19), (2, 31))
        ]
        config = self.config()
        w = torch.randn(experts, 6144, device="cuda", dtype=torch.bfloat16) * 0.01
        bias = torch.arange(experts, device="cuda", dtype=torch.float32) * 4
        for rank in (0, 3, 31):
            for rows in (1, 32, 192):
                with self.subTest(rank=rank, rows=rows):
                    plan = EplbLayer(
                        0,
                        layouts[0].logical_to_physical.clone(),
                        layouts[0].logical_count.clone(),
                        layouts[0].weights,
                        replica_offset=rank,
                    )
                    x = torch.randn(rows, 6144, device="cuda", dtype=torch.bfloat16)
                    reference_weights, logical = mega_gate.mega_gate(
                        x, w, bias, config, None
                    )
                    weights, ids = mega_gate.mega_gate(x, w, bias, config, plan)
                    tracked = (
                        plan.logical_to_physical,
                        plan.logical_count,
                        weights,
                        ids,
                    )
                    pointers = tuple(t.data_ptr() for t in tracked)
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        mega_gate.mega_gate(
                            x, w, bias, config, plan, out=(ids, weights)
                        )
                    torch.cuda.current_stream().wait_stream(stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        mega_gate.mega_gate(
                            x, w, bias, config, plan, out=(ids, weights)
                        )
                    previous = None
                    for layout in layouts:
                        plan.logical_to_physical.copy_(layout.logical_to_physical)
                        plan.logical_count.copy_(layout.logical_count)
                        graph.replay()
                        torch.testing.assert_close(
                            ids, plan.map_ids(logical), rtol=0, atol=0
                        )
                        torch.testing.assert_close(
                            weights, reference_weights, rtol=0, atol=0
                        )
                        if previous is not None:
                            self.assertFalse(torch.equal(ids, previous))
                        previous = ids.clone()
                        self.assertEqual(pointers, tuple(t.data_ptr() for t in tracked))
                    graph.reset()

    def test_disabled_and_unsupported_preserve_old_path(self):
        x = torch.empty(1, 256)
        self.assertFalse(mega_gate.is_supported(x, None, None, self.config(), None))
        with patch.dict("os.environ", {"RTP_LLM_GLM5_MEGA_GATE": "0"}):
            self.assertFalse(mega_gate.is_supported(x, x, x, self.config(), object()))


if __name__ == "__main__":
    unittest.main()
