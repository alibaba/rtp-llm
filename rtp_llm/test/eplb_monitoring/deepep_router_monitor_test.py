# type: ignore
"""Real LL router/FusedMoe wiring with only the external transport stubbed."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    CombineForwardPayload,
    ExpertForwardPayload,
    ExpertTokensMetadata,
    FusedMoe,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepep_low_latency_router import (  # noqa: E402
    DeepEpLowLatencyRouter,
)
from rtp_llm.models_py.utils.expert_load_monitor import ExpertLoadContext


class _Transport:
    def __init__(self):
        self.destination_counts = torch.tensor([7, 3], device="cuda", dtype=torch.int32)
        self.source_ids = None
        self.source_x = None

    def low_latency_dispatch(self, **kwargs):
        self.source_ids = kwargs["topk_idx"]
        self.source_x = kwargs["x"]
        # Destination load intentionally differs from source routing histogram.
        x = torch.zeros((2, 16, 2048), dtype=kwargs["x"].dtype, device="cuda")
        return x, self.destination_counts, ("handle",), None, None

    def low_latency_combine(self, **kwargs):
        return self.source_x, None, None


class _Executor:
    execute_empty_inputs = True

    def execute(self, payload, **kwargs):
        self.expert_load = payload.expert_load
        return CombineForwardPayload(payload.expert_x)


def _router(transport):
    router = object.__new__(DeepEpLowLatencyRouter)
    router.config = SimpleNamespace(
        tp_size=1,
        tp_rank=0,
        ep_size=2,
        ep_rank=1,
        expert_num=4,
        physical_expert_num=4,
    )
    router.quant_config = SimpleNamespace(
        is_quantized=False, is_block_quantized=False, is_per_group_fp4=False
    )
    router.tp_size = 1
    router._num_experts = 4
    router._num_topk = 2
    router._num_max_dispatch_tokens_per_rank = 128
    router._use_fp8_dispatch = False
    router._zero_copy = False
    router._async_finish = False
    router._return_recv_hook = False
    router._handle = None
    router._buffer = transport
    return router


def _context(expert_token_counts):
    return ExpertLoadContext(expert_token_counts)


class DeepEpDestinationMonitorTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available(), "GPU target requires CUDA")

    def test_received_counts_accumulate_until_step_reset(self):
        transport = _Transport()
        layer = FusedMoe(_router(transport), _Executor(), 4)
        counts = torch.zeros((2,), device="cuda", dtype=torch.int32)
        context = _context(counts)
        ids = torch.tensor([[0, 0]], device="cuda", dtype=torch.int64)
        weights = torch.ones_like(ids, dtype=torch.float32)
        x = torch.zeros((1, 2048), device="cuda", dtype=torch.bfloat16)
        layer(x, weights, ids, expert_load=context)
        self.assertEqual(counts.cpu().tolist(), [7, 3])
        transport.destination_counts[0] = 0
        transport.destination_counts[1] = 5
        layer(x, weights, ids, expert_load=context)
        self.assertEqual(counts.cpu().tolist(), [7, 8])
        counts.zero_()
        layer(x, weights, ids, expert_load=context)
        self.assertEqual(counts.cpu().tolist(), [0, 5])
        self.assertEqual(transport.source_ids.cpu().tolist(), [[0, 0]])

    def test_empty_source_still_records_work_received_from_other_ranks(self):
        transport = _Transport()
        layer = FusedMoe(_router(transport), _Executor(), 4)
        counts = torch.zeros((2,), device="cuda", dtype=torch.int32)
        context = _context(counts)
        ids = torch.empty((0, 2), device="cuda", dtype=torch.int64)
        layer(
            torch.empty((0, 2048), device="cuda", dtype=torch.bfloat16),
            torch.empty_like(ids, dtype=torch.float32),
            ids,
            expert_load=context,
        )
        self.assertEqual(counts.cpu().tolist(), [7, 3])

    def test_chunks_sum_destination_loads(self):
        transport = _Transport()
        layer = FusedMoe(_router(transport), _Executor(), 4)
        counts = torch.zeros((2,), device="cuda", dtype=torch.int32)
        context = _context(counts)
        ids = torch.tensor([[0, 1]], device="cuda", dtype=torch.int64)
        weights = torch.ones_like(ids, dtype=torch.float32)
        x = torch.zeros((1, 2048), device="cuda", dtype=torch.bfloat16)
        layer(x, weights, ids, expert_load=context)
        transport.destination_counts.copy_(torch.tensor([2, 8], device="cuda"))
        layer(x, weights, ids, expert_load=context)
        self.assertEqual(counts.cpu().tolist(), [9, 11])

    def test_dispatch_counts_take_precedence_over_executor_counts(self):
        transport = _Transport()
        executor = _Executor()
        layer = FusedMoe(_router(transport), executor, 4)
        counts = torch.zeros((2,), device="cuda", dtype=torch.int32)
        ids = torch.tensor([[0, 0]], device="cuda", dtype=torch.int64)
        layer(
            torch.zeros((1, 2048), device="cuda", dtype=torch.bfloat16),
            torch.ones_like(ids, dtype=torch.float32),
            ids,
            expert_load=_context(counts),
        )
        self.assertEqual(counts.cpu().tolist(), [7, 3])
        self.assertIsNone(executor.expert_load)

    def test_graph_replay_reads_current_destination_counts(self):
        transport = _Transport()
        layer = FusedMoe(_router(transport), _Executor(), 4)
        counts = torch.zeros((2,), device="cuda", dtype=torch.int32)
        ids = torch.tensor([[0, 0]], device="cuda", dtype=torch.int64)
        weights = torch.ones_like(ids, dtype=torch.float32)
        x = torch.zeros((1, 2048), device="cuda", dtype=torch.bfloat16)
        context = _context(counts)
        layer(x, weights, ids, expert_load=context)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            layer(x, weights, ids, expert_load=context)
        for expected in ([1, 9], [0, 0], [5, 2]):
            transport.destination_counts.copy_(
                torch.tensor(expected, device="cuda", dtype=torch.int32)
            )
            for _ in range(2):
                counts.zero_()
                graph.replay()
                self.assertEqual(counts.cpu().tolist(), expected)

    def test_fp4_executor_counts_local_ids_without_dispatch_padding(self):
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.trtllm_fp4_executor import (
            TrtllmFp4Executor,
        )

        executor = object.__new__(TrtllmFp4Executor)
        executor.global_num_experts = 4
        executor.w1 = torch.empty((2, 2, 4), dtype=torch.uint8, device="cuda")
        executor.w2 = torch.empty((2, 8, 1), dtype=torch.uint8, device="cuda")
        executor.w1_scale = executor.w2_scale = torch.ones(
            (2, 1), dtype=torch.uint8, device="cuda"
        )
        executor.expert_x_scale = torch.ones((3, 1), dtype=torch.uint8, device="cuda")
        executor.g1_scale_c = executor.g1_alphas = executor.g2_alphas = None
        executor._enable_pdl = False
        x = torch.zeros((3, 8), dtype=torch.uint8, device="cuda")
        ids = torch.tensor([[0, 1], [1, -1], [0, -1]], device="cuda")
        weights = torch.ones_like(ids, dtype=torch.float32)
        router = Mock()
        router.prepare.side_effect = lambda *args: ExpertForwardPayload(
            expert_x=x,
            expert_x_scale=executor.expert_x_scale,
            expert_topk_ids=ids,
            expert_topk_weights=weights,
            expert_tokens_meta=ExpertTokensMetadata(
                expert_num_tokens=torch.full(
                    (2,), 128, dtype=torch.int32, device="cuda"
                ),
                counts_are_padded=True,
            ),
        )
        router.finalize.side_effect = lambda payload, *args: payload.fused_expert_output
        layer = FusedMoe(router, executor, 4)
        counts = torch.zeros((2,), dtype=torch.int32, device="cuda")
        context = ExpertLoadContext(counts)
        with patch(
            "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors."
            "trtllm_fp4_executor.trtllm_fp4_block_scale_routed_moe",
            return_value=[x],
        ):
            layer(x, weights, ids, expert_load=context)
            self.assertEqual(counts.cpu().tolist(), [2, 2])
            counts.zero_()
            layer(x, weights, ids, expert_load=context)
            layer(x, weights, ids, expert_load=context)
            self.assertEqual(counts.cpu().tolist(), [4, 4])

    def test_scatter_records_assignments_without_expert_alignment_padding(self):
        from rtp_llm.models_py.triton_kernels.moe.ep_kernels import (
            ep_scatter,
            ep_scatter_v2,
        )

        x = torch.ones((3, 128), dtype=torch.bfloat16, device="cuda")
        scale = torch.ones((3, 1), device="cuda")
        ids = torch.tensor([[0, 1], [-1, 1], [0, -1]], device="cuda")
        padded_counts = torch.tensor([128, 128], dtype=torch.int32, device="cuda")
        starts = torch.empty_like(padded_counts)
        output = torch.empty((256, 128), dtype=x.dtype, device="cuda")
        output_scale = torch.empty((256, 1), device="cuda")
        expert_ids = torch.empty((256,), dtype=torch.int32, device="cuda")
        output_index = torch.empty_like(ids)
        counts = torch.zeros((2,), dtype=torch.int32, device="cuda")

        for version in (1, 2):
            with self.subTest(scatter_version=version):

                def scatter():
                    if version == 1:
                        ep_scatter(
                            x,
                            scale,
                            ids,
                            padded_counts,
                            starts,
                            output,
                            output_scale,
                            expert_ids,
                            output_index,
                            expert_token_counts=counts,
                        )
                    else:
                        ep_scatter_v2(
                            x,
                            scale,
                            ids,
                            128,
                            starts,
                            output,
                            output_scale.view(2, 128, 1),
                            output_index,
                            expert_token_counts=counts,
                        )

                ids.copy_(torch.tensor([[0, 1], [-1, 1], [0, -1]], device="cuda"))
                counts.zero_()
                scatter()
                self.assertEqual(counts.cpu().tolist(), [2, 2])
                scatter()
                self.assertEqual(counts.cpu().tolist(), [4, 4])
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    scatter()
                for value, expected in ((-1, [0, 0]), (0, [6, 0]), (1, [0, 6])):
                    ids.fill_(value)
                    for _ in range(2):
                        counts.zero_()
                        graph.replay()
                        self.assertEqual(counts.cpu().tolist(), expected)


if __name__ == "__main__":
    unittest.main()
