# type: ignore
"""Production destination-counter tests; this target requires a CUDA device."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    CombineForwardPayload,
    ExpertForwardPayload,
    FusedMoe,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_tp_router import (  # noqa: E402
    PureTpRouterNoQuant,
)
from rtp_llm.models_py.triton_kernels.moe.ep_kernels import (
    get_cutlass_moe_mm_without_permute_info,
)
from rtp_llm.models_py.utils.expert_load_monitor import (
    expert_load_for_layer,
    record_expert_load,
)


class _PassThroughExecutor:
    execute_empty_inputs = True
    includes_shared_expert = False
    topk_ids_dtype = torch.int64

    def __init__(self, config):
        self.config = config

    def execute(self, payload, **kwargs):
        expert_load = payload.expert_load
        if expert_load is not None:
            counts = expert_load.expert_token_counts
            ids = payload.expert_topk_ids - self.config.ep_rank * counts.numel()
            get_cutlass_moe_mm_without_permute_info(
                ids,
                torch.empty_like(counts, dtype=torch.int32),
                torch.empty((counts.numel(), 3), dtype=torch.int32, device=ids.device),
                torch.empty((counts.numel(), 3), dtype=torch.int32, device=ids.device),
                counts.numel(),
                8,
                8,
                False,
                False,
                expert_token_counts=counts,
            )
        return CombineForwardPayload(payload.expert_x)


def _make_router(ep_size=1, ep_rank=0, tp_size=1, tp_rank=0):
    config = SimpleNamespace(
        expert_num=4,
        physical_expert_num=4,
        ep_size=ep_size,
        ep_rank=ep_rank,
        tp_size=tp_size,
        tp_rank=tp_rank,
    )
    # Use the production constructor, without creating distributed state.
    return PureTpRouterNoQuant(config, SimpleNamespace())


def _make_inputs(layers=1, experts=4):
    return SimpleNamespace(
        expert_token_counts=torch.zeros(
            (layers, experts), dtype=torch.int32, device="cuda"
        )
    )


class _SyntheticModel(nn.Module):
    """Only model compute is synthetic; input views and routing are production."""

    def __init__(self, routers):
        nn.Module.__init__(self)
        self.layer_num = len(routers)
        self.config = SimpleNamespace(moe_layer_index=list(range(self.layer_num)))
        self.parallelism_config = SimpleNamespace(
            ep_size=routers[0].config.ep_size,
            tp_rank=routers[0].config.tp_rank,
            prefill_cp_config=SimpleNamespace(is_enabled=lambda: False),
        )
        self.layers = nn.ModuleList(
            [
                FusedMoe(router, _PassThroughExecutor(router.config), 4)
                for router in routers
            ]
        )

    def forward(self, ids, expert_token_counts):
        x = torch.zeros(ids.size(0), 8, dtype=torch.bfloat16, device=ids.device)
        weights = torch.ones_like(ids, dtype=torch.float32)
        for layer_id, layer in enumerate(self.layers):
            x = layer(
                x,
                weights,
                ids,
                expert_load=expert_load_for_layer(expert_token_counts, layer_id),
            )
        return x


class RecorderDisabledTest(unittest.TestCase):
    def test_disabled_has_no_import_or_native_call(self):
        with patch(
            "builtins.__import__", side_effect=AssertionError("unexpected import")
        ):
            self.assertIsNone(expert_load_for_layer(None, 0))


class ExpertLoadRecorderTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available(), "GPU target requires CUDA")
        torch.cuda.init()

    def _recorder(self, experts=4):
        return expert_load_for_layer(
            _make_inputs(experts=experts).expert_token_counts, 0
        )

    def test_accumulate_strides_sentinels_and_local_range(self):
        for dtype in (torch.int32, torch.int64):
            ids = torch.tensor(
                [[0, 0, 1, 0], [1, 0, -1, 0], [2, 0, -2, 0]],
                dtype=dtype,
                device="cuda",
            )[:, ::2]
            recorder = self._recorder(experts=2)
            recorder.record_ids(ids)
            self.assertEqual(recorder.expert_token_counts.cpu().tolist(), [1, 2])
            recorder.record_ids(ids[:1])
            self.assertEqual(recorder.expert_token_counts.cpu().tolist(), [2, 3])
            recorder.record_ids(ids[:0])
            self.assertEqual(recorder.expert_token_counts.cpu().tolist(), [2, 3])
        ids = torch.tensor([-(2**63), 2**63 - 1, 2**32, 0], device="cuda")
        recorder.expert_token_counts.zero_()
        recorder.record_ids(ids)
        self.assertEqual(recorder.expert_token_counts.cpu().tolist(), [1, 0])

    def test_histogram_multiple_partitions_and_expert_tiles(self):
        recorder = self._recorder(experts=5000)
        ids = torch.arange(30017, device="cuda", dtype=torch.int64) % 5000
        recorder.record_ids(ids)
        torch.testing.assert_close(
            recorder.expert_token_counts, torch.bincount(ids, minlength=5000).int()
        )
        recorder.record_ids(ids[::3])
        torch.testing.assert_close(
            recorder.expert_token_counts,
            (
                torch.bincount(ids, minlength=5000)
                + torch.bincount(ids[::3], minlength=5000)
            ).int(),
        )

    def test_graph_replays_accumulate_until_step_reset_and_include_padding(self):
        inputs = _make_inputs()
        recorder = expert_load_for_layer(inputs.expert_token_counts, 0)
        input_buffer = torch.empty((32, 2), device="cuda", dtype=torch.int64)
        graphs = {}
        for bucket in (4, 32):
            input_buffer.fill_(0)
            view = input_buffer[:bucket]
            recorder.record_ids(view)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                expert_load_for_layer(inputs.expert_token_counts, 0).record_ids(view)
            graphs[bucket] = graph
        pointers = (
            recorder.expert_token_counts.data_ptr(),
            input_buffer.data_ptr(),
        )
        for bucket, expert in ((4, 1), (32, 2), (4, 3), (32, 0)):
            input_buffer.fill_(expert)
            recorder.expert_token_counts.zero_()
            graphs[bucket].replay()
            expected = [0] * 4
            expected[expert] = bucket * 2
            self.assertEqual(recorder.expert_token_counts.cpu().tolist(), expected)
            graphs[bucket].replay()
            self.assertEqual(
                recorder.expert_token_counts.cpu().tolist(), [2 * c for c in expected]
            )
        self.assertEqual(
            pointers,
            (
                recorder.expert_token_counts.data_ptr(),
                input_buffer.data_ptr(),
            ),
        )

    def test_chunks_accumulate_until_next_step_reset(self):
        recorder = self._recorder()
        ids = torch.tensor([[0, 1], [1, 2], [3, 3]], device="cuda")
        recorder.record_ids(ids[:2])
        recorder.record_ids(ids[2:])
        self.assertEqual(recorder.expert_token_counts.cpu().tolist(), [1, 2, 1, 2])
        recorder.expert_token_counts.zero_()
        recorder.record_ids(ids[:0])
        recorder.record_ids(ids[:1])
        self.assertEqual(recorder.expert_token_counts.cpu().tolist(), [1, 1, 0, 0])
        recorder.record_ids(ids[2:])
        self.assertEqual(recorder.expert_token_counts.cpu().tolist(), [1, 1, 0, 2])

    def test_problem_size_kernel_records_counts_before_padding(self):
        recorder = self._recorder()
        ids = torch.tensor([[0, 1], [1, -1], [4, 3]], device="cuda")
        offsets = torch.empty(4, dtype=torch.int32, device="cuda")
        problem1 = torch.empty((4, 3), dtype=torch.int32, device="cuda")
        problem2 = torch.empty_like(problem1)

        def record(route_ids, swap1, swap2):
            get_cutlass_moe_mm_without_permute_info(
                route_ids,
                offsets,
                problem1,
                problem2,
                4,
                32,
                64,
                swap1,
                swap2,
                expert_token_counts=recorder.expert_token_counts,
            )

        for swap1 in (False, True):
            for swap2 in (False, True):
                recorder.expert_token_counts.zero_()
                record(ids, swap1, swap2)
                self.assertEqual(recorder.expert_token_counts.tolist(), [1, 2, 0, 1])
                self.assertEqual(problem1[:, int(swap1)].tolist(), [1, 2, 0, 1])
                self.assertEqual(problem2[:, int(swap2)].tolist(), [1, 2, 0, 1])
                self.assertEqual(offsets.tolist(), [0, 1, 3, 3])
        recorder.expert_token_counts.zero_()
        record(ids[:1], False, False)
        record(ids[1:], False, False)
        self.assertEqual(recorder.expert_token_counts.tolist(), [1, 2, 0, 1])
        record(ids[:0], False, False)
        self.assertEqual(recorder.expert_token_counts.tolist(), [1, 2, 0, 1])

    def test_triton_gemm_records_only_local_routes(self):
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.triton_fused_executor import (
            TritonFusedMoeExecutor,
        )
        from rtp_llm.utils.model_weight import W

        for ep_size in (1, 2):
            rank = ep_size - 1
            config = SimpleNamespace(
                ep_size=ep_size, ep_rank=rank, expert_num=4, moe_k=2
            )
            experts = 4 // ep_size
            executor = TritonFusedMoeExecutor(
                config,
                SimpleNamespace(),
                {
                    W.moe_w1: torch.randn(
                        experts, 64, 32, dtype=torch.bfloat16, device="cuda"
                    ),
                    W.moe_w2: torch.randn(
                        experts, 32, 32, dtype=torch.bfloat16, device="cuda"
                    ),
                },
            )
            recorder = self._recorder(experts=experts)
            ids = torch.tensor([[0, 2], [3, 2], [1, 0]], device="cuda")
            payload = ExpertForwardPayload(
                expert_x=torch.randn(3, 32, dtype=torch.bfloat16, device="cuda"),
                expert_topk_ids=ids,
                expert_topk_weights=torch.ones((3, 2), device="cuda"),
            )

            def run(expert_load):
                payload.expert_load = expert_load
                return executor.execute(
                    payload, "SiGLU", None, None, False, None
                ).fused_expert_output

            reference = run(None)
            actual = run(recorder)
            torch.testing.assert_close(actual, reference)
            expected = [2, 1, 2, 1] if ep_size == 1 else [2, 1]
            self.assertEqual(recorder.expert_token_counts.tolist(), expected)
            recorder.expert_token_counts.zero_()
            run(recorder)
            run(recorder)
            self.assertEqual(
                recorder.expert_token_counts.tolist(), [2 * c for c in expected]
            )
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run(recorder)
            ids.fill_(rank * experts)
            for _ in range(2):
                recorder.expert_token_counts.zero_()
                graph.replay()
                self.assertEqual(
                    recorder.expert_token_counts.tolist(), [6] + [0] * (experts - 1)
                )

    def test_grouped_fp4_capture_records_backend_counts(self):
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.grouped_fp4 import (
            GroupedFp4Executor,
        )
        from rtp_llm.models_py.triton_kernels.moe.ep_kernels import (
            recompute_topk_ids_sum_expert_count,
        )

        executor = object.__new__(GroupedFp4Executor)
        nn.Module.__init__(executor)
        executor.cfg = SimpleNamespace(n_routed_experts=4, moe_inter_dim=32)
        # Only GEMM compute is replaced; capture uses the production routing.
        executor._forward_capture_topk = lambda x, weights, indices: x
        ids = torch.tensor([[0, 1], [1, 3]], device="cuda")
        x = torch.ones((2, 32), dtype=torch.bfloat16, device="cuda")
        weights = torch.ones((2, 2), device="cuda")
        recorder = self._recorder()
        _, counts = recompute_topk_ids_sum_expert_count(ids, 0, 4)
        recorder.expert_token_counts.add_(counts)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            executor.forward(x, weights, ids, expert_load=recorder)
        for expert in (2, 0):
            ids.fill_(expert)
            recorder.expert_token_counts.zero_()
            graph.replay()
            expected = [0] * 4
            expected[expert] = 4
            self.assertEqual(recorder.expert_token_counts.tolist(), expected)

    def test_current_stream(self):
        recorder = self._recorder()
        ids = torch.ones(8193, dtype=torch.int32, device="cuda")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            recorder.record_ids(ids)
            ready = torch.cuda.Event()
            ready.record()
        torch.cuda.current_stream().wait_event(ready)
        self.assertEqual(recorder.expert_token_counts.cpu().tolist(), [0, 8193, 0, 0])

    def test_each_input_selects_output_storage_without_module_binding(self):
        model = _SyntheticModel([_make_router(), _make_router()])
        first = _make_inputs(layers=2)
        second = _make_inputs(layers=2)
        ids = torch.tensor([[0, 1], [1, 3]], device="cuda")
        model(ids, first.expert_token_counts)
        self.assertEqual(first.expert_token_counts.cpu().tolist(), [[1, 2, 0, 1]] * 2)
        model(ids[:1], second.expert_token_counts)
        self.assertEqual(second.expert_token_counts.cpu().tolist(), [[1, 1, 0, 0]] * 2)
        self.assertEqual(first.expert_token_counts.cpu().tolist(), [[1, 2, 0, 1]] * 2)
        model(ids, None)
        self.assertEqual(second.expert_token_counts.cpu().tolist(), [[1, 1, 0, 0]] * 2)
        for layer in model.layers:
            for module in (layer, layer.router, layer.fused_experts):
                self.assertFalse(hasattr(module, "bind_expert_load_monitor"))
                self.assertFalse(hasattr(module, "expert_load_recorder"))

    def test_input_views_keep_each_output_storage(self):
        inputs = _make_inputs(layers=2)
        context = expert_load_for_layer(inputs.expert_token_counts, 1)
        ids = torch.tensor([[0, 1]], device="cuda")
        context.record_ids(ids)
        second_inputs = _make_inputs(layers=2)
        second = expert_load_for_layer(second_inputs.expert_token_counts, 1)
        second.record_ids(ids[:, :1])
        self.assertEqual(
            second.expert_token_counts.data_ptr(),
            second_inputs.expert_token_counts[1].data_ptr(),
        )
        self.assertNotEqual(
            context.expert_token_counts.data_ptr(),
            second.expert_token_counts.data_ptr(),
        )
        self.assertEqual(inputs.expert_token_counts[1].tolist(), [1, 1, 0, 0])
        self.assertEqual(second_inputs.expert_token_counts[1].tolist(), [1, 0, 0, 0])
        inputs.expert_token_counts = None
        self.assertIsNone(expert_load_for_layer(inputs.expert_token_counts, 0))
        self.assertIsNone(expert_load_for_layer(None, 0))

    def test_mega_launch_records_backend_destination_counts(self):
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe import (
            MegaMoeExecutor,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_se import (
            MegaMoeSEExecutor,
        )

        for cls in (MegaMoeExecutor, MegaMoeSEExecutor):
            executor = object.__new__(cls)
            nn.Module.__init__(executor)
            executor.config = SimpleNamespace(n_local_experts=2)
            executor.cfg = SimpleNamespace(layer_id=0, swiglu_limit=0)
            executor._mega_l1_w = executor._mega_l1_sf = None
            executor._mega_l2_w = executor._mega_l2_sf = None
            executor._mega_buf = None
            executor._se_l1_w = executor._se_l1_sf = None
            executor._se_l2_w = executor._se_l2_sf = None
            executor._maybe_pre_kernel_barrier = lambda tokens: None
            context = self._recorder(experts=2)
            counts = context.expert_token_counts
            observed_initial_counts = []

            def backend(*args, cumulative_local_expert_recv_stats, **kwargs):
                self.assertEqual(
                    cumulative_local_expert_recv_stats.data_ptr(), counts.data_ptr()
                )
                observed_initial_counts.append(
                    cumulative_local_expert_recv_stats.cpu().tolist()
                )
                cumulative_local_expert_recv_stats.add_(6)

            with patch.dict(
                sys.modules, {"deep_gemm": SimpleNamespace(fp8_fp4_mega_moe=backend)}
            ):
                with patch(cls.__module__ + ".sync_cuda_graph_warmup_ranks"):
                    output = torch.empty((0, 8), device="cuda")
                    executor._launch(output, 0, output.device, context)
                    executor._launch(output, 0, output.device, context)
            self.assertEqual(counts.cpu().tolist(), [12, 12])
            self.assertEqual(observed_initial_counts, [[0, 0], [6, 6]])

    def test_first_record_call_can_be_captured_with_temporary_workspace(self):
        ids = torch.tensor([[0, 1], [1, 3]], dtype=torch.int32, device="cuda")
        counts = torch.zeros((4,), dtype=torch.int32, device="cuda")
        output_pointer = counts.data_ptr()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            record_expert_load(ids, counts)
        for expert in (2, 0, 3):
            ids.fill_(expert)
            counts.zero_()
            graph.replay()
            expected = [0] * 4
            expected[expert] = ids.numel()
            self.assertEqual(counts.tolist(), expected)
            self.assertEqual(counts.data_ptr(), output_pointer)


if __name__ == "__main__":
    unittest.main()
