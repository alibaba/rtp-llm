# type: ignore
"""GPU tests for the production EPLB expert-load recorder.

All tests run against the REAL production implementation:

* ``rtp_llm.models_py.utils.expert_load_monitor.record_expert_load``
  (native CUDA histogram helper; counts=None is the disabled no-op), and
* ``rtp_llm.models_py.utils.expert_load_monitor.bind_expert_load_monitor``
  (contract binder), with the record hook consumed by the REAL
  ``FusedMoeDataRouter.record_expert_load`` / ``PureTpRouterBase.prepare``.

No fake/re-implemented monitor exists in this file.  Synthetic model shells
only provide the structural surface the binder discovers (model.layers ->
child module with ``.fused_moe.router`` / ``.fused_moe.fused_experts``); the
routers are REAL ``PureTpRouterNoQuant`` instances constructed by attribute
injection, following the existing pattern of
``routers/test/deepep_normal_router_hook_test.py``.

Covered:

* direct helper UTs: strided 2-D ids (row- and col-strided), int32 and int64
  ids, invalid sentinels (-1 and >= num_experts), chunked accumulation, an
  int64 lane starting above 2**31 (catches narrowing), counts=None disabled
  no-op, and argument validation errors;
* binding: GenericMoeLayer-style layers WITHOUT layer_idx bind by
  model.layers index, per-layer views are separate, two model instances
  isolate, rebinding the same buffer is allowed (PyWrappedModel double-init),
  rebinding a different buffer raises, unsupported models raise;
* graph semantics: bucket 4 x topk 2 records exactly 8 activations per layer
  for logical batch 1/2/3 (routed padding counts), genuine capture + multiple
  replays accumulate exactly, counter address is fixed across capture,
  warmup/capture pollution is exactly cleared by the lifecycle clear;
* disabled path imports/calls no native recorder and launches neither recorder
  kernel (profiler-checked);
* recording on a non-default stream is correct after a CUDA-event join;
* TP-replicated inputs count once (canonical owner: tp_rank 0 only).

The profiler requires the native histogram kernel on the enabled path and
checks both the histogram and large-expert fallback kernels when disabled.
"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Tuple
from unittest.mock import patch

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
# Loose-script runs (python -I <path>) also need the workspace root so that
# `import rtp_llm...` resolves; under Bazel runfiles it is already present.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from window_ratio_oracle import StepRecorder as OracleRecorder  # noqa: E402

# --------------------------------------------------------------------------
# Production imports -- hard failure, never a substitute.
# --------------------------------------------------------------------------
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_tp_router import (  # noqa: E402
    PureTpRouterNoQuant,
)
from rtp_llm.models_py.utils.expert_load_monitor import (  # noqa: E402
    bind_expert_load_monitor,
    record_expert_load,
)

RECORDER_KERNEL_NAME = "addExpertLoadHistogram"
RECORDER_KERNEL_NAMES = (RECORDER_KERNEL_NAME, "addExpertLoadDirect")

NUM_LAYERS = 2
NUM_EXPERTS = 4
BUCKET = 4
TOPK = 2
HIDDEN = 8

# Balanced routing over a bucket: each expert hit twice per layer per replay.
BALANCED_ROWS = [[0, 1], [0, 2], [1, 3], [2, 3]]


# --------------------------------------------------------------------------
# Synthetic bindable model shells (structure only; routing is real).
# --------------------------------------------------------------------------
def _make_router(tp_size: int, tp_rank: int, ep_size: int = 1, ep_rank: int = 0):
    """REAL PureTpRouterNoQuant with the attributes its methods touch."""
    router = object.__new__(PureTpRouterNoQuant)
    router.config = SimpleNamespace(
        tp_size=tp_size,
        tp_rank=tp_rank,
        expert_num=NUM_EXPERTS,
        physical_expert_num=NUM_EXPERTS,
    )
    router.quant_config = None
    router.tp_size = tp_size
    router.ep_size = ep_size
    router.ep_rank = ep_rank
    router.expert_num = NUM_EXPERTS
    router.expert_num_per_rank = NUM_EXPERTS // ep_size
    router.expert_start_id = ep_rank * router.expert_num_per_rank
    router.do_recompute_topk = False
    # Mirror FusedMoeDataRouter.__init__ (bypassed by object.__new__).
    router._expert_load_counts = None
    return router


class _SyntheticMoeBlock(nn.Module):
    """Inner MoE submodule with NO layer_id: binds via model.layers index."""

    def __init__(self, router) -> None:
        super().__init__()
        self.fused_moe = SimpleNamespace(router=router, fused_experts=SimpleNamespace())


class _SyntheticLayer(nn.Module):
    """GenericMoeLayer-style shell WITHOUT layer_idx."""

    def __init__(self, router) -> None:
        super().__init__()
        self.moe = _SyntheticMoeBlock(router)

    def forward(self, a1, topk_ids, topk_weights):
        return self.moe.fused_moe.router.prepare(a1, None, None, topk_weights, topk_ids)


class _SyntheticModel(nn.Module):
    def __init__(self, routers) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_SyntheticLayer(r) for r in routers])

    def forward(self, per_layer_inputs) -> None:
        for layer, (a1, ids, weights) in zip(self.layers, per_layer_inputs):
            layer(a1, ids, weights)


def _routing_tensors(rows: List[List[int]], hidden: int = HIDDEN):
    ids = torch.tensor(rows, dtype=torch.int64, device="cuda")
    weights = torch.ones_like(ids, dtype=torch.float32)
    a1 = torch.zeros(ids.shape[0], hidden, dtype=torch.bfloat16, device="cuda")
    return a1, ids, weights


def _expected_hist(num_layers: int, rows_per_layer: List[List[List[int]]]):
    """Oracle expectation; invalid sentinel ids excluded, per contract."""
    oracle = OracleRecorder(num_layers, NUM_EXPERTS)
    for layer in range(num_layers):
        for rows in rows_per_layer[layer]:
            valid = [e for row in rows for e in row if 0 <= e < NUM_EXPERTS]
            oracle.record(layer, valid)
    return oracle.snapshot()


def _counts_list(step_counts: torch.Tensor) -> List[List[int]]:
    return step_counts.detach().to("cpu").tolist()


class _GpuTestBase(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.fail("This GPU target requires an available CUDA device")
        torch.cuda.init()

    def _build_bound_model(
        self,
        num_layers: int = NUM_LAYERS,
        tp_size: int = 1,
        tp_rank: int = 0,
    ) -> Tuple[_SyntheticModel, torch.Tensor]:
        step_counts = torch.zeros(
            num_layers, NUM_EXPERTS, dtype=torch.int64, device="cuda"
        )
        routers = [
            _make_router(tp_size=tp_size, tp_rank=tp_rank) for _ in range(num_layers)
        ]
        model = _SyntheticModel(routers).cuda()
        bind_expert_load_monitor(model, step_counts)
        return model, step_counts

    def _capture_graph(self, model, per_layer_inputs):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):  # warmup runs: pollution that must be cleared
                model(per_layer_inputs)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            model(per_layer_inputs)
        return graph


class RecorderDisabledPathTest(unittest.TestCase):
    def test_binding_on_rocm_rejects_before_tensor_validation(self) -> None:
        with patch.object(torch.version, "hip", "test"):
            with self.assertRaisesRegex(NotImplementedError, "NVIDIA CUDA only"):
                bind_expert_load_monitor(None, None)

    def test_disabled_does_not_import_or_call_native_recorder(self) -> None:
        from rtp_llm.ops import compute_ops

        with patch.object(
            compute_ops,
            "record_expert_load",
            side_effect=AssertionError("disabled recorder called native code"),
        ) as native:
            with patch(
                "builtins.__import__",
                side_effect=AssertionError("disabled recorder attempted an import"),
            ):
                # No tensor properties or native argument conversion is allowed.
                record_expert_load(object(), None)
            native.assert_not_called()


class RecordExpertLoadHelperTest(_GpuTestBase):
    """Direct UTs of the production record_expert_load helper + native kernels."""

    def test_counts_none_is_disabled_noop(self) -> None:
        ids = torch.tensor(BALANCED_ROWS, dtype=torch.int64, device="cuda")
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            record_expert_load(ids, None)
            torch.cuda.synchronize()
        names = [e.key for e in prof.key_averages()]
        self.assertFalse(
            any(kernel in n for n in names for kernel in RECORDER_KERNEL_NAMES),
            f"counts=None must not launch recorder kernels: {names}",
        )

    def test_basic_histogram(self) -> None:
        counts = torch.zeros(NUM_EXPERTS, dtype=torch.int64, device="cuda")
        ids = torch.tensor(BALANCED_ROWS, dtype=torch.int64, device="cuda")
        record_expert_load(ids, counts)
        torch.cuda.synchronize()
        self.assertEqual(counts.to("cpu").tolist(), [2, 2, 2, 2])

    def test_strided_row_major_view(self) -> None:
        """2-D view with row_stride != width (slice of a wider matrix)."""
        base = torch.tensor(
            [[0, 1, 99, 2], [0, 2, 99, 1], [1, 3, 99, 0], [2, 3, 99, 3]],
            dtype=torch.int64,
            device="cuda",
        )
        view = base[:, :2]  # shape (4,2), stride (4,1): not contiguous
        self.assertFalse(view.is_contiguous())
        counts = torch.zeros(NUM_EXPERTS, dtype=torch.int64, device="cuda")
        record_expert_load(view, counts)
        torch.cuda.synchronize()
        expected = [0] * NUM_EXPERTS
        for row in view.to("cpu").tolist():
            for e in row:
                expected[e] += 1
        self.assertEqual(counts.to("cpu").tolist(), expected)

    def test_strided_transposed_view(self) -> None:
        """col_stride != 1 via transpose."""
        base = torch.tensor(
            [[0, 0, 1, 2], [1, 2, 3, 3]], dtype=torch.int64, device="cuda"
        )
        view = base.t()  # shape (4,2), stride (1,4)
        self.assertFalse(view.is_contiguous())
        counts = torch.zeros(NUM_EXPERTS, dtype=torch.int64, device="cuda")
        record_expert_load(view, counts)
        torch.cuda.synchronize()
        self.assertEqual(counts.to("cpu").tolist(), [2, 2, 2, 2])

    def test_int32_ids(self) -> None:
        counts = torch.zeros(NUM_EXPERTS, dtype=torch.int64, device="cuda")
        ids = torch.tensor(BALANCED_ROWS, dtype=torch.int32, device="cuda")
        record_expert_load(ids, counts)
        torch.cuda.synchronize()
        self.assertEqual(counts.to("cpu").tolist(), [2, 2, 2, 2])

    def test_1d_ids(self) -> None:
        counts = torch.zeros(NUM_EXPERTS, dtype=torch.int64, device="cuda")
        ids = torch.tensor([0, 0, 3, 3, 3], dtype=torch.int64, device="cuda")
        record_expert_load(ids, counts)
        torch.cuda.synchronize()
        self.assertEqual(counts.to("cpu").tolist(), [2, 0, 0, 3])

    def test_1d_strided_ids(self) -> None:
        for dtype in (torch.int32, torch.int64):
            with self.subTest(dtype=dtype):
                ids = torch.tensor(
                    [0, -1, 3, -1, 3, -1, 1], dtype=dtype, device="cuda"
                )[::2]
                counts = torch.zeros(NUM_EXPERTS, dtype=torch.int64, device="cuda")
                record_expert_load(ids, counts)
                self.assertEqual(counts.cpu().tolist(), [1, 1, 0, 2])

    def test_zero_stride_expanded_ids(self) -> None:
        for shape in ((7,), (5, 2)):
            with self.subTest(shape=shape):
                ids = torch.tensor([3], dtype=torch.int64, device="cuda").expand(shape)
                counts = torch.zeros(NUM_EXPERTS, dtype=torch.int64, device="cuda")
                record_expert_load(ids, counts)
                self.assertEqual(counts.cpu().tolist(), [0, 0, 0, ids.numel()])

    def test_multiblock_non_power_of_two_expert_count(self) -> None:
        experts = 37
        values = [i % 41 - 2 for i in range(5003)]
        expected = [values.count(expert) for expert in range(experts)]
        counts = torch.zeros(experts, dtype=torch.int64, device="cuda")
        record_expert_load(
            torch.tensor(values, dtype=torch.int64, device="cuda"), counts
        )
        self.assertEqual(counts.cpu().tolist(), expected)

    def test_large_expert_count_uses_direct_kernel(self) -> None:
        experts = 4097
        values = [0, 4096, 4096, 17, -1, experts, 2**32]
        ids = torch.tensor(values, dtype=torch.int64, device="cuda")
        counts = torch.zeros(experts, dtype=torch.int64, device="cuda")
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            record_expert_load(ids, counts)
            torch.cuda.synchronize()
        self.assertEqual(
            counts.cpu().tolist(), [values.count(expert) for expert in range(experts)]
        )
        self.assertTrue(
            any("addExpertLoadDirect" in event.key for event in prof.key_averages())
        )

    def test_invalid_sentinels_skipped(self) -> None:
        counts = torch.zeros(NUM_EXPERTS, dtype=torch.int64, device="cuda")
        ids = torch.tensor(
            [[0, 1], [-1, -1], [2, 3], [-1, NUM_EXPERTS + 5]],
            dtype=torch.int64,
            device="cuda",
        )
        record_expert_load(ids, counts)
        torch.cuda.synchronize()
        self.assertEqual(counts.to("cpu").tolist(), [1, 1, 1, 1])
        self.assertEqual(int(counts.sum().item()), 4)

    def test_int64_invalid_ids_do_not_wrap_to_valid_experts(self) -> None:
        counts = torch.zeros(NUM_EXPERTS, dtype=torch.int64, device="cuda")
        ids = torch.tensor(
            [0, 3, 2**32, 2**32 + 3, -(2**32), 2**63 - 1, -(2**63)],
            dtype=torch.int64,
            device="cuda",
        )
        record_expert_load(ids, counts)
        self.assertEqual(counts.cpu().tolist(), [1, 0, 0, 1])

    def test_chunked_accumulation(self) -> None:
        counts = torch.zeros(NUM_EXPERTS, dtype=torch.int64, device="cuda")
        record_expert_load(
            torch.tensor([[0, 1]], dtype=torch.int64, device="cuda"), counts
        )
        record_expert_load(
            torch.tensor([[0, 1], [2, 3]], dtype=torch.int64, device="cuda"),
            counts,
        )
        record_expert_load(torch.tensor([3], dtype=torch.int64, device="cuda"), counts)
        torch.cuda.synchronize()
        self.assertEqual(counts.to("cpu").tolist(), [2, 2, 1, 2])

    def test_int64_lane_above_2p31_not_narrowed(self) -> None:
        """A lane already above 2**31 must accumulate exactly (no int32 cast)."""
        counts = torch.zeros(NUM_EXPERTS, dtype=torch.int64, device="cuda")
        counts[3] = 2**31 + 5
        ids = torch.tensor([3, 3, 0], dtype=torch.int64, device="cuda")
        record_expert_load(ids, counts)
        torch.cuda.synchronize()
        got = counts.to("cpu").tolist()
        self.assertEqual(got[3], 2**31 + 7)
        self.assertEqual(got[0], 1)

    def test_empty_ids_noop(self) -> None:
        counts = torch.zeros(NUM_EXPERTS, dtype=torch.int64, device="cuda")
        ids = torch.empty(0, TOPK, dtype=torch.int64, device="cuda")
        record_expert_load(ids, counts)
        torch.cuda.synchronize()
        self.assertEqual(int(counts.sum().item()), 0)

    def test_validation_errors(self) -> None:
        counts = torch.zeros(NUM_EXPERTS, dtype=torch.int64, device="cuda")
        ids = torch.tensor(BALANCED_ROWS, dtype=torch.int64, device="cuda")
        # counts dtype
        with self.assertRaises(ValueError):
            record_expert_load(
                ids, torch.zeros(NUM_EXPERTS, dtype=torch.int32, device="cuda")
            )
        # counts rank
        with self.assertRaises(ValueError):
            record_expert_load(
                ids, torch.zeros(1, NUM_EXPERTS, dtype=torch.int64, device="cuda")
            )
        # counts contiguity
        with self.assertRaises(ValueError):
            record_expert_load(
                ids,
                torch.zeros(2 * NUM_EXPERTS, dtype=torch.int64, device="cuda")[::2],
            )
        # counts device
        with self.assertRaises(ValueError):
            record_expert_load(ids, torch.zeros(NUM_EXPERTS, dtype=torch.int64))
        # ids dtype
        with self.assertRaises(ValueError):
            record_expert_load(ids.float(), counts)
        # ids rank
        with self.assertRaises(ValueError):
            record_expert_load(ids.unsqueeze(0), counts)
        # empty counts
        with self.assertRaises(ValueError):
            record_expert_load(ids, torch.zeros(0, dtype=torch.int64, device="cuda"))
        # disabled path performs NO validation work and never raises
        record_expert_load(ids.float(), None)


class BindExpertLoadMonitorTest(_GpuTestBase):
    """Binder structure: model.layers indexing, isolation, rebinding."""

    def test_layers_without_layer_idx_bind_by_model_index(self) -> None:
        model, step_counts = self._build_bound_model(num_layers=2)
        # Distinct per-layer routing proves the views are separate and the
        # layer index comes from model.layers order.
        rows_l0 = [[0, 1], [0, 1], [0, 1], [0, 1]]
        rows_l1 = [[2, 3], [2, 3], [2, 3], [2, 3]]
        inputs = [_routing_tensors(rows_l0), _routing_tensors(rows_l1)]
        model(inputs)
        torch.cuda.synchronize()
        self.assertEqual(_counts_list(step_counts)[0], [4, 4, 0, 0])
        self.assertEqual(_counts_list(step_counts)[1], [0, 0, 4, 4])

    def test_bind_sets_fused_experts_counts(self) -> None:
        model, step_counts = self._build_bound_model(num_layers=1)
        router = model.layers[0].moe.fused_moe.router
        fused_experts = model.layers[0].moe.fused_moe.fused_experts
        self.assertEqual(
            router._expert_load_counts.data_ptr(), step_counts[0].data_ptr()
        )
        self.assertIs(fused_experts._expert_load_counts, router._expert_load_counts)

    def test_two_instances_isolated(self) -> None:
        model_a, counts_a = self._build_bound_model()
        model_b, counts_b = self._build_bound_model()
        rows = [list(r) for r in BALANCED_ROWS]
        model_a([_routing_tensors(rows)] * NUM_LAYERS)
        torch.cuda.synchronize()
        self.assertGreater(int(counts_a.sum().item()), 0)
        self.assertEqual(
            int(counts_b.sum().item()),
            0,
            "instance B must not observe instance A traffic",
        )

    def test_rebind_same_buffer_allowed_different_rejected(self) -> None:
        """PyWrappedModel initializes py_model twice when CUDA graph is on:
        binding must survive a repeated bind of the same buffer and reject a
        different one."""
        model, step_counts = self._build_bound_model()
        bind_expert_load_monitor(model, step_counts)  # same buffer: OK
        other = torch.zeros_like(step_counts)
        with self.assertRaises(RuntimeError):
            bind_expert_load_monitor(model, other)

    def test_unsupported_model_raises(self) -> None:
        step_counts = torch.zeros(1, NUM_EXPERTS, dtype=torch.int64, device="cuda")
        with self.assertRaises(NotImplementedError):
            bind_expert_load_monitor(nn.Linear(4, 4).cuda(), step_counts)

    def test_redundant_expert_router_rejected(self) -> None:
        """physical_expert_num != expert_num must be explicit, not silent."""
        router = _make_router(tp_size=1, tp_rank=0)
        router.config.physical_expert_num = NUM_EXPERTS + 1
        model = _SyntheticModel([router]).cuda()
        step_counts = torch.zeros(1, NUM_EXPERTS + 1, dtype=torch.int64, device="cuda")
        with self.assertRaises(NotImplementedError):
            bind_expert_load_monitor(model, step_counts)

    def test_step_counts_validation(self) -> None:
        model = _SyntheticModel([_make_router(1, 0)]).cuda()
        with self.assertRaises(ValueError):
            bind_expert_load_monitor(
                model, torch.zeros(NUM_EXPERTS, dtype=torch.int64, device="cuda")
            )
        with self.assertRaises(ValueError):
            bind_expert_load_monitor(
                model, torch.zeros(1, NUM_EXPERTS, dtype=torch.int32, device="cuda")
            )
        with self.assertRaises(ValueError):
            bind_expert_load_monitor(
                model, torch.zeros(1, NUM_EXPERTS, dtype=torch.int64)
            )
        with self.assertRaises(ValueError):
            bind_expert_load_monitor(
                model, torch.zeros(0, NUM_EXPERTS, dtype=torch.int64, device="cuda")
            )


class RecorderGraphSemanticsTest(_GpuTestBase):
    """Graph capture/replay, padding, streams, TP owner, disabled path."""

    def test_bucket_padding_counts_all_logical_batch_sizes(self) -> None:
        """Bucket 4 x topk 2 -> 8 activations for logical batch 1/2/3."""
        model, step_counts = self._build_bound_model()
        for logical_batch in (1, 2, 3):
            step_counts.zero_()
            # Rows past the logical batch are padding; in a real replay they
            # still hold routed ids, and every routed row must count.
            rows = [list(BALANCED_ROWS[i]) for i in range(BUCKET)]
            model([_routing_tensors(rows)] * NUM_LAYERS)
            torch.cuda.synchronize()
            expected = _expected_hist(NUM_LAYERS, [[rows]] * NUM_LAYERS)
            self.assertEqual(
                _counts_list(step_counts),
                expected,
                f"logical_batch={logical_batch}: all bucket rows must count",
            )
            self.assertEqual(int(step_counts.sum().item()), NUM_LAYERS * BUCKET * TOPK)

    def test_graph_capture_replay_multiple_times(self) -> None:
        model, step_counts = self._build_bound_model()
        inputs = [_routing_tensors([list(r) for r in BALANCED_ROWS])] * NUM_LAYERS
        ptr_before = step_counts.data_ptr()
        graph = self._capture_graph(model, inputs)
        self.assertEqual(
            step_counts.data_ptr(),
            ptr_before,
            "step_counts address must stay fixed across capture",
        )
        # Lifecycle clear (C++-owned in production) removes warmup/capture
        # pollution; afterwards every replay must accumulate exactly.
        step_counts.zero_()
        for replay in range(1, 4):
            graph.replay()
            torch.cuda.synchronize()
            expected = _expected_hist(NUM_LAYERS, [[BALANCED_ROWS]] * NUM_LAYERS)
            for row in expected:
                for e in range(NUM_EXPERTS):
                    row[e] *= replay
            self.assertEqual(
                _counts_list(step_counts), expected, f"after {replay} replays"
            )

    def test_capture_warmup_pollution_cleared_before_production(self) -> None:
        model, step_counts = self._build_bound_model()
        inputs = [_routing_tensors([list(r) for r in BALANCED_ROWS])] * NUM_LAYERS
        graph = self._capture_graph(model, inputs)
        torch.cuda.synchronize()
        polluted = int(step_counts.sum().item())
        self.assertGreater(
            polluted,
            0,
            "warmup/capture must record; otherwise the recorder is not "
            "active inside graphs and later checks are void",
        )
        step_counts.zero_()
        self.assertEqual(int(step_counts.sum().item()), 0)
        graph.replay()
        torch.cuda.synchronize()
        self.assertEqual(int(step_counts.sum().item()), NUM_LAYERS * BUCKET * TOPK)

    def test_eager_and_graph_counters_match(self) -> None:
        model, step_counts = self._build_bound_model()
        rows = [list(r) for r in BALANCED_ROWS]
        model([_routing_tensors(rows)] * NUM_LAYERS)
        torch.cuda.synchronize()
        eager = _counts_list(step_counts)

        model2, step_counts2 = self._build_bound_model()
        inputs = [_routing_tensors(rows)] * NUM_LAYERS
        graph = self._capture_graph(model2, inputs)
        step_counts2.zero_()
        graph.replay()
        torch.cuda.synchronize()
        self.assertEqual(eager, _counts_list(step_counts2))

    def test_chunk_accumulation_within_step(self) -> None:
        model, step_counts = self._build_bound_model()
        rows_a = [[0, 1], [0, 1], [0, 1], [0, 1]]
        rows_b = [[2, 3], [2, 3], [2, 3], [2, 3]]
        model([_routing_tensors(rows_a)] * NUM_LAYERS)
        model([_routing_tensors(rows_b)] * NUM_LAYERS)  # same step, 2nd chunk
        torch.cuda.synchronize()
        expected = _expected_hist(NUM_LAYERS, [[rows_a, rows_b]] * NUM_LAYERS)
        self.assertEqual(_counts_list(step_counts), expected)

    def test_disabled_router_launches_no_histogram_kernel(self) -> None:
        """Unbound router: _expert_load_counts stays None (base init) and the
        prepare path must not launch the histogram kernel."""
        routers = [_make_router(1, 0) for _ in range(NUM_LAYERS)]
        model = _SyntheticModel(routers).cuda()  # never bound
        inputs = [_routing_tensors([list(r) for r in BALANCED_ROWS])] * NUM_LAYERS
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            model(inputs)
            torch.cuda.synchronize()
        names = [e.key for e in prof.key_averages()]
        self.assertFalse(
            any(kernel in n for n in names for kernel in RECORDER_KERNEL_NAMES),
            f"disabled path must not launch recorder kernels: {names}",
        )

    def test_enabled_router_launches_histogram_kernel(self) -> None:
        """Positive control for the profiler assertion above."""
        model, step_counts = self._build_bound_model()
        inputs = [_routing_tensors([list(r) for r in BALANCED_ROWS])] * NUM_LAYERS
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            model(inputs)
            torch.cuda.synchronize()
        names = [e.key for e in prof.key_averages()]
        self.assertTrue(
            any(RECORDER_KERNEL_NAME in n for n in names),
            f"enabled path must launch {RECORDER_KERNEL_NAME}: {names}",
        )

    def test_side_stream_recording_joined_by_event(self) -> None:
        """Counts produced on a non-default stream are correct after the
        consumer stream joins via a CUDA event (the production pattern)."""
        model, step_counts = self._build_bound_model()
        inputs = [_routing_tensors([list(r) for r in BALANCED_ROWS])] * NUM_LAYERS
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            model(inputs)
            done = torch.cuda.Event()
            done.record(side)
        torch.cuda.current_stream().wait_event(done)
        observed = torch.empty_like(step_counts)
        observed.copy_(step_counts)  # default stream, after the join
        torch.cuda.synchronize()
        expected = _expected_hist(NUM_LAYERS, [[BALANCED_ROWS]] * NUM_LAYERS)
        self.assertEqual(observed.to("cpu").tolist(), expected)

    def test_tp_replicated_inputs_counted_once(self) -> None:
        """tp_rank 0 is the canonical owner; tp_rank 1 binds counts=None."""
        model_r0, counts_r0 = self._build_bound_model(tp_size=2, tp_rank=0)
        model_r1, counts_r1 = self._build_bound_model(tp_size=2, tp_rank=1)
        router_r1 = model_r1.layers[0].moe.fused_moe.router
        self.assertIsNone(
            router_r1._expert_load_counts,
            "non-owner TP rank must bind counts=None",
        )
        rows = [list(r) for r in BALANCED_ROWS]
        model_r0([_routing_tensors(rows)] * NUM_LAYERS)
        model_r1([_routing_tensors(rows)] * NUM_LAYERS)
        torch.cuda.synchronize()
        expected = _expected_hist(NUM_LAYERS, [[rows]] * NUM_LAYERS)
        combined = [
            [
                counts_r0[l][e].item() + counts_r1[l][e].item()
                for e in range(NUM_EXPERTS)
            ]
            for l in range(NUM_LAYERS)
        ]
        self.assertEqual(
            combined,
            expected,
            "TP-replicated route inputs must be counted exactly once",
        )

    def test_fp8_fp4_ep1_uses_model_tp_owner_and_preserves_cp_shards(self) -> None:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.fp8_fp4_router import (
            Fp8Fp4Router,
        )

        # The FP8/FP4 executor config always exposes tp_rank=0, including on
        # non-owner attention TP ranks. CP instead exposes effective TP rank 0
        # on each source and must keep both independent token shards.
        rows = [list(r) for r in BALANCED_ROWS]
        for model_ranks, expected_copies in (((0, 1), 1), ((0, 0), 2)):
            with self.subTest(model_ranks=model_ranks):
                combined = torch.zeros(1, NUM_EXPERTS, dtype=torch.int64, device="cuda")
                for model_rank in model_ranks:
                    router = object.__new__(Fp8Fp4Router)
                    router.config = SimpleNamespace(
                        expert_num=NUM_EXPERTS,
                        physical_expert_num=NUM_EXPERTS,
                        tp_size=1,
                        tp_rank=0,
                        ep_size=1,
                    )
                    model = _SyntheticModel([router])
                    model._v4_args = SimpleNamespace(tp_rank=model_rank)
                    counts = torch.zeros_like(combined)
                    bind_expert_load_monitor(model, counts)
                    model([_routing_tensors(rows)])
                    combined.add_(counts)
                self.assertEqual(
                    combined.cpu().tolist(),
                    [[2 * expected_copies] * NUM_EXPERTS],
                )

    def test_invalid_sentinel_ids_not_counted(self) -> None:
        model, step_counts = self._build_bound_model()
        rows = [[0, 1], [-1, -1], [2, 3], [-1, -1]]
        model([_routing_tensors(rows)] * NUM_LAYERS)
        torch.cuda.synchronize()
        expected = _expected_hist(NUM_LAYERS, [[rows]] * NUM_LAYERS)
        self.assertEqual(_counts_list(step_counts), expected)
        self.assertEqual(int(step_counts.sum().item()), NUM_LAYERS * 2 * TOPK)


if __name__ == "__main__":
    unittest.main()
