# type: ignore
"""GPU tests for expert-load recording on the REAL DeepEpLowLatencyRouter.

These tests close the wiring gap that helper-level UTs cannot prove: that
``DeepEpLowLatencyRouter.prepare`` records expert load AFTER the TP slice, so
each EP source contributes exactly the rows it actually dispatches.

The full prepare chain is the real production code, unmodified and unmocked:

    prepare -> _prepare_pre_tp_slice -> FusedMoeDataRouter.record_expert_load
            -> rtp_llm.models_py.utils.expert_load_monitor.record_expert_load
            -> native addExpertLoadHistogram kernel -> _normal_prepare

Routers are REAL ``DeepEpLowLatencyRouter`` instances built with
``object.__new__`` + attribute injection (the established pattern of
expert_load_recorder_test.py) so no real DeepEP network buffer is created.
The ONLY stub is the external network primitive ``buffer.low_latency_dispatch``;
it records the exact ``topk_idx`` it receives and returns structurally legal
outputs.  These tests say nothing about real multi-node DeepEP communication.

Counts are real CUDA int64 buffers, bound either through the PUBLIC contract
binder ``bind_expert_load_monitor`` (synthetic model shell only provides the
model.layers -> fused_moe.router discovery surface) or by setting the real
``router._expert_load_counts`` field that the binder itself writes.

Expectations come from the independent pure-Python oracle
``window_ratio_oracle.StepRecorder`` plus handwritten slice bounds.

Multiple "ranks" are simulated sequentially on ONE GPU; each rank owns its own
router/buffer/counts.  This is a TP-slice + histogram UT, not a distributed
test.

Requires a CUDA device: the dedicated Bazel target (planned name
``deepep_router_monitor_test``) MUST FAIL when no GPU is present -- a green
run made of skips is not acceptable.
"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

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
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepep_low_latency_router import (  # noqa: E402
    DeepEpLowLatencyRouter,
)
from rtp_llm.models_py.utils.expert_load_monitor import (  # noqa: E402
    bind_expert_load_monitor,
)

RECORDER_KERNEL_NAME = "addExpertLoadHistogram"
RECORDER_KERNEL_NAMES = (RECORDER_KERNEL_NAME, "addExpertLoadDirect")

NUM_EXPERTS = 4
TOPK = 2
# 2048 is in the router's SUPPORTED_HIDDEN_SIZES; the production shape guard
# is exercised, not relaxed (no hidden-size-8 workaround).
HIDDEN = 2048


# --------------------------------------------------------------------------
# Stub for the EXTERNAL DeepEP network buffer only.
# --------------------------------------------------------------------------
class _StubLowLatencyBuffer:
    """Records low_latency_dispatch calls; returns structurally legal outputs."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def low_latency_dispatch(self, **kwargs):
        self.calls.append(
            {k: (v.clone() if torch.is_tensor(v) else v) for k, v in kwargs.items()}
        )
        x = kwargs["x"]
        expert_x = torch.zeros(x.size(0), x.size(1), dtype=x.dtype, device=x.device)
        expert_num_tokens = torch.zeros(NUM_EXPERTS, dtype=torch.int32, device=x.device)
        handle = ("stub_handle", x.size(0))
        return expert_x, expert_num_tokens, handle, None, None


def _make_router(
    tp_size: int,
    tp_rank: int,
    buffer: _StubLowLatencyBuffer,
    counts: Optional[torch.Tensor] = None,
    ep_size: int = 2,
) -> DeepEpLowLatencyRouter:
    """REAL router with the attributes its prepare chain touches.

    ``DeepEpLowLatencyRouter.__init__`` would create the real DeepEP network
    buffer via a singleton; we bypass it and inject exactly the members the
    production prepare path reads.  ``counts`` takes the real field the public
    binder writes (None = monitoring disabled).
    """
    router = object.__new__(DeepEpLowLatencyRouter)
    router.config = SimpleNamespace(
        tp_size=tp_size,
        tp_rank=tp_rank,
        ep_size=ep_size,
        expert_num=NUM_EXPERTS,
        physical_expert_num=NUM_EXPERTS,
    )
    router.quant_config = SimpleNamespace(
        is_quantized=False,
        is_block_quantized=False,
        is_per_group_fp4=False,
    )
    # Mirror FusedMoeDataRouter.__init__ (bypassed by object.__new__).
    router.tp_size = tp_size
    router._num_experts = NUM_EXPERTS
    router._num_topk = TOPK
    router._num_max_dispatch_tokens_per_rank = 128
    router._use_fp8_dispatch = False
    router._async_finish = False
    router._return_recv_hook = False
    router._handle = None
    router._buffer = buffer
    router._expert_load_counts = counts
    return router


# --------------------------------------------------------------------------
# Synthetic bindable model shells (structure only; routing is real).
# --------------------------------------------------------------------------
class _SyntheticMoeBlock(nn.Module):
    def __init__(self, router) -> None:
        super().__init__()
        self.fused_moe = SimpleNamespace(router=router, fused_experts=SimpleNamespace())


class _SyntheticLayer(nn.Module):
    def __init__(self, router) -> None:
        super().__init__()
        self.moe = _SyntheticMoeBlock(router)


class _SyntheticModel(nn.Module):
    def __init__(self, routers) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_SyntheticLayer(r) for r in routers])


def _oracle_hist(chunks: List[List[List[int]]]) -> List[int]:
    """Independent CPU oracle histogram for one layer over routed id chunks."""
    oracle = OracleRecorder(1, NUM_EXPERTS)
    for rows in chunks:
        oracle.record(0, [e for row in rows for e in row if 0 <= e < NUM_EXPERTS])
    return oracle.snapshot()[0]


class _GpuTestBase(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.fail("This GPU target requires an available CUDA device")
        torch.cuda.init()

    def _bind(self, router) -> torch.Tensor:
        """Bind via the PUBLIC contract binder; return the step_counts row."""
        model = _SyntheticModel([router]).cuda()
        step_counts = torch.zeros(1, NUM_EXPERTS, dtype=torch.int64, device="cuda")
        bind_expert_load_monitor(model, step_counts)
        return step_counts

    def _inputs(self, rows: List[List[int]], dtype=torch.int64):
        ids = torch.tensor(rows, dtype=dtype, device="cuda")
        weights = torch.linspace(
            0.5,
            0.5 + 0.5 * (ids.numel() - 1),
            ids.numel(),
            dtype=torch.float32,
            device="cuda",
        ).reshape(ids.shape)
        a1 = torch.zeros(ids.size(0), HIDDEN, dtype=torch.bfloat16, device="cuda")
        return a1, ids, weights


class DeepEpRouterMonitorTest(_GpuTestBase):
    """TP-slice recording position on the real DeepEpLowLatencyRouter.prepare."""

    def test_tp2_records_exactly_each_ranks_dispatched_slice(self) -> None:
        """TP2, 5 tokens x topk2: rank0 must count rows 0-2, rank1 rows 3-4.

        A hook placed BEFORE the TP slice would count all 5 rows on every rank
        (each rank sums to 10, cross-rank sum 20); the assertions below fail
        for that implementation.
        """
        rows = [[0, 1], [2, 3], [0, 3], [1, 2], [2, 3]]
        a1, ids, weights = self._inputs(rows)
        # Handwritten slice bounds for ceil(5/2)=3 rows per rank.
        expected_slice = {0: rows[0:3], 1: rows[3:5]}

        per_rank_counts: List[List[int]] = []
        for tp_rank in (0, 1):
            buffer = _StubLowLatencyBuffer()
            router = _make_router(tp_size=2, tp_rank=tp_rank, buffer=buffer)
            step_counts = self._bind(router)
            # Binder must treat DeepEP LL ranks as independent owners (unlike
            # pure-TP's canonical tp_rank==0 owner).
            self.assertEqual(
                router._expert_load_counts.data_ptr(), step_counts[0].data_ptr()
            )

            payload = router.prepare(a1, None, None, weights, ids)
            torch.cuda.synchronize()

            # The dispatch primitive received exactly this rank's TP slice.
            self.assertEqual(len(buffer.calls), 1)
            dispatched = buffer.calls[0]["topk_idx"]
            self.assertEqual(dispatched.dtype, torch.int64)
            self.assertEqual(dispatched.to("cpu").tolist(), expected_slice[tp_rank])
            # The recorded histogram equals exactly those dispatched rows.
            self.assertEqual(
                step_counts[0].to("cpu").tolist(),
                _oracle_hist([expected_slice[tp_rank]]),
                f"tp_rank={tp_rank} must count its dispatched slice only",
            )
            # Payload wiring: sliced ids/weights travel with the payload.
            self.assertEqual(
                payload.expert_topk_ids.to("cpu").tolist(), expected_slice[tp_rank]
            )
            self.assertTrue(
                torch.equal(
                    payload.expert_topk_weights,
                    weights[tp_rank * 3 : tp_rank * 3 + len(expected_slice[tp_rank])],
                )
            )
            per_rank_counts.append(step_counts[0].to("cpu").tolist())

        # Cross-source SUM equals the full 10-assignment histogram exactly once.
        combined = [a + b for a, b in zip(per_rank_counts[0], per_rank_counts[1])]
        self.assertEqual(combined, _oracle_hist([rows]))
        self.assertEqual(sum(combined), 10)

    def test_tp4_single_token_empty_slices_record_nothing(self) -> None:
        """TP4 with 1 token: ranks 1-3 get an empty slice and must record 0.

        No out-of-bounds, no fabricated activations; the stub still returns a
        legal (empty) payload so the real prepare chain completes.
        """
        rows = [[2, 3]]
        a1, ids, weights = self._inputs(rows)

        combined = [0] * NUM_EXPERTS
        for tp_rank in range(4):
            buffer = _StubLowLatencyBuffer()
            router = _make_router(tp_size=4, tp_rank=tp_rank, buffer=buffer)
            step_counts = self._bind(router)
            payload = router.prepare(a1, None, None, weights, ids)
            torch.cuda.synchronize()

            expected_rows = rows if tp_rank == 0 else []
            self.assertEqual(
                buffer.calls[0]["topk_idx"].to("cpu").tolist(), expected_rows
            )
            counts = step_counts[0].to("cpu").tolist()
            if tp_rank == 0:
                self.assertEqual(counts, [0, 0, 1, 1])
            else:
                self.assertEqual(
                    counts,
                    [0, 0, 0, 0],
                    f"tp_rank={tp_rank}: empty TP slice must record nothing",
                )
                self.assertEqual(payload.expert_topk_ids.numel(), 0)
                self.assertEqual(payload.expert_x.size(0), 0)
            combined = [c + n for c, n in zip(combined, counts)]

        self.assertEqual(combined, [0, 0, 1, 1])
        self.assertEqual(sum(combined), 2)

    def test_int32_sentinels_chunks_and_padding_accumulate(self) -> None:
        """int32 ids convert to int64 before dispatch; -1 sentinels are skipped;
        chunks accumulate on a shared counts buffer; routed padding rows count.

        prepare() asserts ``self._handle is None``, so each chunk uses a fresh
        real router (the alternative production path is finalize() clearing the
        handle); all chunk routers share one counts buffer like the persistent
        production counter.
        """
        # 4 physical rows: rows 0-1 "logical", rows 2-3 padding that still hold
        # routed ids and MUST be counted per contract.
        chunk1 = [[0, 1], [-1, -1], [2, 3], [0, 3]]
        chunk2 = [[1, 2], [-1, 3]]
        counts = torch.zeros(NUM_EXPERTS, dtype=torch.int64, device="cuda")

        for chunk, dtype in ((chunk1, torch.int32), (chunk2, torch.int64)):
            buffer = _StubLowLatencyBuffer()
            router = _make_router(tp_size=1, tp_rank=0, buffer=buffer, counts=counts)
            a1, ids, weights = self._inputs(chunk, dtype=dtype)
            payload = router.prepare(a1, None, None, weights, ids)
            # The router converts ids to int64 BEFORE recording and dispatch.
            self.assertEqual(buffer.calls[0]["topk_idx"].dtype, torch.int64)
            self.assertEqual(payload.expert_topk_ids.dtype, torch.int64)
            self.assertIsNotNone(
                router.handle,
                "real prepare sets the handle; a second chunk on this router "
                "would assert, hence the fresh router per chunk",
            )
        torch.cuda.synchronize()

        self.assertEqual(
            counts.to("cpu").tolist(),
            _oracle_hist([chunk1, chunk2]),
            "int32/-1/padding/chunk accumulation mismatch",
        )
        # Handwritten cross-check: chunk1 -> [2,1,1,2], chunk2 -> [0,1,1,1].
        self.assertEqual(counts.to("cpu").tolist(), [2, 2, 2, 3])
        self.assertEqual(int(counts.sum().item()), 9)

    def test_disabled_counts_none_prepare_unchanged(self) -> None:
        """counts=None: prepare output and dispatch args are byte-identical to
        the enabled run, inputs are not mutated, and no histogram kernel runs."""
        rows = [[0, 1], [2, 3], [0, 3], [1, 2], [2, 3]]
        a1, ids, weights = self._inputs(rows)
        ids_before = ids.clone()
        weights_before = weights.clone()

        enabled_buffer = _StubLowLatencyBuffer()
        enabled = _make_router(
            tp_size=2,
            tp_rank=0,
            buffer=enabled_buffer,
            counts=torch.zeros(NUM_EXPERTS, dtype=torch.int64, device="cuda"),
        )
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]
        ) as prof_on:
            payload_on = enabled.prepare(a1, None, None, weights, ids)
            torch.cuda.synchronize()
        names_on = [e.key for e in prof_on.key_averages()]
        self.assertTrue(
            any(RECORDER_KERNEL_NAME in n for n in names_on),
            f"enabled prepare must launch {RECORDER_KERNEL_NAME}: {names_on}",
        )

        disabled_buffer = _StubLowLatencyBuffer()
        disabled = _make_router(
            tp_size=2, tp_rank=0, buffer=disabled_buffer, counts=None
        )
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]
        ) as prof_off:
            payload_off = disabled.prepare(a1, None, None, weights, ids)
            torch.cuda.synchronize()
        names_off = [e.key for e in prof_off.key_averages()]
        self.assertFalse(
            any(kernel in n for n in names_off for kernel in RECORDER_KERNEL_NAMES),
            f"counts=None must not launch recorder kernels: {names_off}",
        )

        # Inputs untouched by the monitoring hook.
        self.assertTrue(torch.equal(ids, ids_before))
        self.assertTrue(torch.equal(weights, weights_before))
        # Dispatch arguments identical between enabled and disabled runs.
        self.assertEqual(len(enabled_buffer.calls), 1)
        self.assertEqual(len(disabled_buffer.calls), 1)
        call_on, call_off = enabled_buffer.calls[0], disabled_buffer.calls[0]
        self.assertEqual(set(call_on.keys()), set(call_off.keys()))
        for key in call_on:
            v_on, v_off = call_on[key], call_off[key]
            if torch.is_tensor(v_on):
                self.assertTrue(torch.equal(v_on, v_off), f"dispatch arg {key}")
            else:
                self.assertEqual(v_on, v_off, f"dispatch arg {key}")
        # Payload parity.
        self.assertTrue(torch.equal(payload_on.expert_x, payload_off.expert_x))
        self.assertTrue(
            torch.equal(payload_on.expert_topk_ids, payload_off.expert_topk_ids)
        )
        self.assertTrue(
            torch.equal(payload_on.expert_topk_weights, payload_off.expert_topk_weights)
        )
        self.assertEqual(
            payload_on.expert_tokens_meta.expected_m,
            payload_off.expert_tokens_meta.expected_m,
        )
        self.assertEqual(
            payload_on.expert_x_origin_dtype, payload_off.expert_x_origin_dtype
        )

    def test_hidden_size_guard_not_relaxed(self) -> None:
        """The production shape guard stays: hidden=8 is rejected, 2048 works."""
        rows = [[0, 1]]
        ids = torch.tensor(rows, dtype=torch.int64, device="cuda")
        weights = torch.ones_like(ids, dtype=torch.float32)
        bad_a1 = torch.zeros(1, 8, dtype=torch.bfloat16, device="cuda")
        router = _make_router(tp_size=1, tp_rank=0, buffer=_StubLowLatencyBuffer())
        with self.assertRaises(AssertionError):
            router.prepare(bad_a1, None, None, weights, ids)


if __name__ == "__main__":
    unittest.main()
