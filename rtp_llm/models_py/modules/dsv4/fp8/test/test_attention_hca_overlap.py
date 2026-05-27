"""UT: AttentionFP8 HCA CP-overlap orchestrator + dispatch gate.

Locks the externally observable behaviour of the Phase-Z overlap path
without spinning up a real DSV4 layer (no DeepGEMM / no real KV pool):

  * ``_should_overlap_cp_for_prefill`` gate matrix
      - env off                → False
      - env on, cp inactive    → False
      - env on, ratio==0       → False
      - env on, cp on, ratio>0 → True (CUDA-graph capture is best-effort
        and not exercised here — the capture branch is a CUDA-runtime
        guard, not a pure-Python conditional)

  * ``_forward_prefill_hca_overlapped`` orchestration order
      1. compressor.start_prefill (with the per-instance cp_gather_stream)
      2. _prefill_write_swa_fp8_paged (default-stream work — overlap zone)
      3. compressor.finish_prefill on the returned pending
      4. _forward_prefill_compressed(_skip_compressor_write=True) for the
         workspace path (no second synchronous compressor call)

  * ``_forward_prefill_compressed(_skip_compressor_write=True)`` does NOT
    invoke ``self.compressor`` — the orchestrator already drained it.

  * Per-instance compressor cp_gather_stream is cached + reused (FIFO ordering
    contract for layers that issue >1 compressor NCCL collective inside one
    step, e.g. CSA nested-indexer + main compressor). SWA kv_full uses a
    separate cached stream so it does not sit in front of compressor gathers.
"""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from rtp_llm.models_py.modules.dsv4.fp8.attention import (
    AttentionFP8,
    PrefillMeta,
    PrefillQKV,
    _prefill_cp_overlap_enabled,
)


def _make_attention_stub(
    *, compress_ratio: int, has_compressor: bool = True
) -> AttentionFP8:
    """Construct an AttentionFP8 via __new__ (bypassing __init__'s heavy
    weight loading) and set just the attributes the orchestrator + gate
    touch."""
    layer = AttentionFP8.__new__(AttentionFP8)
    # torch.nn.Module init is required so attribute assignment of
    # MagicMocks etc. doesn't get rejected by __setattr__.
    torch.nn.Module.__init__(layer)
    layer.compress_ratio = compress_ratio
    layer.indexer = None
    if has_compressor:
        layer.compressor = SimpleNamespace(
            start_prefill=MagicMock(name="compressor.start_prefill"),
            wait_prefill_gather=MagicMock(name="compressor.wait_prefill_gather"),
            finish_prefill=MagicMock(name="compressor.finish_prefill"),
            __call__=MagicMock(name="compressor.__call__"),
        )
        # SimpleNamespace doesn't make __call__ work via instance dispatch;
        # wrap the namespace in a callable shim that delegates to its mock.
        original_call = layer.compressor.__call__
        compressor_callable = MagicMock(
            name="compressor",
            side_effect=lambda *a, **kw: original_call(*a, **kw),
        )
        compressor_callable.start_prefill = layer.compressor.start_prefill
        compressor_callable.wait_prefill_gather = layer.compressor.wait_prefill_gather
        compressor_callable.finish_prefill = layer.compressor.finish_prefill
        layer.compressor = compressor_callable
    else:
        layer.compressor = None
    # Stubs for the methods the orchestrator routes through. Replaced
    # per-test with MagicMocks to observe call order.
    layer._prefill_write_swa_fp8_paged = MagicMock(name="_prefill_write_swa_fp8_paged")
    layer._forward_prefill_compressed = MagicMock(
        name="_forward_prefill_compressed",
        return_value=torch.zeros(2, 8, dtype=torch.bfloat16),
    )
    return layer


def _make_common(
    *, cp_on: bool, cp_size: int = 2, device: torch.device = torch.device("cpu")
) -> PrefillMeta:
    """Minimal PrefillMeta with just the fields the orchestrator + gate
    inspect (cp_on, cp_ctx, sp_int, hca_meta)."""
    tensor_device = (
        device
        if device.type != "cuda" or torch.cuda.is_available()
        else torch.device("cpu")
    )
    if cp_on:
        cp_ctx = SimpleNamespace(cp_size=cp_size, seq_len_full=4)
    else:
        cp_ctx = None
    hca_meta = SimpleNamespace(
        compressor_meta=SimpleNamespace(name="hoisted_hca_compressor_meta"),
        workspace_meta=SimpleNamespace(name="hca_workspace_meta"),
    )
    return PrefillMeta(
        seqlen=2,
        seqlen_full=4 if cp_on else 2,
        rd=0,
        device=device,
        cp_ctx=cp_ctx,
        cp_on=cp_on,
        freqs_cis=torch.zeros(1, dtype=torch.float32, device=tensor_device),
        topk_idxs=torch.zeros(1, dtype=torch.int32, device=tensor_device),
        sp_int=0,
        any_cont=False,
        row_seqlens_full=torch.tensor([2], dtype=torch.long, device=tensor_device),
        hca_meta=hca_meta,
    )


def _make_qkv(device: torch.device) -> PrefillQKV:
    return PrefillQKV(
        qr=torch.zeros(2, 4, dtype=torch.bfloat16, device=device),
        q=torch.zeros(2, 4, dtype=torch.bfloat16, device=device),
        kv_full=torch.zeros(2, 4, dtype=torch.bfloat16, device=device),
    )


class HCAOverlapGateTest(unittest.TestCase):
    """Pure-Python gate matrix — no CUDA required."""

    def setUp(self) -> None:
        self.device = torch.device("cpu")

    def _with_env(self, value: str):
        return patch.dict(os.environ, {"DSV4_PREFILL_CP_OVERLAP": value})

    def test_env_off_is_false(self) -> None:
        layer = _make_attention_stub(compress_ratio=128)
        common = _make_common(cp_on=True, device=self.device)
        with self._with_env("0"):
            self.assertFalse(_prefill_cp_overlap_enabled())
            self.assertFalse(layer._should_overlap_cp_for_prefill(common))

    def test_env_on_cp_off_is_false(self) -> None:
        layer = _make_attention_stub(compress_ratio=128)
        common = _make_common(cp_on=False, device=self.device)
        with self._with_env("1"):
            self.assertFalse(layer._should_overlap_cp_for_prefill(common))

    def test_env_on_cp_size_1_is_false(self) -> None:
        layer = _make_attention_stub(compress_ratio=128)
        common = _make_common(cp_on=True, cp_size=1, device=self.device)
        with self._with_env("1"):
            self.assertFalse(layer._should_overlap_cp_for_prefill(common))

    def test_env_on_swa_only_ratio_zero_is_false(self) -> None:
        layer = _make_attention_stub(compress_ratio=0, has_compressor=False)
        common = _make_common(cp_on=True, device=self.device)
        with self._with_env("1"):
            self.assertFalse(layer._should_overlap_cp_for_prefill(common))

    def test_env_on_cp_active_cpu_device_is_false(self) -> None:
        layer = _make_attention_stub(compress_ratio=128)
        common = _make_common(cp_on=True, device=self.device)
        with self._with_env("1"):
            self.assertFalse(layer._should_overlap_cp_for_prefill(common))

    def test_env_on_cp_active_cuda_hca_is_true(self) -> None:
        layer = _make_attention_stub(compress_ratio=128)
        common = _make_common(cp_on=True, device=torch.device("cuda"))
        with self._with_env("1"):
            self.assertTrue(layer._should_overlap_cp_for_prefill(common))

    def test_env_on_cp_active_cuda_unsupported_ratio_is_false(self) -> None:
        layer = _make_attention_stub(compress_ratio=128)
        common = _make_common(cp_on=True, device=torch.device("cuda"))
        layer.compress_ratio = 16
        with self._with_env("1"):
            self.assertFalse(layer._should_overlap_cp_for_prefill(common))

    def test_env_on_cp_active_cuda_csa_is_true(self) -> None:
        layer = _make_attention_stub(compress_ratio=4)
        common = _make_common(cp_on=True, device=torch.device("cuda"))
        with self._with_env("1"):
            self.assertTrue(layer._should_overlap_cp_for_prefill(common))

    def test_swa_kv_overlap_is_gated_by_master_flag(self) -> None:
        layer = _make_attention_stub(compress_ratio=0, has_compressor=False)
        common = _make_common(cp_on=True, device=torch.device("cuda"))
        with self._with_env("0"):
            self.assertFalse(layer._should_overlap_swa_kv_gather_for_prefill(common))
        with self._with_env("1"):
            self.assertTrue(layer._should_overlap_swa_kv_gather_for_prefill(common))


@unittest.skipUnless(torch.cuda.is_available(), "stream allocation requires CUDA")
class HCAOverlapStreamCacheTest(unittest.TestCase):
    """Per-instance stream reuse and SWA/compressor stream separation."""

    def test_returns_same_stream_on_repeated_calls(self) -> None:
        layer = _make_attention_stub(compress_ratio=128)
        device = torch.device("cuda")
        s1 = layer._get_cp_gather_stream(device)
        s2 = layer._get_cp_gather_stream(device)
        self.assertIs(s1, s2)
        self.assertIsInstance(s1, torch.cuda.Stream)

    def test_swa_stream_is_cached_but_separate_from_compressor_stream(self) -> None:
        layer = _make_attention_stub(compress_ratio=128)
        device = torch.device("cuda")
        compressor_stream = layer._get_cp_gather_stream(device)
        swa_stream_1 = layer._get_swa_cp_gather_stream(device)
        swa_stream_2 = layer._get_swa_cp_gather_stream(device)

        self.assertIs(swa_stream_1, swa_stream_2)
        self.assertIsInstance(swa_stream_1, torch.cuda.Stream)
        self.assertIsNot(swa_stream_1, compressor_stream)


class HCAOverlapOrchestratorTest(unittest.TestCase):
    """Order-of-operations contract for the HCA orchestrator."""

    def setUp(self) -> None:
        self.device = torch.device("cpu")

    def test_orchestrator_runs_start_swa_finish_workspace_in_order(self) -> None:
        layer = _make_attention_stub(compress_ratio=128)
        # Stub the cp_gather_stream helper so we don't need CUDA.
        stream_sentinel = object()
        layer._get_cp_gather_stream = MagicMock(  # type: ignore[assignment]
            return_value=stream_sentinel
        )
        pending_sentinel = object()
        layer.compressor.start_prefill.return_value = pending_sentinel

        # Order-tracking: every observable call records into one list so
        # the test can assert the exact sequence in one place.
        seq: list = []
        layer.compressor.start_prefill.side_effect = (
            lambda *a, **kw: seq.append(("start", a, kw)) or pending_sentinel
        )
        layer._prefill_write_swa_fp8_paged.side_effect = lambda *a, **kw: seq.append(
            ("swa_write", a, kw)
        )
        layer.compressor.finish_prefill.side_effect = lambda p: seq.append(
            ("finish", p)
        )
        layer._forward_prefill_compressed.side_effect = lambda *a, **kw: seq.append(
            ("compressed", a, kw)
        ) or torch.zeros(2, 8, dtype=torch.bfloat16)

        common = _make_common(cp_on=True, device=self.device)
        qkv = _make_qkv(self.device)
        x = torch.zeros(2, 4, dtype=torch.bfloat16, device=self.device)

        out = layer._forward_prefill_hca_overlapped(x, qkv, common)

        # Sequence stamps.
        self.assertEqual(
            [step[0] for step in seq],
            ["start", "swa_write", "finish", "compressed"],
        )

        # start_prefill kwargs: meta = hca_meta.compressor_meta; stream
        # is the cached cp_gather_stream; positional args = (x, sp_int).
        start_args, start_kwargs = seq[0][1], seq[0][2]
        self.assertIs(start_args[0], x)
        self.assertEqual(start_args[1], common.sp_int)
        self.assertIs(start_kwargs["meta"], common.hca_meta.compressor_meta)
        self.assertIs(start_kwargs["cp_gather_stream"], stream_sentinel)

        # finish_prefill receives the pending start_prefill returned.
        self.assertIs(seq[2][1], pending_sentinel)

        # Final call hits _forward_prefill_compressed with
        # _skip_compressor_write=True so the orchestrator's finish is the
        # sole compressor write.
        comp_args, comp_kwargs = seq[3][1], seq[3][2]
        self.assertIs(comp_args[0], x)
        self.assertIs(comp_args[1], qkv)
        self.assertIs(comp_args[2], common)
        self.assertIsNone(comp_kwargs["cmp_topk_runtime"])
        self.assertIs(comp_kwargs["compressor_meta"], common.hca_meta.compressor_meta)
        self.assertIs(comp_kwargs["workspace_meta"], common.hca_meta.workspace_meta)
        self.assertTrue(comp_kwargs["_skip_compressor_write"])

        # Output threaded through.
        self.assertEqual(tuple(out.shape), (2, 8))

    def test_orchestrator_drains_pending_gather_on_exception(self) -> None:
        layer = _make_attention_stub(compress_ratio=128)
        stream_sentinel = object()
        layer._get_cp_gather_stream = MagicMock(  # type: ignore[assignment]
            return_value=stream_sentinel
        )
        pending_sentinel = object()
        layer.compressor.start_prefill.return_value = pending_sentinel
        layer._prefill_write_swa_fp8_paged.side_effect = RuntimeError("boom")

        common = _make_common(cp_on=True, device=self.device)
        qkv = _make_qkv(self.device)
        x = torch.zeros(2, 4, dtype=torch.bfloat16, device=self.device)

        with self.assertRaisesRegex(RuntimeError, "boom"):
            layer._forward_prefill_hca_overlapped(x, qkv, common)

        layer.compressor.wait_prefill_gather.assert_called_once_with(pending_sentinel)
        layer.compressor.finish_prefill.assert_not_called()
        layer._forward_prefill_compressed.assert_not_called()


class CompressedSkipCompressorWriteTest(unittest.TestCase):
    """`_forward_prefill_compressed(_skip_compressor_write=True)` must
    NOT call ``self.compressor`` (the orchestrator already drained it)."""

    def setUp(self) -> None:
        self.device = torch.device("cpu")

    def test_skip_flag_suppresses_compressor_call(self) -> None:
        layer = _make_attention_stub(compress_ratio=128)
        # Stub the warmup-fallback attn so we don't need flash_mla.
        layer._attn_fp8_swa_via_kv_full = MagicMock(  # type: ignore[assignment]
            return_value=torch.zeros(2, 8, dtype=torch.bfloat16)
        )
        layer._prefill_output_proj = MagicMock(  # type: ignore[assignment]
            return_value=torch.zeros(1, 2, 8, dtype=torch.bfloat16)
        )
        # Don't stub _forward_prefill_compressed this time — exercise the
        # real method with workspace_meta=None (warmup → SWA fallback).
        del layer._forward_prefill_compressed  # restore real method
        common = _make_common(cp_on=False, device=self.device)
        qkv = _make_qkv(self.device)
        x = torch.zeros(2, 4, dtype=torch.bfloat16, device=self.device)

        layer._forward_prefill_compressed(
            x,
            qkv,
            common,
            cmp_topk_runtime=None,
            compressor_meta=common.hca_meta.compressor_meta,
            workspace_meta=None,  # warmup branch — skip workspace_attn
            _skip_compressor_write=True,
        )

        layer.compressor.assert_not_called()
        layer.compressor.start_prefill.assert_not_called()
        layer.compressor.finish_prefill.assert_not_called()

    def test_default_skip_false_invokes_compressor(self) -> None:
        layer = _make_attention_stub(compress_ratio=128)
        layer._attn_fp8_swa_via_kv_full = MagicMock(  # type: ignore[assignment]
            return_value=torch.zeros(2, 8, dtype=torch.bfloat16)
        )
        layer._prefill_output_proj = MagicMock(  # type: ignore[assignment]
            return_value=torch.zeros(1, 2, 8, dtype=torch.bfloat16)
        )
        del layer._forward_prefill_compressed
        common = _make_common(cp_on=False, device=self.device)
        qkv = _make_qkv(self.device)
        x = torch.zeros(2, 4, dtype=torch.bfloat16, device=self.device)

        layer._forward_prefill_compressed(
            x,
            qkv,
            common,
            cmp_topk_runtime=None,
            compressor_meta=common.hca_meta.compressor_meta,
            workspace_meta=None,
        )

        # Baseline path: one synchronous compressor invocation.
        layer.compressor.assert_called_once_with(
            x, common.sp_int, meta=common.hca_meta.compressor_meta
        )


if __name__ == "__main__":
    unittest.main()
