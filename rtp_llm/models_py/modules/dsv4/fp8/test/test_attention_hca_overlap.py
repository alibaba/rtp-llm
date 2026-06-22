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

  * Process-wide CP streams are cached + reused: serialized compressor/cache
    communication, SWA kv_full communication, and post-gather local work.
"""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

import rtp_llm.models_py.modules.dsv4.fp8.attention as attention_mod
from rtp_llm.models_py.modules.dsv4.fp8.attention import (
    AttentionFP8,
    PrefillMeta,
    PrefillQKV,
    WorkspaceMeta,
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
    layer.tp_size = 1
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

    def test_workspace_read_async_requires_overlap_env_and_page_rr(self) -> None:
        layer = _make_attention_stub(compress_ratio=128)
        common = _make_common(cp_on=True, device=torch.device("cuda"))
        common.cp_ctx.kv_cache_sharded = True
        layer._cp_ctx = common.cp_ctx

        with self._with_env("0"):
            self.assertFalse(layer._should_async_workspace_reads_for_prefill(common))

        common.cp_ctx.kv_cache_sharded = False
        with self._with_env("1"):
            self.assertFalse(layer._should_async_workspace_reads_for_prefill(common))

        common.cp_ctx.kv_cache_sharded = True
        with self._with_env("1"):
            self.assertTrue(layer._should_async_workspace_reads_for_prefill(common))


@unittest.skipUnless(torch.cuda.is_available(), "stream allocation requires CUDA")
class HCAOverlapStreamCacheTest(unittest.TestCase):
    """Process-wide stream reuse and SWA/compressor stream separation."""

    def setUp(self) -> None:
        self._cp_streams_patch = patch.object(attention_mod, "_CP_GATHER_STREAMS", {})
        self._swa_streams_patch = patch.object(
            attention_mod,
            "_SWA_CP_GATHER_STREAMS",
            {},
        )
        self._cp_streams_patch.start()
        self._swa_streams_patch.start()

    def tearDown(self) -> None:
        self._swa_streams_patch.stop()
        self._cp_streams_patch.stop()

    def test_returns_same_stream_on_repeated_calls(self) -> None:
        layer = _make_attention_stub(compress_ratio=128)
        device = torch.device("cuda")
        s1 = layer._get_cp_gather_stream(device)
        s2 = layer._get_cp_gather_stream(device)
        self.assertIs(s1, s2)
        self.assertIsInstance(s1, torch.cuda.Stream)

    def test_cp_stream_is_shared_across_layer_instances(self) -> None:
        layer1 = _make_attention_stub(compress_ratio=128)
        layer2 = _make_attention_stub(compress_ratio=4)
        device = torch.device("cuda")

        self.assertIs(
            layer1._get_cp_gather_stream(device),
            layer2._get_cp_gather_stream(device),
        )

    def test_swa_stream_is_shared_but_separate_from_compressor_stream(self) -> None:
        layer = _make_attention_stub(compress_ratio=128)
        other_layer = _make_attention_stub(compress_ratio=4)
        device = torch.device("cuda")
        compressor_stream = layer._get_cp_gather_stream(device)
        swa_stream_1 = layer._get_swa_cp_gather_stream(device)
        swa_stream_2 = layer._get_swa_cp_gather_stream(device)

        self.assertIs(swa_stream_1, swa_stream_2)
        self.assertIs(swa_stream_1, other_layer._get_swa_cp_gather_stream(device))
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
            return_value=torch.zeros(2, 8, dtype=torch.bfloat16)
        )
        layer._prefill_output_all_reduce = MagicMock(  # type: ignore[assignment]
            name="_prefill_output_all_reduce"
        )
        # Don't stub _forward_prefill_compressed this time — exercise the
        # real method with workspace_meta=None (warmup → SWA fallback).
        del layer._forward_prefill_compressed  # restore real method
        common = _make_common(cp_on=False, device=self.device)
        qkv = _make_qkv(self.device)
        x = torch.zeros(2, 4, dtype=torch.bfloat16, device=self.device)

        out = layer._forward_prefill_compressed(
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
        layer._prefill_output_all_reduce.assert_called_once()
        self.assertEqual(
            tuple(layer._prefill_output_all_reduce.call_args.args[0].shape),
            (2, 8),
        )
        self.assertEqual(tuple(out.shape), (2, 8))

    def test_default_skip_false_invokes_compressor(self) -> None:
        layer = _make_attention_stub(compress_ratio=128)
        layer._attn_fp8_swa_via_kv_full = MagicMock(  # type: ignore[assignment]
            return_value=torch.zeros(2, 8, dtype=torch.bfloat16)
        )
        layer._prefill_output_proj = MagicMock(  # type: ignore[assignment]
            return_value=torch.zeros(2, 8, dtype=torch.bfloat16)
        )
        layer._prefill_output_all_reduce = MagicMock(  # type: ignore[assignment]
            name="_prefill_output_all_reduce"
        )
        del layer._forward_prefill_compressed
        common = _make_common(cp_on=False, device=self.device)
        qkv = _make_qkv(self.device)
        x = torch.zeros(2, 4, dtype=torch.bfloat16, device=self.device)

        out = layer._forward_prefill_compressed(
            x,
            qkv,
            common,
            cmp_topk_runtime=None,
            compressor_meta=common.hca_meta.compressor_meta,
            workspace_meta=None,
        )

        # Baseline path: one synchronous compressor invocation. The per-forward
        # prefill workspace is threaded through to the compressor (None here,
        # since this CPU test doesn't bind one).
        layer.compressor.assert_called_once_with(
            x,
            common.sp_int,
            meta=common.hca_meta.compressor_meta,
            workspace=common.workspace,
        )
        layer._prefill_output_all_reduce.assert_called_once()
        self.assertEqual(
            tuple(layer._prefill_output_all_reduce.call_args.args[0].shape),
            (2, 8),
        )
        self.assertEqual(tuple(out.shape), (2, 8))

    def test_workspace_branch_returns_projected_output_without_outer_projection(
        self,
    ) -> None:
        layer = _make_attention_stub(compress_ratio=128)
        projected = torch.zeros(2, 8, dtype=torch.bfloat16)
        layer._attn_via_workspace = MagicMock(  # type: ignore[assignment]
            return_value=projected
        )
        layer._prefill_output_proj = MagicMock(  # type: ignore[assignment]
            name="_prefill_output_proj"
        )
        del layer._forward_prefill_compressed
        common = _make_common(cp_on=False, device=self.device)
        qkv = _make_qkv(self.device)
        x = torch.zeros(2, 4, dtype=torch.bfloat16, device=self.device)
        workspace_meta = SimpleNamespace(name="workspace_meta")

        out = layer._forward_prefill_compressed(
            x,
            qkv,
            common,
            cmp_topk_runtime=None,
            compressor_meta=common.hca_meta.compressor_meta,
            workspace_meta=workspace_meta,
        )

        layer._attn_via_workspace.assert_called_once_with(
            qkv,
            common,
            workspace_meta,
            None,
        )
        layer._prefill_output_proj.assert_not_called()
        self.assertEqual(tuple(out.shape), (2, 8))


class WorkspaceStreamingOutputProjectionTest(unittest.TestCase):
    def test_workspace_streams_flash_chunks_into_projected_output(self) -> None:
        layer = _make_attention_stub(compress_ratio=128)
        layer.head_dim = 2
        layer.dim = 8
        layer.softmax_scale = 1.0
        layer.attn_sink = None
        layer.window_size = 16
        layer._pool_view_3d_fp8 = MagicMock(  # type: ignore[assignment]
            return_value=torch.empty(1, 1, 1, dtype=torch.uint8)
        )
        layer._prefill_output_all_reduce = MagicMock(  # type: ignore[assignment]
            name="_prefill_output_all_reduce"
        )

        common = PrefillMeta(
            seqlen=5,
            seqlen_full=5,
            rd=0,
            device=torch.device("cpu"),
            cp_ctx=None,
            cp_on=False,
            freqs_cis=torch.arange(10, dtype=torch.float32).view(5, 2),
            topk_idxs=torch.zeros(5, 1, dtype=torch.int32),
            sp_int=0,
            any_cont=False,
            row_seqlens_full=torch.tensor([5], dtype=torch.long),
            batch_size=1,
        )
        qkv = PrefillQKV(
            qr=torch.zeros(5, 2, dtype=torch.bfloat16),
            q=torch.arange(10, dtype=torch.float32).view(5, 1, 2).to(torch.bfloat16),
            kv_full=torch.arange(10, dtype=torch.float32).view(5, 2),
        )
        workspace_meta = WorkspaceMeta(
            M=5,
            N=0,
            swa_eb=1,
            cmp_eb=1,
            swa_bt_int32=torch.zeros(1, 1, dtype=torch.int32),
            cmp_bt_int32=torch.zeros(1, 1, dtype=torch.int32),
            swa_seq_lens=torch.tensor([5], dtype=torch.int32),
            cmp_seq_lens=torch.tensor([0], dtype=torch.int32),
            swa_gather_lens=torch.tensor([5], dtype=torch.int32),
            swa_cache_seq_lens=torch.tensor([0], dtype=torch.int32),
            swa_cache_gather_lens=torch.tensor([0], dtype=torch.int32),
            qsl=torch.tensor([0, 5], dtype=torch.int32),
            dense_cmp_topk=torch.zeros(5, 1, dtype=torch.int32),
            new_k_slot_in_flat=torch.arange(5, dtype=torch.long),
        )
        flash_q_shapes = []

        def fake_flash_mla_sparse_fwd(q, kv, indices, sm_scale, attn_sink, topk_length):
            flash_q_shapes.append(tuple(q.shape))
            fill = float(len(flash_q_shapes))
            return (
                torch.full(
                    (q.shape[0], 1, 2),
                    fill,
                    dtype=torch.bfloat16,
                    device=q.device,
                ),
                None,
                None,
            )

        def fake_combine_topk_swa_indices(**kwargs):
            return (
                torch.zeros(5, 1, dtype=torch.int32),
                torch.ones(5, dtype=torch.int32),
            )

        projected_chunks = []

        def fake_output_proj_into(o, freqs_cis, *, out):
            projected_chunks.append(
                (
                    tuple(o.shape),
                    int(freqs_cis.shape[0]),
                    freqs_cis.clone(),
                    tuple(out.shape),
                    out.is_contiguous(),
                )
            )
            out.fill_(len(projected_chunks))

        layer._prefill_output_proj_into = MagicMock(  # type: ignore[assignment]
            side_effect=fake_output_proj_into
        )

        with patch.dict(
            os.environ,
            {"DSV4_FLASH_MLA_SPARSE_Q_CHUNK": "2"},
        ), patch.dict(
            "sys.modules",
            {
                "flash_mla": SimpleNamespace(
                    flash_mla_sparse_fwd=fake_flash_mla_sparse_fwd
                ),
                "rtp_llm.models_py.modules.dsv4.fp8._swa_dequant_triton": SimpleNamespace(),
                "rtp_llm.models_py.modules.dsv4.fp8._swa_ops_triton": SimpleNamespace(
                    combine_topk_swa_indices=fake_combine_topk_swa_indices,
                    combine_topk_swa_indices_cp=MagicMock(
                        name="combine_topk_swa_indices_cp"
                    ),
                ),
            },
        ):
            out = layer._attn_via_workspace(
                qkv,
                common,
                workspace_meta,
                cmp_topk_runtime=None,
            )

        self.assertEqual(tuple(out.shape), (5, 8))
        self.assertEqual(flash_q_shapes, [(2, 1, 2), (2, 1, 2), (1, 1, 2)])
        self.assertEqual(layer._prefill_output_proj_into.call_count, 3)
        self.assertEqual(
            [chunk[0] for chunk in projected_chunks],
            [(2, 1, 2), (2, 1, 2), (1, 1, 2)],
        )
        self.assertEqual([chunk[1] for chunk in projected_chunks], [2, 2, 1])
        self.assertEqual(
            [chunk[3] for chunk in projected_chunks],
            [(2, 8), (2, 8), (1, 8)],
        )
        self.assertTrue(all(chunk[4] for chunk in projected_chunks))
        self.assertTrue(torch.equal(projected_chunks[0][2], common.freqs_cis[0:2]))
        self.assertTrue(torch.equal(projected_chunks[1][2], common.freqs_cis[2:4]))
        self.assertTrue(torch.equal(projected_chunks[2][2], common.freqs_cis[4:5]))
        self.assertTrue(torch.all(out[0:2] == 1))
        self.assertTrue(torch.all(out[2:4] == 2))
        self.assertTrue(torch.all(out[4:5] == 3))
        layer._prefill_output_all_reduce.assert_called_once_with(out)


class PrefillOutputProjectionContractTest(unittest.TestCase):
    def test_out_is_2d_and_forwarded_to_wo_b(self) -> None:
        layer = AttentionFP8.__new__(AttentionFP8)
        torch.nn.Module.__init__(layer)
        layer.n_heads = 2
        layer.head_dim = 4
        layer.dim = 6
        layer.n_groups = 1
        layer.rope_head_dim = 2

        device = torch.device("cpu")
        seqlen = 3
        freqs_cis = torch.zeros(seqlen, 2, dtype=torch.float32, device=device)
        o = torch.randn(
            seqlen,
            layer.n_heads,
            layer.head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        o_proj = torch.randn(1, seqlen, 1, 5, dtype=torch.bfloat16, device=device)
        layer._wo_a_einsum_from_fp8 = MagicMock(  # type: ignore[assignment]
            return_value=o_proj
        )
        out = torch.empty(seqlen, layer.dim, dtype=torch.bfloat16, device=device)
        expected = (
            torch.arange(seqlen * layer.dim, dtype=torch.float32, device=device)
            .reshape(seqlen, layer.dim)
            .to(torch.bfloat16)
        )
        wo_b_calls = []

        def fake_wo_b(x: torch.Tensor, *, out: torch.Tensor | None = None):
            wo_b_calls.append((x, out))
            self.assertIs(out, expected_out)
            out.copy_(expected)
            return out

        expected_out = out
        layer.wo_b = fake_wo_b  # type: ignore[assignment]
        fused_ret = (
            torch.empty(0, dtype=torch.bfloat16, device=device),
            torch.empty(0, dtype=torch.float32, device=device),
        )

        with patch.object(
            attention_mod,
            "fused_inv_rope_fp8_quant",
            return_value=fused_ret,
        ) as fused:
            ret = layer._prefill_output_proj_into(o, freqs_cis, out=out)

        self.assertIsNone(ret)
        self.assertTrue(torch.equal(out, expected))
        fused.assert_called_once()
        fused_args, fused_kwargs = fused.call_args
        self.assertEqual(tuple(fused_args[0].shape), (seqlen, 2, 4))
        self.assertEqual(tuple(fused_args[1].shape), (seqlen, 2))
        self.assertEqual(fused_kwargs["n_groups"], 1)
        layer._wo_a_einsum_from_fp8.assert_called_once_with(
            fused_ret[0], fused_ret[1], 1, seqlen
        )
        self.assertEqual(len(wo_b_calls), 1)
        self.assertEqual(tuple(wo_b_calls[0][0].shape), (seqlen, 5))
        self.assertIs(wo_b_calls[0][1], out)


if __name__ == "__main__":
    unittest.main()
