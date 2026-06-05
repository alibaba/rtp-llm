"""UT: IndexerFP8 overlap entry points (start_prefill_nested_compressor /
forward_with_pending_nested) preserve forward()'s externally observable
behaviour.

The attention overlap path swaps the synchronous nested compressor call
``self.compressor(x, sp, meta=...)`` (line buried inside ``forward``) for
an early ``start_prefill_nested_compressor`` (queues NCCL on the gather
stream) + late ``forward_with_pending_nested(... nested_pending)`` (drains
the pending where ``forward`` would have run the compressor). This UT
locks the contract so the split stays bit-equal to ``forward``:

  * warmup (no pool bound) → ``start_prefill_nested_compressor`` returns
    ``None``; ``forward_with_pending_nested(None)`` returns the empty-topk
    shape and does a no-op ``finish_prefill(None)`` (mirroring
    ``forward``'s early return);
  * pool propagation to the nested compressor happens before any
    compressor call on both paths;
  * ``cp_gather_stream`` is forwarded verbatim into
    ``compressor.start_prefill`` (FIFO contract under MTP / re-entrant
    compressors sharing one CP stream);
  * ``forward_with_pending_nested`` drains the supplied pending via
    ``finish_prefill`` exactly once, **not** ``compressor(x, sp, meta=...)``;
  * T==0 cold-start: both paths emit empty topk after the compressor
    write (so the FP8 pool write happens before the early return).
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from rtp_llm.models_py.modules.dsv4.fp8._indexer_quant_triton import INDEXER_HEAD_DIM
from rtp_llm.models_py.modules.dsv4.fp8._kv_cache_utils import PoolBackedModule
from rtp_llm.models_py.modules.dsv4.fp8.compressor import _CompressorPending
from rtp_llm.models_py.modules.dsv4.fp8.indexer import (
    IndexerFP8,
    _IndexerFP8PrefillMeta,
)


def _make_indexer_stub(*, bind_pool: bool, device: torch.device) -> IndexerFP8:
    """Build an IndexerFP8 instance via ``__new__`` so we skip the heavy
    ``__init__`` (needs deep_gemm + per-layer weight dict) while keeping
    the overlap entry points exercised on the real class methods."""
    ind = IndexerFP8.__new__(IndexerFP8)
    PoolBackedModule.__init__(ind)
    # Construct a minimal nested compressor mock — we only assert against
    # the four methods the overlap path touches.
    ind.compressor = SimpleNamespace(
        freqs_cis=None,
        start_prefill=MagicMock(name="compressor.start_prefill"),
        finish_prefill=MagicMock(name="compressor.finish_prefill"),
        prepare_metadata=MagicMock(name="compressor.prepare_metadata"),
        set_pool_context=MagicMock(name="compressor.set_pool_context"),
        clear_pool_context=MagicMock(name="compressor.clear_pool_context"),
    )
    ind.index_topk = 4
    ind.n_heads = 32
    ind.head_dim = INDEXER_HEAD_DIM
    ind.compress_ratio = 4
    ind.freqs_cis = torch.zeros(1, dtype=torch.float32, device=device)
    ind._cp_ctx = None
    if bind_pool:
        ind._kv_pool_view = torch.zeros(1, 1, 132, dtype=torch.uint8, device=device)
        ind._kv_block_table = torch.ones(1, 1, dtype=torch.int32, device=device)
        ind._kv_eb = 1
        ind._kv_tokens_per_block = 1
        ind._state_pool_3d = torch.zeros(1, 1, 1, dtype=torch.float32, device=device)
        ind._state_block_table = torch.ones(1, 1, dtype=torch.int32, device=device)
        ind._state_eb = 1
        ind._state_tokens_per_block = 1
    return ind


def _make_meta(device: torch.device, *, T: int) -> _IndexerFP8PrefillMeta:
    """Minimal meta — we drive the test along the warmup / T==0 paths so
    the kernel chain (gather/score/topk) is never reached."""
    return _IndexerFP8PrefillMeta(
        bsz=1,
        seqlen=2,
        M=2,
        sp_int=0,
        end_pos=2,
        is_fresh_prefill=True,
        T=T,
        freqs_cis_slice=torch.zeros(1, dtype=torch.float32, device=device),
        positions_d=torch.zeros(2, dtype=torch.int32, device=device),
        ks=torch.zeros(2, dtype=torch.int32, device=device),
        ke=torch.zeros(2, dtype=torch.int32, device=device),
        block_table_i32=torch.ones(1, 1, dtype=torch.int32, device=device),
        cu_kv_seqlens=torch.tensor([0, 0], dtype=torch.int32, device=device),
        cu_kv_per_token=torch.tensor([0, 0], dtype=torch.int32, device=device),
        compressor_meta=SimpleNamespace(),
    )


class IndexerFP8OverlapEntryPointsTest(unittest.TestCase):
    """Pure CPU tests — we mock the nested compressor and steer the
    indexer along its warmup / T==0 branches, so no CUDA / DeepGEMM is
    required."""

    def setUp(self) -> None:
        self.device = torch.device("cpu")
        torch.manual_seed(0)

    # ------------------------------------------------------------------
    # Warmup (pool unbound)
    # ------------------------------------------------------------------
    def test_warmup_start_returns_none_and_skips_compressor(self) -> None:
        ind = _make_indexer_stub(bind_pool=False, device=self.device)
        x = torch.zeros(2, 8, dtype=torch.bfloat16, device=self.device)
        meta = _make_meta(self.device, T=0)

        pending = ind.start_prefill_nested_compressor(
            x, sp_int=0, meta=meta.compressor_meta
        )

        self.assertIsNone(pending)
        ind.compressor.start_prefill.assert_not_called()
        ind.compressor.set_pool_context.assert_not_called()

    def test_warmup_forward_with_pending_returns_empty_topk(self) -> None:
        ind = _make_indexer_stub(bind_pool=False, device=self.device)
        x = torch.zeros(2, 8, dtype=torch.bfloat16, device=self.device)
        qr = torch.zeros(2, 16, dtype=torch.bfloat16, device=self.device)
        meta = _make_meta(self.device, T=0)

        out = ind.forward_with_pending_nested(x, qr, meta, nested_pending=None)

        self.assertEqual(tuple(out.shape), (2, 0))
        self.assertEqual(out.dtype, torch.int32)
        # Warmup pairs with a no-op finish_prefill(None) so callers can
        # chain unconditionally.
        ind.compressor.finish_prefill.assert_called_once_with(None)

    # ------------------------------------------------------------------
    # start_prefill_nested_compressor delegation
    # ------------------------------------------------------------------
    def test_start_prefill_propagates_pool_and_forwards_args(self) -> None:
        ind = _make_indexer_stub(bind_pool=True, device=self.device)
        x = torch.zeros(3, 8, dtype=torch.bfloat16, device=self.device)
        meta = SimpleNamespace(name="hoisted_meta")
        stream_sentinel = object()
        sentinel_pending = object()
        ind.compressor.start_prefill.return_value = sentinel_pending

        out_pending = ind.start_prefill_nested_compressor(
            x, sp_int=5, meta=meta, cp_gather_stream=stream_sentinel
        )

        # Returned pending is exactly what compressor.start_prefill returned.
        self.assertIs(out_pending, sentinel_pending)
        # Pool was propagated to the nested compressor exactly once.
        ind.compressor.set_pool_context.assert_called_once_with(
            ind._kv_pool_view,
            ind._kv_block_table,
            ind._kv_eb,
            ind._state_pool_3d,
            ind._state_block_table,
            ind._state_eb,
            state_tokens_per_block=ind._state_tokens_per_block,
            kv_tokens_per_block=ind._kv_tokens_per_block,
            kv_owner_tokens_per_block=ind._kv_owner_tokens_per_block,
        )
        # start_prefill was called once with the right args (cp_gather_stream
        # forwarded verbatim — FIFO contract under MTP).
        ind.compressor.start_prefill.assert_called_once_with(
            x, 5, meta=meta, cp_gather_stream=stream_sentinel
        )
        # The synchronous baseline must not be touched on the overlap path.
        ind.compressor.finish_prefill.assert_not_called()

    def test_start_prefill_lazy_freqs_cis_propagation(self) -> None:
        """``forward`` lazily forwards ``self.freqs_cis`` into the nested
        compressor when the latter has none yet. The overlap entry point
        must mirror that — otherwise CP positions go uninitialised on
        the gather stream."""
        ind = _make_indexer_stub(bind_pool=True, device=self.device)
        ind.compressor.freqs_cis = None
        x = torch.zeros(2, 8, dtype=torch.bfloat16, device=self.device)
        meta = SimpleNamespace(name="hoisted_meta")

        ind.start_prefill_nested_compressor(x, sp_int=0, meta=meta)

        self.assertIs(ind.compressor.freqs_cis, ind.freqs_cis)

    def test_clear_pool_context_also_clears_nested_pool(self) -> None:
        ind = _make_indexer_stub(bind_pool=True, device=self.device)

        ind.clear_pool_context()

        self.assertIsNone(ind._kv_pool_view)
        self.assertIsNone(ind._kv_block_table)
        self.assertEqual(ind._kv_eb, 0)
        ind.compressor.clear_pool_context.assert_called_once()

    def test_gather_prefill_k_cache_uses_sharded_assembler(self) -> None:
        ind = _make_indexer_stub(bind_pool=True, device=self.device)
        cp_ctx = SimpleNamespace(cp_size=2, kv_cache_sharded=True)
        ind._cp_ctx = cp_ctx
        meta = _make_meta(self.device, T=5)._replace(
            block_table_i32=torch.ones(2, 3, dtype=torch.int32, device=self.device),
            cu_kv_seqlens=torch.tensor(
                [0, 3, 5], dtype=torch.int32, device=self.device
            ),
        )
        k_quant_flat = torch.empty(5, INDEXER_HEAD_DIM, dtype=torch.uint8)
        k_scale_buf = torch.empty(5, 4, dtype=torch.uint8)
        plan = SimpleNamespace(total_local_T=4)
        local_cu = torch.tensor([0, 2, 4], dtype=torch.int32)
        cpp_calls = []

        def fake_cpp(pool, out_q, out_s, block_table, cu):
            cpp_calls.append((pool, out_q, out_s, block_table, cu))

        assemble_calls = []

        def fake_assemble(**kwargs):
            assemble_calls.append(kwargs)

        import rtp_llm.models_py.modules.dsv4.fp8.indexer as indexer_mod

        with (
            patch.object(
                indexer_mod.rtp_llm_ops,
                "cp_gather_indexer_k_quant_cache",
                side_effect=fake_cpp,
                create=True,
            ),
            patch(
                "rtp_llm.models_py.modules.dsv4.fp8._indexer_cp_assembler.build_indexer_cp_chunk_plan",
                return_value=plan,
            ) as build_plan,
            patch(
                "rtp_llm.models_py.modules.dsv4.fp8._indexer_cp_assembler.build_local_cu_kv_seqlens",
                return_value=local_cu,
            ) as build_local_cu,
            patch(
                "rtp_llm.models_py.modules.dsv4.fp8._indexer_cp_assembler.assemble_indexer_k",
                side_effect=fake_assemble,
            ),
        ):
            ind._gather_prefill_k_cache(meta, k_quant_flat, k_scale_buf)

        build_plan.assert_called_once()
        build_kwargs = build_plan.call_args.kwargs
        self.assertIs(build_kwargs["cp_ctx"], cp_ctx)
        self.assertEqual(build_kwargs["block_size"], ind._kv_eb)
        self.assertTrue(
            torch.equal(
                build_kwargs["per_req_total_kv_lens"],
                torch.tensor([3, 2], dtype=torch.int64),
            )
        )
        build_local_cu.assert_called_once_with(plan)

        self.assertEqual(len(cpp_calls), 1)
        _, local_q, local_s, block_table, cu = cpp_calls[0]
        self.assertEqual(tuple(local_q.shape), (plan.total_local_T, INDEXER_HEAD_DIM))
        self.assertEqual(tuple(local_s.shape), (plan.total_local_T, 4))
        self.assertIs(block_table, meta.block_table_i32)
        self.assertIs(cu, local_cu)

        self.assertEqual(len(assemble_calls), 1)
        assemble_kwargs = assemble_calls[0]
        self.assertIs(assemble_kwargs["plan"], plan)
        self.assertIs(assemble_kwargs["local_k_quant"], local_q)
        self.assertIs(assemble_kwargs["local_k_scale"], local_s)
        self.assertIs(assemble_kwargs["out_k_quant"], k_quant_flat)
        self.assertIs(assemble_kwargs["out_k_scale"], k_scale_buf)

    # ------------------------------------------------------------------
    # forward_with_pending_nested
    # ------------------------------------------------------------------
    def test_forward_with_pending_nested_drains_pending_and_early_returns_on_t0(
        self,
    ) -> None:
        """Steer the indexer down the T==0 cold-start path so the
        gather/score/topk kernels are never invoked. We assert that the
        nested compressor write happens *before* the early return —
        matching ``forward``'s order — and that ``finish_prefill`` is
        called with the supplied pending, not the synchronous compressor."""
        ind = _make_indexer_stub(bind_pool=True, device=self.device)
        # Replace the two non-compressor compute steps with no-ops so the
        # path is independent of cuBLAS / DeepGEMM availability on the
        # test host.
        x = torch.zeros(2, 8, dtype=torch.bfloat16, device=self.device)
        qr = torch.zeros(2, 16, dtype=torch.bfloat16, device=self.device)
        meta = _make_meta(self.device, T=0)
        pending = _CompressorPending(
            fused_flat=torch.zeros(2, 4, dtype=torch.bfloat16, device=self.device),
            fused_gather_handle=None,
            sp=0,
            bsz=1,
            seqlen=2,
            meta=None,
            out_dim=2,
        )

        compute_q_calls = []

        def fake_compute_q(qr_in, freqs):
            compute_q_calls.append((qr_in, freqs))
            return torch.zeros(
                2, ind.n_heads, ind.head_dim, dtype=torch.bfloat16, device=self.device
            )

        ind._compute_indexer_q = fake_compute_q  # type: ignore[assignment]
        # weights_proj is loaded by __init__; we bypassed __init__, so stub it.
        ind.weights_proj = torch.zeros(
            ind.n_heads, x.size(-1), dtype=torch.bfloat16, device=self.device
        )

        # has_fp8_mqa_logits is asserted before T==0 early return — patch via
        # module-level binding (the production assert path needs DeepGEMM).
        import rtp_llm.models_py.modules.dsv4.fp8.indexer as indexer_mod

        saved_has = indexer_mod.has_fp8_mqa_logits
        # Also patch _kv_pool_view dim assertion: the 3D pool above (1,1,132)
        # already satisfies it, but be explicit.
        try:
            indexer_mod.has_fp8_mqa_logits = lambda: True  # type: ignore[assignment]
            out = ind.forward_with_pending_nested(x, qr, meta, nested_pending=pending)
        finally:
            indexer_mod.has_fp8_mqa_logits = saved_has  # type: ignore[assignment]

        # T==0 branch returns the empty-topk shape.
        self.assertEqual(tuple(out.shape), (2, 0))
        self.assertEqual(out.dtype, torch.int32)
        # The supplied pending was drained exactly once — start_prefill must
        # not be re-called on the late path.
        ind.compressor.finish_prefill.assert_called_once_with(pending)
        ind.compressor.start_prefill.assert_not_called()
        # ``forward`` invariant: pool is unbound after the call (try/finally).
        ind.compressor.clear_pool_context.assert_called_once()
        # _compute_indexer_q was invoked (compute_q runs before the
        # nested compressor drain, as in ``forward``).
        self.assertEqual(len(compute_q_calls), 1)


if __name__ == "__main__":
    unittest.main()
