"""Unit tests for mixed high-reuse (~90%) + cold (0%) cache-reuse prefill.

Covers mixed varlen batches (e.g. ``B=4`` with ``prefix=[4096, 2048, 0, 0]``
and ``new=[512, 256, 512, 1024]``, plus ``prefix=1280`` tiny-prefix edges):

1. IndexerFP8 chunk-local K slicing, per-chunk ``fast_topk_v2_variable``
   eligibility, and skewed descending ``row_indices`` ordering vs full-width
   reference.
2. Compressor HCA ``may_use_cache=False`` (compress-ratio-aligned prefixes)
   and CSA/Indexer mixed continuation+cold pool writes + ``run_save_partial_states``
   vs per-request ``may_use_cache=True`` references.
3. SWA/workspace Triton kernels (``combine_topk_swa_indices_cp`` regular &
   ``dense_arange_topk=True``, vectorized ``row_scatter``, flat 1D tile-grid
   ``_swa_dequant_triton`` with empty ``gather_len=0`` cold requests) plus
   AttentionFP8 Parts 1-3 orchestration contracts (CSA full overlap with
   ``before_gather_k=None``, HCA disjoint Q/gather overlap, and hoisted HCA
   ``combine_topk_swa_indices_cp`` caching).
"""

from __future__ import annotations

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from rtp_llm.models_py.modules.dsv4.cp import (
    _CP_ROLE_MAIN,
    combine_topk_swa_indices_cp_varlen,
)
from rtp_llm.models_py.modules.dsv4.fp8._compressor_consts import (
    KV_ENTRY_BYTES,
    KV_HEAD_DIM,
)
from rtp_llm.models_py.modules.dsv4.fp8._compressor_vllm_triton import (
    run_fused_compress_kv_write,
    run_save_partial_states,
)
from rtp_llm.models_py.modules.dsv4.fp8._indexer_quant_triton import (
    INDEXER_ENTRY_BYTES,
    INDEXER_HEAD_DIM,
)
from rtp_llm.models_py.modules.dsv4.fp8._indexer_score import (
    fp8_mqa_indexer_score,
    has_fp8_mqa_logits,
)
from rtp_llm.models_py.modules.dsv4.fp8._swa_dequant_triton import (
    ENTRY_BYTES as SWA_ENTRY_BYTES,
    HEAD_DIM as SWA_HEAD_DIM,
    TILE_T,
    build_dequant_cu_tiles_device,
    build_dequant_cu_tiles_from_host,
    dequantize_and_gather_k_cache,
    dequantize_and_gather_k_cache_slots,
    direct_triton_fast_path_supported,
    try_dequantize_and_gather_k_cache_slots_to_workspace,
)
from rtp_llm.models_py.modules.dsv4.fp8._swa_kv_insert_triton import (
    quantize_and_insert_k_cache,
    row_scatter,
)
from rtp_llm.models_py.modules.dsv4.fp8._swa_ops_triton import (
    combine_topk_swa_indices_cp,
)
from rtp_llm.models_py.modules.dsv4.fp8.attention import (
    AttentionFP8,
    PrefillMeta,
    PrefillQKV,
    WorkspaceMeta,
)
from rtp_llm.models_py.modules.dsv4.fp8.compressor import (
    CompressorFP8,
    _cp_sliced_state_read_needed,
    _linear_bf16_bf16_fp32,
)
from rtp_llm.models_py.modules.dsv4.fp8.indexer import (
    _FAST_PREFILL_TOPK_MAX_INPUT_TOKENS,
    _FAST_PREFILL_TOPK_OK,
    _TOPK_SKEW_MIN_ROWS,
    IndexerFP8,
    _build_score_chunks,
    _IndexerFP8PrefillMeta,
)
from rtp_llm.models_py.modules.dsv4.prefill_workspace import PrefillWorkspace

try:
    from rtp_llm.ops.compute_ops import rtp_llm_ops
except Exception:  # pragma: no cover - optional in CPU-only environments
    rtp_llm_ops = None


TOKENS_PER_STATE_BLOCK = 256


def _build_varlen_inputs(
    prefix_lengths: list[int],
    input_lengths: list[int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cu = [0]
    pos_chunks = []
    req_chunks = []
    for req_id, (sp, length) in enumerate(zip(prefix_lengths, input_lengths)):
        cu.append(cu[-1] + length)
        pos_chunks.append(
            torch.arange(sp, sp + length, device=device, dtype=torch.int64)
        )
        req_chunks.append(
            torch.full((length,), req_id, device=device, dtype=torch.int32)
        )
    return (
        torch.tensor(cu, device=device, dtype=torch.int32),
        torch.tensor(input_lengths, device=device, dtype=torch.int32),
        torch.tensor(prefix_lengths, device=device, dtype=torch.int32),
        torch.cat(pos_chunks, dim=0),
        torch.cat(req_chunks, dim=0),
    )


def _build_test_compressor(
    *,
    dim: int,
    head_dim: int,
    rope_head_dim: int,
    compress_ratio: int,
    max_batch_size: int,
    max_seq_len: int,
    device: torch.device,
    seed: int = 123,
) -> CompressorFP8:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    coff = 1 + (compress_ratio == 4)
    weights = {
        "ape": torch.randn(
            compress_ratio, coff * head_dim, dtype=torch.float32, generator=gen
        )
        * 0.1,
        "wkv": torch.randn(coff * head_dim, dim, dtype=torch.float32, generator=gen)
        * 0.05,
        "wgate": torch.randn(coff * head_dim, dim, dtype=torch.float32, generator=gen)
        * 0.05,
        "norm": torch.ones(head_dim, dtype=torch.bfloat16),
    }
    cmp = CompressorFP8(
        dim=dim,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        compress_ratio=compress_ratio,
        max_batch_size=max_batch_size,
        cp_role=_CP_ROLE_MAIN,
        norm_eps=1e-6,
        compressor_weights=weights,
    ).to(device)
    if cmp._wkv_wgate_fused is not None and cmp._wkv_wgate_fused.device != device:
        cmp._wkv_wgate_fused = cmp._wkv_wgate_fused.to(device)
    if cmp.ape.device != device:
        cmp.ape = torch.nn.Parameter(cmp.ape.data.to(device), requires_grad=False)
    cmp.init_rope_cache(
        torch.ones(max_seq_len, rope_head_dim // 2, dtype=torch.complex64, device=device)
    )
    return cmp


def _bind_compressor_pools(
    cmp: CompressorFP8,
    *,
    bsz: int,
    head_dim: int,
    coff: int,
    compress_ratio: int,
    max_seq_total: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    state_eb = TOKENS_PER_STATE_BLOCK
    state_blocks_per_req = max(1, (max_seq_total + state_eb - 1) // state_eb)
    state_total_blocks = 1 + bsz * state_blocks_per_req
    hidden = 2 * coff * head_dim
    state_view_2d = torch.zeros(
        state_total_blocks * state_eb, hidden, dtype=torch.float32, device=device
    )
    state_block_table = torch.tensor(
        [
            [1 + b * state_blocks_per_req + i for i in range(state_blocks_per_req)]
            for b in range(bsz)
        ],
        dtype=torch.int32,
        device=device,
    )

    kv_eb = TOKENS_PER_STATE_BLOCK // compress_ratio
    n_compressed = (max_seq_total + compress_ratio - 1) // compress_ratio
    kv_blocks_per_req = max(1, (n_compressed + kv_eb - 1) // kv_eb)
    kv_total_blocks = 1 + bsz * kv_blocks_per_req
    entry_bytes = INDEXER_ENTRY_BYTES if head_dim == INDEXER_HEAD_DIM else KV_ENTRY_BYTES
    kv_pool_3d = torch.zeros(
        kv_total_blocks, kv_eb, entry_bytes, dtype=torch.uint8, device=device
    )
    kv_block_table = torch.tensor(
        [
            [1 + b * kv_blocks_per_req + i for i in range(kv_blocks_per_req)]
            for b in range(bsz)
        ],
        dtype=torch.int32,
        device=device,
    )
    cmp.set_pool_context(
        kv_pool_view=kv_pool_3d,
        kv_block_table=kv_block_table,
        kv_eb=kv_eb,
        state_pool_view=state_view_2d,
        state_block_table=state_block_table,
        state_eb=state_eb,
        state_tokens_per_block=state_eb,
        kv_tokens_per_block=kv_eb * compress_ratio,
    )
    return state_view_2d, kv_pool_3d


class IndexerMixedCacheReuseTest(unittest.TestCase):
    """Chunk-local K slicing, per-chunk fast_topk eligibility, and skewed row_indices."""

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        self.device = torch.device("cuda")

    def test_mixed_reuse_chunk_slicing_and_fast_topk_eligibility(self) -> None:
        ratio = 4
        topk = 2048
        # Req 0 exceeds 12K fast_topk threshold; Req 1 (1280) and Reqs 2-3 (0 prefix)
        # remain fast_topk eligible and must stay decoupled from Req 0's K span.
        prefixes = [12800, 1280, 0, 0]
        inputs = [256, 512, 512, 1024]
        cu_seqlens, input_lengths, prefix_lengths, position_ids, req_ids = (
            _build_varlen_inputs(prefixes, inputs, self.device)
        )
        T_per_req_host = [(sp + ln) // ratio for sp, ln in zip(prefixes, inputs)]
        cu_q_host = cu_seqlens.tolist()
        cu_k_host = [0]
        for t_b in T_per_req_host:
            cu_k_host.append(cu_k_host[-1] + t_b)
        T_total = sum(T_per_req_host)
        M = int(position_ids.numel())

        stub = IndexerFP8.__new__(IndexerFP8)
        stub.compress_ratio = ratio
        stub.n_heads = 64
        stub.head_dim = INDEXER_HEAD_DIM
        stub.index_topk = topk
        stub.freqs_cis = torch.ones(
            16384, 32, dtype=torch.complex64, device=self.device
        )
        stub._kv_block_table = torch.ones(
            len(prefixes), 1, dtype=torch.int32, device=self.device
        )
        stub._kv_eb = 16
        stub._kv_cache_t = T_total
        stub._cp_ctx = None
        stub._kv_pool_view = None
        stub._state_block_table = None
        stub._state_eb = 0

        meta = IndexerFP8.prepare(
            stub,
            len(prefixes),
            M,
            prefixes[0],
            self.device,
            kv_block_table=stub._kv_block_table,
            kv_eb=stub._kv_eb,
            use_varlen=True,
            batch_size=len(prefixes),
            cu_seqlens=cu_seqlens,
            input_lengths=input_lengths,
            prefix_lengths=prefix_lengths,
            position_ids=position_ids,
            req_id_per_token=req_ids,
            has_prefix=True,
        )
        self.assertIsNotNone(meta.score_chunks)
        chunks = meta.score_chunks
        assert chunks is not None

        # Verify contiguous partition of [0, M) and tight chunk-local K rebasing.
        self.assertEqual(chunks[0].row_start, 0)
        self.assertEqual(chunks[-1].row_end, M)
        for idx, c in enumerate(chunks):
            if idx > 0:
                self.assertEqual(c.row_start, chunks[idx - 1].row_end)
            self.assertGreaterEqual(c.k_lo, 0)
            self.assertLessEqual(c.k_hi, T_total)
            ks_raw = meta.ks[c.row_start : c.row_end]
            ke_raw = meta.ke[c.row_start : c.row_end]
            self.assertTrue(torch.equal(c.ks_chunk, ks_raw - c.k_lo))
            self.assertTrue(torch.equal(c.ke_chunk, ke_raw - c.k_lo))

        if _FAST_PREFILL_TOPK_OK:
            fast_k_cap = _FAST_PREFILL_TOPK_MAX_INPUT_TOKENS // ratio
            # Req 0 (rows 0..256) exceeds fast_k_cap and must not merge into Req 1..3.
            req0_chunks = [c for c in chunks if c.row_start < inputs[0]]
            cold_chunks = [c for c in chunks if c.row_start >= inputs[0]]
            self.assertTrue(all(c.row_end <= inputs[0] for c in req0_chunks))
            self.assertTrue(
                all(c.max_row_len * ratio <= fast_k_cap * ratio for c in cold_chunks)
            )
            self.assertTrue(any(c.max_row_len * ratio > fast_k_cap * ratio for c in req0_chunks))

        # Skewed cold/tiny-prefix batch where max_len > 2 * min_len triggers
        # descending row_indices ordering for kernel warp load balance.
        skew_rows = max(_TOPK_SKEW_MIN_ROWS, 512)
        ks_skew = torch.zeros(skew_rows, dtype=torch.int32, device=self.device)
        ke_skew = (
            torch.arange(1, skew_rows + 1, dtype=torch.int32, device=self.device) // 2
        ).clamp_min(1)
        skew_chunks = _build_score_chunks(
            ks_skew,
            ke_skew,
            M=skew_rows,
            T=int(ke_skew.max().item()),
            ratio=ratio,
            topk=topk,
            use_varlen=False,
            sp_int=0,
            cu_q_host=[0, skew_rows],
            cu_k_host=[0, int(ke_skew.max().item())],
            T_per_req_host=[int(ke_skew.max().item())],
        )
        self.assertTrue(any(c.row_indices is not None for c in skew_chunks))
        for c in skew_chunks:
            if c.row_indices is not None:
                lens = (c.ke_chunk - c.ks_chunk)[c.row_indices.to(torch.long)]
                self.assertTrue(bool((lens[:-1] >= lens[1:]).all().item()))

    def test_mixed_reuse_score_and_topk_matches_full_width_ref(self) -> None:
        if rtp_llm_ops is None or not has_fp8_mqa_logits():
            self.skipTest("rtp_llm_ops and fp8_mqa_logits required")

        torch.manual_seed(11)
        ratio = 4
        topk = 256
        # Classic mixed reuse: ~90% reuse + cold 0% reuse + tiny 1280 prefix.
        prefixes = [4096, 2048, 1280, 0, 0]
        inputs = [512, 256, 256, 512, 1024]
        cu_seqlens, input_lengths, prefix_lengths, position_ids, req_ids = (
            _build_varlen_inputs(prefixes, inputs, self.device)
        )
        T_per_req_host = [(sp + ln) // ratio for sp, ln in zip(prefixes, inputs)]
        T_total = sum(T_per_req_host)
        M = int(position_ids.numel())

        # Build well-conditioned FP8 inputs so scores vary smoothly across keys.
        q_fp8 = torch.randn(
            M, 64, INDEXER_HEAD_DIM, device=self.device, dtype=torch.bfloat16
        ).to(torch.float8_e4m3fn)
        w_fold = torch.randn(M, 64, device=self.device, dtype=torch.float32)
        k_quant = torch.randn(
            T_total, INDEXER_HEAD_DIM, device=self.device, dtype=torch.bfloat16
        ).to(torch.float8_e4m3fn)
        k_scale = torch.rand(T_total, dtype=torch.float32, device=self.device) + 0.25

        # Naive unchunked full-width score tensor shared across topk=256 and topk=512.
        stub_scores_meta = IndexerFP8.__new__(IndexerFP8)
        stub_scores_meta.compress_ratio = ratio
        stub_scores_meta.n_heads = 64
        stub_scores_meta.head_dim = INDEXER_HEAD_DIM
        stub_scores_meta.freqs_cis = torch.ones(
            8192, 32, dtype=torch.complex64, device=self.device
        )
        stub_scores_meta._kv_block_table = torch.ones(
            len(prefixes), 1, dtype=torch.int32, device=self.device
        )
        stub_scores_meta._kv_eb = 16
        stub_scores_meta._kv_cache_t = T_total
        stub_scores_meta._cp_ctx = None
        stub_scores_meta._kv_pool_view = None
        stub_scores_meta._state_block_table = None
        stub_scores_meta._state_eb = 0

        for topk in (256, 512):
            stub_scores_meta.index_topk = topk
            meta = IndexerFP8.prepare(
                stub_scores_meta,
                len(prefixes),
                M,
                prefixes[0],
                self.device,
                kv_block_table=stub_scores_meta._kv_block_table,
                kv_eb=stub_scores_meta._kv_eb,
                use_varlen=True,
                batch_size=len(prefixes),
                cu_seqlens=cu_seqlens,
                input_lengths=input_lengths,
                prefix_lengths=prefix_lengths,
                position_ids=position_ids,
                req_id_per_token=req_ids,
                has_prefix=True,
            )

            out_chunked = stub_scores_meta._score_and_topk_prefill(
                q_fp8, w_fold, k_quant, k_scale, meta, (M, topk)
            )
            full_logits = fp8_mqa_indexer_score(
                q_fp8, w_fold, k_quant, k_scale, meta.ks, meta.ke, clean_logits=False
            )
            out_ref = torch.empty((M, topk), dtype=torch.int32, device=self.device)
            rtp_llm_ops.dsv4_top_k_per_row_prefill(
                full_logits,
                meta.ks,
                meta.ke,
                out_ref,
                M,
                full_logits.stride(0),
                full_logits.stride(1),
                topk,
                True,
            )

            # Valid count, bounds, padding (-1), and sorted Top-K scores/indices must match.
            self.assertEqual(out_chunked.dtype, torch.int32)
            self.assertEqual(out_chunked.shape, out_ref.shape)
            valid_counts_ref = (meta.ke - meta.ks).clamp_max(topk)
            for row in range(M):
                vc = int(valid_counts_ref[row].item())
                ks_r = int(meta.ks[row].item())
                got_row = out_chunked[row]
                ref_row = out_ref[row]
                self.assertTrue(bool((got_row[:vc] >= 0).all().item()))
                if vc < topk:
                    self.assertTrue(bool((got_row[vc:] == -1).all().item()))
                    self.assertTrue(bool((ref_row[vc:] == -1).all().item()))
                got_sorted = got_row[:vc].sort().values
                ref_sorted = ref_row[:vc].sort().values
                if not torch.equal(got_sorted, ref_sorted):
                    got_scores = (
                        full_logits[row].index_select(0, ks_r + got_sorted.to(torch.long))
                        .sort()
                        .values
                    )
                    ref_scores = (
                        full_logits[row].index_select(0, ks_r + ref_sorted.to(torch.long))
                        .sort()
                        .values
                    )
                    self.assertTrue(
                        torch.equal(got_scores, ref_scores),
                        f"row {row} topk={topk} score mismatch: vc={vc}",
                    )

    def test_score_chunks_and_fp8_mqa_score_unaligned_slices(self) -> None:
        if rtp_llm_ops is None or not has_fp8_mqa_logits():
            self.skipTest("rtp_llm_ops and fp8_mqa_logits required")

        torch.manual_seed(29)
        ratio = 4
        topk = 256
        # Non-4-aligned Q rows (3, 7) and non-4-aligned K spans (5, 9 -> k_lo=5).
        T_per_req = [5, 9]
        cu_q_host = [0, 3, 10]
        cu_k_host = [0, 5, 14]
        M = cu_q_host[-1]
        T_total = cu_k_host[-1]

        ks = torch.tensor(
            [0, 0, 0, 5, 5, 5, 5, 5, 5, 5], dtype=torch.int32, device=self.device
        )
        ke = torch.tensor(
            [1, 3, 5, 6, 7, 8, 10, 11, 13, 14], dtype=torch.int32, device=self.device
        )
        with patch(
            "rtp_llm.models_py.modules.dsv4.fp8.indexer._fp8_prefill_score_chunk_rows",
            return_value=3,
        ):
            chunks = _build_score_chunks(
                ks,
                ke,
                M=M,
                T=T_total,
                ratio=ratio,
                topk=topk,
                use_varlen=True,
                sp_int=0,
                cu_q_host=cu_q_host,
                cu_k_host=cu_k_host,
                T_per_req_host=T_per_req,
            )
        self.assertGreaterEqual(len(chunks), 2)

        q_fp8 = torch.randn(
            M, 64, INDEXER_HEAD_DIM, device=self.device, dtype=torch.bfloat16
        ).to(torch.float8_e4m3fn)
        w_fold = torch.randn(M, 64, device=self.device, dtype=torch.float32)
        k_quant = torch.randn(
            T_total, INDEXER_HEAD_DIM, device=self.device, dtype=torch.bfloat16
        ).to(torch.float8_e4m3fn)
        k_scale = torch.rand(T_total, dtype=torch.float32, device=self.device) + 0.25

        for c in chunks:
            self.assertEqual(c.k_lo % 4, 0)
            k_scale_c = k_scale[c.k_lo : c.k_hi]
            self.assertEqual(k_scale_c.data_ptr() % 16, 0)
            self.assertEqual(c.ks_chunk.data_ptr() % 16, 0)
            self.assertEqual(c.ke_chunk.data_ptr() % 16, 0)

        # Also exercise rs % 4 != 0 when k_lo == 0 across split query chunks.
        with patch(
            "rtp_llm.models_py.modules.dsv4.fp8.indexer._fp8_prefill_score_chunk_rows",
            return_value=3,
        ):
            split_chunks = _build_score_chunks(
                torch.zeros(7, dtype=torch.int32, device=self.device),
                torch.tensor([1, 2, 3, 4, 5, 6, 7], dtype=torch.int32, device=self.device),
                M=7,
                T=7,
                ratio=ratio,
                topk=topk,
                use_varlen=False,
                sp_int=0,
                cu_q_host=None,
                cu_k_host=None,
                T_per_req_host=None,
            )
        self.assertEqual(len(split_chunks), 3)
        for c in split_chunks:
            self.assertEqual(c.k_lo, 0)
            self.assertEqual(c.ks_chunk.data_ptr() % 16, 0)
            self.assertEqual(c.ke_chunk.data_ptr() % 16, 0)

        # Trigger high_padding_waste (merged > 4M and > 2*raw) without patches:
        # req 0 has many rows (299) and tiny span T_0=5, req 1 has 7 rows and T_1=16384,
        # so chunk 1 starts at rs=299 (3 mod 4) and k_lo_raw=5 -> aligned k_lo=4.
        huge_T = [5, 16384]
        q0_huge, q1_huge = 299, 7
        M_huge = q0_huge + q1_huge
        q_huge_fp8 = torch.randn(
            M_huge, 64, INDEXER_HEAD_DIM, device=self.device, dtype=torch.bfloat16
        ).to(torch.float8_e4m3fn)
        w_huge_fold = torch.randn(M_huge, 64, device=self.device, dtype=torch.float32)
        k_quant_big = torch.randn(
            sum(huge_T), INDEXER_HEAD_DIM, device=self.device, dtype=torch.bfloat16
        ).to(torch.float8_e4m3fn)
        k_scale_big = torch.rand(sum(huge_T), dtype=torch.float32, device=self.device) + 0.25
        ks_big = torch.cat(
            [
                torch.zeros(q0_huge, dtype=torch.int32, device=self.device),
                torch.full((q1_huge,), 5, dtype=torch.int32, device=self.device),
            ]
        )
        ke_big = torch.cat(
            [
                torch.full((q0_huge,), 5, dtype=torch.int32, device=self.device),
                torch.tensor(
                    [6, 100, 500, 1000, 4000, 8000, 16389],
                    dtype=torch.int32,
                    device=self.device,
                ),
            ]
        )
        big_chunks = _build_score_chunks(
            ks_big,
            ke_big,
            M=M_huge,
            T=sum(huge_T),
            ratio=ratio,
            topk=topk,
            use_varlen=True,
            sp_int=0,
            cu_q_host=[0, q0_huge, M_huge],
            cu_k_host=[0, 5, sum(huge_T)],
            T_per_req_host=huge_T,
        )
        self.assertEqual(len(big_chunks), 2)
        self.assertEqual(big_chunks[1].row_start, q0_huge)
        self.assertEqual(big_chunks[1].k_lo, 4)
        self.assertEqual(k_scale_big[big_chunks[1].k_lo : big_chunks[1].k_hi].data_ptr() % 16, 0)

        stub = IndexerFP8.__new__(IndexerFP8)
        stub.compress_ratio = ratio
        stub.n_heads = 64
        stub.head_dim = INDEXER_HEAD_DIM
        stub.index_topk = topk
        meta = _IndexerFP8PrefillMeta(
            bsz=2,
            seqlen=M_huge,
            M=M_huge,
            sp_int=0,
            end_pos=0,
            is_fresh_prefill=False,
            T=sum(huge_T),
            freqs_cis_slice=torch.empty(0),
            positions_d=torch.empty(0),
            ks=ks_big,
            ke=ke_big,
            block_table_i32=None,
            cu_kv_seqlens=None,
            cu_kv_per_token=None,
            compressor_meta=None,
            indexer_cp_plan=None,
            indexer_cp_local_cu=None,
            score_chunks=big_chunks,
        )
        out_chunked = stub._score_and_topk_prefill(
            q_huge_fp8, w_huge_fold, k_quant_big, k_scale_big, meta, (M_huge, topk)
        )
        self.assertEqual(out_chunked.shape, (M_huge, topk))

        # Direct fp8_mqa_indexer_score call with explicitly unaligned 1D slices (offset=1,3).
        logits_unaligned = fp8_mqa_indexer_score(
            q_fp8[1:6],
            w_fold[1:6],
            k_quant[1:12],
            k_scale[1:12],
            ks[3:8],
            ke[3:8],
            clean_logits=False,
        )
        logits_ref = fp8_mqa_indexer_score(
            q_fp8[1:6].clone(),
            w_fold[1:6].clone(),
            k_quant[1:12].clone(),
            k_scale[1:12].clone(),
            ks[3:8].clone(),
            ke[3:8].clone(),
            clean_logits=False,
        )
        self.assertEqual(logits_unaligned.shape, logits_ref.shape)
        for idx, (s, e) in enumerate(zip(ks[3:8].tolist(), ke[3:8].tolist())):
            self.assertTrue(
                torch.equal(logits_unaligned[idx, s:e], logits_ref[idx, s:e]),
                f"row {idx} logits mismatch on unaligned direct call",
            )


class CompressorMixedCacheReuseTest(unittest.TestCase):
    """HCA may_use_cache=False + CSA/Idx mixed continuation+cold vs per-req refs."""

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        self.device = torch.device("cuda")

    def test_hca_mixed_reuse_may_use_cache_false_matches_true_ref(self) -> None:
        head_dim = KV_HEAD_DIM
        rope_head_dim = 64
        compress_ratio = 128
        dim = 64
        # 256-aligned high reuse prefixes + cold requests + tiny 1280 prefix.
        prefixes = [4096, 2048, 0, 1280]
        inputs = [512, 256, 1024, 512]
        bsz = len(prefixes)
        max_seq = max(sp + ln for sp, ln in zip(prefixes, inputs))

        torch.manual_seed(99)
        x_per_req = [
            torch.randn(ln, dim, dtype=torch.bfloat16, device=self.device)
            for ln in inputs
        ]
        x_flat = torch.cat(x_per_req, dim=0)

        cu_seqlens, _, prefix_lengths, positions, b_idx = _build_varlen_inputs(
            prefixes, inputs, self.device
        )
        sp_per_req = prefix_lengths.to(torch.long)
        cu_seq_per_req = cu_seqlens.to(torch.long)

        # Build identical weight compressors for batched may_use_cache=False vs ref.
        cmp_opt = _build_test_compressor(
            dim=dim,
            head_dim=head_dim,
            rope_head_dim=rope_head_dim,
            compress_ratio=compress_ratio,
            max_batch_size=bsz,
            max_seq_len=max_seq + 16,
            device=self.device,
            seed=7,
        )
        state_opt, kv_opt = _bind_compressor_pools(
            cmp_opt,
            bsz=bsz,
            head_dim=head_dim,
            coff=1,
            compress_ratio=compress_ratio,
            max_seq_total=max_seq,
            device=self.device,
        )
        # Poison state cache so any unintended cache read would corrupt outputs.
        state_opt.fill_(float("nan"))

        meta_opt = cmp_opt.prepare_metadata(
            positions,
            b_idx.to(torch.long),
            has_prefix=True,
            is_batched=True,
            seq_start_per_req=sp_per_req,
            cu_seq_per_req=cu_seq_per_req,
            prefix_lengths_host=tuple(prefixes),
        )
        self.assertTrue(meta_opt.can_skip_state_cache_read)
        self.assertIsNotNone(meta_opt.cached_boundary_indices)
        assert meta_opt.cached_boundary_indices is not None
        self.assertEqual(int(meta_opt.cached_boundary_indices.numel()), 0)
        ws_hca = PrefillWorkspace(
            self.device,
            q_rows=max(sum(inputs), 1),
            q_dim=512,
            reserve_cp=False,
            align_bytes=1,
        )
        cmp_opt.forward(x_flat, prefixes[0], meta=meta_opt, workspace=ws_hca)
        with torch.no_grad():
            fused = _linear_bf16_bf16_fp32(x_flat, cmp_opt._wkv_wgate_fused)
            kv_flat = fused[:, :head_dim].contiguous()
            score_flat = fused[:, head_dim:].contiguous()
        kv_opt_direct = kv_opt.clone()
        kv_opt.zero_()
        run_fused_compress_kv_write(
            state_opt,
            meta_opt.token_to_req,
            meta_opt.positions,
            meta_opt.state_slots,
            cmp_opt._state_block_table,
            cmp_opt.norm.weight,
            cmp_opt.norm_eps,
            cmp_opt._cos_sin_cache,
            kv_opt,
            meta_opt.kv_slots,
            kv_flat,
            score_flat,
            cmp_opt.ape,
            0,
            disable_raw_path=False,
            head_dim=head_dim,
            rope_head_dim=rope_head_dim,
            compress_ratio=compress_ratio,
            overlap=False,
            seq_start_per_req=sp_per_req,
            cu_seq_per_req=cu_seq_per_req,
            state_tokens_per_block=cmp_opt._state_tokens_per_block,
            may_use_cache=False,
            active_token_indices=meta_opt.raw_boundary_indices,
        )
        self.assertTrue(torch.equal(kv_opt_direct, kv_opt))

        # Reference: MAY_USE_CACHE=True with valid state pool.
        cmp_ref = _build_test_compressor(
            dim=dim,
            head_dim=head_dim,
            rope_head_dim=rope_head_dim,
            compress_ratio=compress_ratio,
            max_batch_size=bsz,
            max_seq_len=max_seq + 16,
            device=self.device,
            seed=7,
        )
        state_ref, kv_ref = _bind_compressor_pools(
            cmp_ref,
            bsz=bsz,
            head_dim=head_dim,
            coff=1,
            compress_ratio=compress_ratio,
            max_seq_total=max_seq,
            device=self.device,
        )
        meta_ref = cmp_ref.prepare_metadata(
            positions,
            b_idx.to(torch.long),
            has_prefix=True,
            is_batched=True,
            seq_start_per_req=sp_per_req,
            cu_seq_per_req=cu_seq_per_req,
        )
        run_fused_compress_kv_write(
            state_ref,
            meta_ref.token_to_req,
            meta_ref.positions,
            meta_ref.state_slots,
            cmp_ref._state_block_table,
            cmp_ref.norm.weight,
            cmp_ref.norm_eps,
            cmp_ref._cos_sin_cache,
            kv_ref,
            meta_ref.kv_slots,
            kv_flat,
            score_flat,
            cmp_ref.ape,
            0,
            disable_raw_path=False,
            head_dim=head_dim,
            rope_head_dim=rope_head_dim,
            compress_ratio=compress_ratio,
            overlap=False,
            seq_start_per_req=sp_per_req,
            cu_seq_per_req=cu_seq_per_req,
            state_tokens_per_block=cmp_ref._state_tokens_per_block,
            may_use_cache=True,
        )

        self.assertTrue(
            torch.equal(kv_opt, kv_ref),
            "HCA may_use_cache=False with aligned prefixes must be byte-equal to may_use_cache=True",
        )

    def test_csa_and_idx_mixed_reuse_plus_non_aligned_boundary_split(self) -> None:
        # Test both CSA/Idx ratio=4 (overlap=True) with:
        # - Case 1: mixed high reuse [4096, 2048] + cold [0, 0]
        # - Case 2: non-aligned & tiny prefixes [1280, 130, 0, 258]
        for head_dim, prefixes, inputs in (
            (KV_HEAD_DIM, [4096, 2048, 0, 0], [512, 256, 512, 1024]),
            (INDEXER_HEAD_DIM, [1280, 130, 0, 258], [256, 128, 256, 130]),
        ):
            compress_ratio = 4
            rope_head_dim = 64
            dim = 64
            coff = 2
            bsz = len(prefixes)
            max_seq = max(sp + ln for sp, ln in zip(prefixes, inputs))

            torch.manual_seed(31 + head_dim)
            x_prefixes = [
                torch.randn(sp, dim, dtype=torch.bfloat16, device=self.device)
                for sp in prefixes
            ]
            x_cont = [
                torch.randn(ln, dim, dtype=torch.bfloat16, device=self.device)
                for ln in inputs
            ]

            cmp_batched = _build_test_compressor(
                dim=dim,
                head_dim=head_dim,
                rope_head_dim=rope_head_dim,
                compress_ratio=compress_ratio,
                max_batch_size=bsz,
                max_seq_len=max_seq + 16,
                device=self.device,
                seed=19,
            )
            state_batched, kv_batched = _bind_compressor_pools(
                cmp_batched,
                bsz=bsz,
                head_dim=head_dim,
                coff=coff,
                compress_ratio=compress_ratio,
                max_seq_total=max_seq,
                device=self.device,
            )
            ws = PrefillWorkspace(
                self.device, q_rows=max(sum(inputs), 1), q_dim=512, reserve_cp=False, align_bytes=1
            )

            # Seed prefix cache per continuation request on both batched & reference.
            ref_kv_slices = []
            ref_state_slices = []
            kv_eb = cmp_batched._kv_eb
            state_eb = cmp_batched._state_eb
            kv_blocks_per_req = int(cmp_batched._kv_block_table.shape[1])
            state_blocks_per_req = int(cmp_batched._state_block_table.shape[1])

            for req_id, (sp, ln) in enumerate(zip(prefixes, inputs)):
                cmp_r = _build_test_compressor(
                    dim=dim,
                    head_dim=head_dim,
                    rope_head_dim=rope_head_dim,
                    compress_ratio=compress_ratio,
                    max_batch_size=1,
                    max_seq_len=max_seq + 16,
                    device=self.device,
                    seed=19,
                )
                state_r, kv_r = _bind_compressor_pools(
                    cmp_r,
                    bsz=1,
                    head_dim=head_dim,
                    coff=coff,
                    compress_ratio=compress_ratio,
                    max_seq_total=sp + ln,
                    device=self.device,
                )
                if sp > 0:
                    pos_p = torch.arange(0, sp, device=self.device, dtype=torch.long)
                    b_p = torch.zeros(sp, device=self.device, dtype=torch.long)
                    meta_p = cmp_r.prepare_metadata(
                        pos_p,
                        b_p,
                        has_prefix=False,
                        seq_start_per_req=torch.tensor([0], dtype=torch.long, device=self.device),
                        cu_seq_per_req=torch.tensor([0, sp], dtype=torch.long, device=self.device),
                    )
                    cmp_r.forward(x_prefixes[req_id], 0, meta=meta_p, workspace=ws)

                    # Copy the same prefix state + KV into batched request slot req_id.
                    pos_pb = torch.full((sp,), req_id, device=self.device, dtype=torch.long)
                    # Only req_id active via direct forward using req_id's single-req slice:
                    single_cu = [0] * (bsz + 1)
                    for j in range(req_id + 1, bsz + 1):
                        single_cu[j] = sp
                    meta_pb = cmp_batched.prepare_metadata(
                        pos_p,
                        pos_pb,
                        has_prefix=False,
                        is_batched=True,
                        seq_start_per_req=torch.zeros(bsz, dtype=torch.long, device=self.device),
                        cu_seq_per_req=torch.tensor(single_cu, dtype=torch.long, device=self.device),
                    )
                    cmp_batched.forward(x_prefixes[req_id], 0, meta=meta_pb, workspace=ws)

                pos_c = torch.arange(sp, sp + ln, device=self.device, dtype=torch.long)
                b_c = torch.zeros(ln, device=self.device, dtype=torch.long)
                meta_r = cmp_r.prepare_metadata(
                    pos_c,
                    b_c,
                    has_prefix=sp > 0,
                    seq_start_per_req=torch.tensor([sp], dtype=torch.long, device=self.device),
                    cu_seq_per_req=torch.tensor([0, ln], dtype=torch.long, device=self.device),
                )
                cmp_r.forward(x_cont[req_id], sp, meta=meta_r, workspace=ws)
                r_kv_blocks = int(cmp_r._kv_block_table.shape[1])
                r_state_blocks = int(cmp_r._state_block_table.shape[1])
                ref_kv_slices.append(
                    (
                        kv_r[1 : 1 + r_kv_blocks].clone(),
                        state_r.view(-1, state_eb, state_r.shape[-1])[
                            1 : 1 + r_state_blocks
                        ].clone(),
                    )
                )

            # Run batched continuation + cold forward on cmp_batched.
            cu_seqlens, _, prefix_lengths, positions, b_idx = _build_varlen_inputs(
                prefixes, inputs, self.device
            )
            meta_batched = cmp_batched.prepare_metadata(
                positions,
                b_idx.to(torch.long),
                has_prefix=any(sp > 0 for sp in prefixes),
                is_batched=True,
                seq_start_per_req=prefix_lengths.to(torch.long),
                cu_seq_per_req=cu_seqlens.to(torch.long),
                prefix_lengths_host=tuple(prefixes),
            )
            self.assertFalse(meta_batched.can_skip_state_cache_read)
            self.assertIsNotNone(meta_batched.cached_boundary_indices)
            self.assertIsNotNone(meta_batched.raw_boundary_indices)
            assert meta_batched.cached_boundary_indices is not None
            assert meta_batched.raw_boundary_indices is not None
            window_size = (1 + 1) * compress_ratio
            expected_cached = sum(
                1
                for sp, ln in zip(prefixes, inputs)
                if sp > 0
                for pos in range(sp, sp + min(ln, window_size - 1))
                if (pos + 1) % compress_ratio == 0
            )
            self.assertEqual(
                int(meta_batched.cached_boundary_indices.numel()),
                expected_cached,
            )
            if all((sp % compress_ratio) == 0 for sp in prefixes):
                self.assertLessEqual(expected_cached, bsz)
            self.assertGreater(int(meta_batched.raw_boundary_indices.numel()), 0)
            cmp_batched.forward(torch.cat(x_cont, dim=0), prefixes[0], meta=meta_batched, workspace=ws)

            for req_id, (ref_kv_b, ref_st_b) in enumerate(ref_kv_slices):
                k_start = 1 + req_id * kv_blocks_per_req
                k_count = int(ref_kv_b.shape[0])
                self.assertTrue(
                    torch.equal(kv_batched[k_start : k_start + k_count], ref_kv_b),
                    f"req {req_id} KV pool mismatch for head_dim={head_dim}, prefixes={prefixes}",
                )
                s_start = 1 + req_id * state_blocks_per_req
                s_count = int(ref_st_b.shape[0])
                batched_st_view = state_batched.view(
                    -1, state_eb, state_batched.shape[-1]
                )
                self.assertTrue(
                    torch.equal(batched_st_view[s_start : s_start + s_count], ref_st_b),
                    f"req {req_id} state pool mismatch for head_dim={head_dim}, prefixes={prefixes}",
                )

            # Also verify _save_partial_states_kernel with compact active_indices vs full grid.
            save_state_batched = torch.zeros_like(state_batched)
            save_state_seq = torch.zeros_like(state_batched)
            T_tot = int(positions.numel())
            kv_raw_s = torch.randn(T_tot, coff * head_dim, dtype=torch.float32, device=self.device)
            sc_raw_s = torch.randn(T_tot, coff * head_dim, dtype=torch.float32, device=self.device)
            run_save_partial_states(
                kv_raw_s,
                sc_raw_s,
                cmp_batched.ape,
                positions,
                save_state_batched.view(-1, state_eb, save_state_batched.shape[-1]),
                meta_batched.state_slots,
                compress_ratio,
                active_indices=meta_batched.active_state_indices,
            )
            for req_id in range(bsz):
                s_lo = int(cu_seqlens[req_id].item())
                s_hi = int(cu_seqlens[req_id + 1].item())
                run_save_partial_states(
                    kv_raw_s[s_lo:s_hi],
                    sc_raw_s[s_lo:s_hi],
                    cmp_batched.ape,
                    positions[s_lo:s_hi],
                    save_state_seq.view(-1, state_eb, save_state_seq.shape[-1]),
                    meta_batched.state_slots[s_lo:s_hi],
                    compress_ratio,
                )
            self.assertTrue(torch.equal(save_state_batched, save_state_seq))

    def test_cp2_sharded_asymmetric_cached_boundary_collective_symmetry(self) -> None:
        """All CP ranks must participate in state read collective when any rank has a cached boundary."""
        dim = 64
        head_dim = 128
        rope_head_dim = 64
        prefix_len = 512
        q_len = 42
        cp_size = 2
        state_tpb = 256
        state_eb = state_tpb // cp_size
        kv_owner_tpb = 256

        cu_seqlens, _, prefix_lengths, positions, b_idx = _build_varlen_inputs(
            [prefix_len], [q_len], self.device
        )
        kv_raw = torch.randn(q_len, 2 * head_dim, dtype=torch.float32, device=self.device)
        sc_raw = torch.randn(q_len, 2 * head_dim, dtype=torch.float32, device=self.device)

        # CSA/Indexer (ratio=4, overlap=True): pos=515 needs state cache read, and under
        # owner_block = 515 // 256 = 2 (% 2 == 0), ONLY cp_rank=0 owns the KV slot.
        # Both cp_rank=0 and cp_rank=1 MUST still enter _build_cp_full_state_read_cache.
        for cp_rank, expected_local_cached in ((0, 1), (1, 0)):
            cmp = _build_test_compressor(
                dim=dim,
                head_dim=head_dim,
                rope_head_dim=rope_head_dim,
                compress_ratio=4,
                max_batch_size=1,
                max_seq_len=prefix_len + q_len + 16,
                device=self.device,
                seed=31,
            )
            cp_ctx = SimpleNamespace(
                cp_size=cp_size,
                cp_rank=cp_rank,
                kv_cache_sharded=True,
                prefix_lengths_host=(prefix_len,),
                prefix_lengths=prefix_lengths,
            )
            cmp.set_cp_ctx(cp_ctx)

            state_bt = torch.tensor([[1, 2, 3]], dtype=torch.int32, device=self.device)
            kv_eb = kv_owner_tpb // (cp_size * 4)
            kv_tpb = kv_eb * 4
            kv_bt = torch.tensor([[1, 2, 3, 4]], dtype=torch.int32, device=self.device)
            state_pool = torch.zeros(4 * state_eb, 4 * head_dim, dtype=torch.float32, device=self.device)
            kv_pool = torch.zeros(5, kv_eb, INDEXER_ENTRY_BYTES, dtype=torch.uint8, device=self.device)
            cmp.set_pool_context(
                kv_pool_view=kv_pool,
                kv_block_table=kv_bt,
                kv_eb=kv_eb,
                state_pool_view=state_pool,
                state_block_table=state_bt,
                state_eb=state_eb,
                state_tokens_per_block=state_tpb,
                kv_tokens_per_block=kv_tpb,
                kv_owner_tokens_per_block=kv_owner_tpb,
            )
            meta = cmp.prepare_metadata(
                positions,
                b_idx.to(torch.long),
                has_prefix=True,
                is_batched=True,
                seq_start_per_req=prefix_lengths.to(torch.long),
                cu_seq_per_req=cu_seqlens.to(torch.long),
                prefix_lengths_host=(prefix_len,),
            )
            self.assertIsNotNone(meta.cached_boundary_indices)
            assert meta.cached_boundary_indices is not None
            self.assertEqual(int(meta.cached_boundary_indices.numel()), expected_local_cached)
            self.assertTrue(meta.any_cached_boundary_global)
            self.assertTrue(_cp_sliced_state_read_needed(meta, cp_ctx, raw_disabled=False))

            full_state_pool = torch.zeros(4, state_tpb, 4 * head_dim, dtype=torch.float32, device=self.device)
            with patch(
                "rtp_llm.models_py.modules.dsv4.fp8.compressor._build_cp_full_state_read_cache",
                return_value=(full_state_pool, state_bt),
            ) as mock_gather:
                cmp._launch(
                    kv_raw,
                    sc_raw,
                    meta,
                    seq_start=None,
                )
                self.assertEqual(
                    mock_gather.call_count,
                    1,
                    f"cp_rank={cp_rank} diverged on CP state-read collective",
                )

        # Complementary HCA (ratio=128, overlap=False) with aligned prefix=512:
        # neither rank should enter the state-read collective.
        for cp_rank in (0, 1):
            cmp_hca = _build_test_compressor(
                dim=dim,
                head_dim=KV_HEAD_DIM,
                rope_head_dim=rope_head_dim,
                compress_ratio=128,
                max_batch_size=1,
                max_seq_len=prefix_len + q_len + 16,
                device=self.device,
                seed=37,
            )
            cp_ctx = SimpleNamespace(
                cp_size=cp_size,
                cp_rank=cp_rank,
                kv_cache_sharded=True,
                prefix_lengths_host=(prefix_len,),
                prefix_lengths=prefix_lengths,
            )
            cmp_hca.set_cp_ctx(cp_ctx)
            state_bt = torch.tensor([[1, 2, 3]], dtype=torch.int32, device=self.device)
            kv_bt = torch.tensor([[1, 2, 3, 4]], dtype=torch.int32, device=self.device)
            state_pool = torch.zeros(4 * state_eb, 2 * KV_HEAD_DIM, dtype=torch.float32, device=self.device)
            kv_pool = torch.zeros(5, 1, KV_ENTRY_BYTES, dtype=torch.uint8, device=self.device)
            cmp_hca.set_pool_context(
                kv_pool_view=kv_pool,
                kv_block_table=kv_bt,
                kv_eb=1,
                state_pool_view=state_pool,
                state_block_table=state_bt,
                state_eb=state_eb,
                state_tokens_per_block=state_tpb,
                kv_tokens_per_block=128,
                kv_owner_tokens_per_block=kv_owner_tpb,
            )
            meta_hca = cmp_hca.prepare_metadata(
                positions,
                b_idx.to(torch.long),
                has_prefix=True,
                is_batched=True,
                seq_start_per_req=prefix_lengths.to(torch.long),
                cu_seq_per_req=cu_seqlens.to(torch.long),
                prefix_lengths_host=(prefix_len,),
            )
            self.assertFalse(meta_hca.any_cached_boundary_global)
            self.assertFalse(_cp_sliced_state_read_needed(meta_hca, cp_ctx, raw_disabled=False))


class SwaMixedCacheReuseOpsTest(unittest.TestCase):
    """Tests for Parts 1-6 kernels + AttentionFP8/PrefillWorkspace contracts."""

    def test_prefill_workspace_disjoint_hca_gather_union_and_hbm_fallback(self) -> None:
        ws = PrefillWorkspace(
            torch.device("cpu"),
            q_rows=128,
            q_dim=16,  # q_bytes = 4096 B
            reserve_cp=True,
            cp_rows=32,
            main_w=8,  # main_bytes = 32*8*4 = 1024 B
            idx_w=8,  # idx_bytes = 1024 B; cp_region_bytes = 4096 B
            align_bytes=1,
        )
        # q_tokens=16 -> q_end=512 B; union len = 4096 B -> plenty of tail room.
        buf = ws.try_disjoint_hca_gather(16, 32, 8, torch.float32)
        self.assertIsNotNone(buf)
        assert buf is not None
        q_view = ws.prefill_q(16)
        q_end = q_view.numel() * q_view.element_size()
        self.assertGreaterEqual(buf.data_ptr() - ws._union.data_ptr(), q_end)

        with ws.use_disjoint_main_gather(buf, live_q_tokens=16):
            gathered = ws.cp_gather_main(32, 8, torch.float32)
            self.assertEqual(gathered.data_ptr(), buf.data_ptr())
            # Fill prefill_q and disjoint gather; verify cp_restore_main doesn't stomp either.
            q_view.fill_(3.0)
            gathered.fill_(7.0)
            restored = ws.cp_restore_main(32, 8, torch.float32)
            restored.fill_(-1.0)
            self.assertTrue(bool((q_view == 3.0).all().item()))
            self.assertTrue(bool((gathered == 7.0).all().item()))

        # After context exits, cp_gather_main returns the standard offset-0 view.
        std_gather = ws.cp_gather_main(32, 8, torch.float32)
        self.assertEqual(std_gather.data_ptr(), ws._union.data_ptr())

        # Large q_tokens fills _union -> CPU cannot call CUDA mem_get_info -> None fallback.
        no_buf = ws.try_disjoint_hca_gather(128, 32, 8, torch.float32)
        self.assertIsNone(no_buf)

        if torch.cuda.is_available():
            dev = torch.device("cuda")
            ws_cu = PrefillWorkspace(
                dev,
                q_rows=128,
                q_dim=16,
                reserve_cp=True,
                cp_rows=32,
                main_w=8,
                idx_w=8,
                align_bytes=1,
            )
            nbytes = 32 * 8 * 4
            # Tight HBM -> returns None cleanly.
            with patch(
                "torch.cuda.mem_get_info", return_value=(nbytes + 1024, 16 << 30)
            ):
                self.assertIsNone(ws_cu.try_disjoint_hca_gather(128, 32, 8, torch.float32))
            # Comfortable HBM (> 2*nbytes + 4 GiB) -> caches lazy buffer once.
            with patch(
                "torch.cuda.mem_get_info", return_value=(2 * nbytes + (5 << 30), 16 << 30)
            ):
                b1 = ws_cu.try_disjoint_hca_gather(128, 32, 8, torch.float32)
                b2 = ws_cu.try_disjoint_hca_gather(128, 32, 8, torch.float32)
                self.assertIsNotNone(b1)
                self.assertIsNotNone(b2)
                assert b1 is not None and b2 is not None
                self.assertEqual(b1.data_ptr(), b2.data_ptr())

    def test_attention_parts1_2_3_orchestration_contracts(self) -> None:
        # Part 1: CSA overlap passes before_gather_k=None and keeps main_pending alive until finish.
        csa_layer = AttentionFP8.__new__(AttentionFP8)
        torch.nn.Module.__init__(csa_layer)
        csa_layer.compress_ratio = 4
        csa_layer.layer_id = 1
        seq: list[str] = []
        csa_layer.compressor = SimpleNamespace(
            start_prefill=MagicMock(side_effect=lambda *a, **k: seq.append("start_main") or "MAIN"),
            wait_prefill_gather=MagicMock(side_effect=lambda *a, **k: seq.append("wait_main")),
            finish_prefill=MagicMock(side_effect=lambda *a, **k: seq.append("finish_main")),
        )
        indexer_stub = IndexerFP8.__new__(IndexerFP8)
        torch.nn.Module.__init__(indexer_stub)
        indexer_stub.start_prefill_nested_compressor = MagicMock(
            side_effect=lambda *a, **k: seq.append("start_nested") or "NESTED"
        )
        indexer_stub.forward_with_pending_nested = MagicMock(
            side_effect=lambda *a, **k: (seq.append(("indexer", k.get("before_gather_k"))) or torch.zeros(2, 4, dtype=torch.int32))
        )
        csa_layer.indexer = indexer_stub
        csa_layer._get_cp_gather_stream = MagicMock(return_value="STREAM")
        csa_layer._ensure_prefill_kv_full = MagicMock(side_effect=lambda qkv, c: qkv)
        csa_layer._prefill_write_swa_fp8_paged = MagicMock(side_effect=lambda *a: seq.append("swa_write"))
        csa_layer._forward_prefill_compressed = MagicMock(return_value=torch.zeros(2, 8))

        csa_common = PrefillMeta(
            seqlen=2,
            seqlen_full=4,
            rd=0,
            device=torch.device("cpu"),
            cp_ctx=SimpleNamespace(cp_size=2, seq_len_full=4),
            cp_on=True,
            freqs_cis=torch.zeros(1),
            topk_idxs=torch.zeros(1, dtype=torch.int32),
            sp_int=0,
            any_cont=False,
            row_seqlens_full=torch.tensor([2], dtype=torch.long),
            csa_meta=SimpleNamespace(
                indexer_meta=SimpleNamespace(sp_int=0, compressor_meta=None),
                compressor_meta=None,
                workspace_meta=None,
            ),
        )
        qkv_stub = PrefillQKV(
            qr=torch.zeros(2, 4, dtype=torch.bfloat16),
            q=torch.zeros(2, 4, dtype=torch.bfloat16),
            kv_full=torch.zeros(2, 4, dtype=torch.bfloat16),
        )
        csa_layer._forward_prefill_csa_overlapped(torch.zeros(2, 4, dtype=torch.bfloat16), qkv_stub, csa_common)
        self.assertEqual(
            seq,
            ["start_nested", "start_main", "swa_write", ("indexer", None), "finish_main"],
        )
        csa_layer.compressor.wait_prefill_gather.assert_not_called()

        # Part 2: HCA overlap materializes Q before finish_prefill when disjoint buffer exists.
        ws = PrefillWorkspace(
            torch.device("cpu"),
            q_rows=16,
            q_dim=8,
            reserve_cp=True,
            cp_rows=4,
            main_w=8,
            idx_w=4,
            align_bytes=1,
        )
        hca_layer = AttentionFP8.__new__(AttentionFP8)
        torch.nn.Module.__init__(hca_layer)
        hca_layer.compress_ratio = 128
        hca_layer.head_dim = 4
        hca_layer.indexer = None
        hca_seq: list[str] = []
        seen_gather_ptr: list[int] = []
        hca_layer.compressor = SimpleNamespace(
            overlap=False,
            head_dim=4,
            start_prefill=MagicMock(
                side_effect=lambda *a, **kw: (
                    seen_gather_ptr.append(kw["workspace"].cp_gather_main(4, 8, torch.float32).data_ptr())
                    or hca_seq.append("start_hca")
                    or "HCA_PENDING"
                )
            ),
            finish_prefill=MagicMock(side_effect=lambda *a, **kw: hca_seq.append("finish_hca")),
        )
        hca_layer._get_cp_gather_stream = MagicMock(return_value="STREAM")
        hca_layer._ensure_prefill_kv_full = MagicMock(side_effect=lambda qkv, c: qkv)
        hca_layer._prefill_write_swa_fp8_paged = MagicMock(side_effect=lambda *a: hca_seq.append("swa_write"))
        hca_layer._materialize_prefill_q = MagicMock(side_effect=lambda qkv, c: (hca_seq.append("materialize_q") or qkv))
        hca_layer._forward_prefill_compressed = MagicMock(return_value=torch.zeros(2, 8))
        hca_common = PrefillMeta(
            seqlen=2,
            seqlen_full=4,
            rd=0,
            device=torch.device("cpu"),
            cp_ctx=SimpleNamespace(cp_size=2, seq_len_full=4, padded_seq_len=4),
            cp_on=True,
            freqs_cis=torch.zeros(1),
            topk_idxs=torch.zeros(1, dtype=torch.int32),
            sp_int=0,
            any_cont=False,
            row_seqlens_full=torch.tensor([2], dtype=torch.long),
            hca_meta=SimpleNamespace(compressor_meta=None, workspace_meta=None),
            workspace=ws,
        )
        hca_layer._forward_prefill_hca_overlapped(torch.zeros(2, 4, dtype=torch.bfloat16), qkv_stub, hca_common)
        self.assertEqual(hca_seq, ["start_hca", "swa_write", "materialize_q", "finish_hca"])
        self.assertGreater(seen_gather_ptr[0], ws._union.data_ptr())

        # Part 3: combine_topk_swa_indices_cp runs ONCE across multiple HCA layers sharing WorkspaceMeta.
        N = 4
        T_tok = 3
        dense_topk = torch.arange(N, dtype=torch.int32).view(1, N).expand(T_tok, N)
        wm = WorkspaceMeta(
            M=8,
            N=N,
            swa_eb=1,
            cmp_eb=1,
            swa_bt_int32=torch.zeros(1, 1, dtype=torch.int32),
            cmp_bt_int32=torch.zeros(1, 1, dtype=torch.int32),
            swa_seq_lens=torch.tensor([5], dtype=torch.int32),
            cmp_seq_lens=torch.tensor([0], dtype=torch.int32),
            swa_gather_lens=torch.tensor([3], dtype=torch.int32),
            swa_cache_seq_lens=torch.tensor([0], dtype=torch.int32),
            swa_cache_gather_lens=torch.tensor([0], dtype=torch.int32),
            qsl=torch.tensor([0, 3], dtype=torch.int32),
            dense_cmp_topk=dense_topk,
            new_k_slot_in_flat=torch.arange(N, N + T_tok, dtype=torch.int64),
            hca_combine_cache={},
        )
        common_hca_ws = PrefillMeta(
            seqlen=T_tok,
            seqlen_full=T_tok,
            rd=0,
            device=torch.device("cpu"),
            cp_ctx=SimpleNamespace(
                cp_size=2,
                cp_rank=0,
                prefix_length=0,
                global_positions=torch.arange(T_tok, dtype=torch.int64),
            ),
            cp_on=True,
            freqs_cis=torch.ones(T_tok, dtype=torch.float32),
            topk_idxs=torch.zeros(1, dtype=torch.int32),
            sp_int=0,
            any_cont=False,
            row_seqlens_full=torch.tensor([T_tok], dtype=torch.long),
            batch_size=1,
            prefix_lengths=torch.tensor([0], dtype=torch.int32),
            req_id_per_token=torch.zeros(T_tok, dtype=torch.int32),
        )
        mock_combine_cp = MagicMock(
            return_value=(
                torch.full((T_tok, 1, 128), -1, dtype=torch.int32),
                torch.tensor([3, 3, 3], dtype=torch.int32),
            )
        )
        with (
            patch.dict(
                "sys.modules",
                {
                    "rtp_llm.models_py.modules.dsv4.fp8._swa_ops_triton": SimpleNamespace(
                        combine_topk_swa_indices=MagicMock(),
                        combine_topk_swa_indices_cp=mock_combine_cp,
                    )
                },
            ),
            patch(
                "rtp_llm.models_py.modules.dsv4.fp8._pool_reader.dequantize_and_gather_k_cache",
                MagicMock(),
            ),
        ):
            for lid in range(3):
                l = AttentionFP8.__new__(AttentionFP8)
                torch.nn.Module.__init__(l)
                l.layer_id = lid
                l.compress_ratio = 128
                l.window_size = 512
                l.head_dim = 4
                l.dim = 8
                l._swa_cp_byte_sliced = MagicMock(return_value=False)
                l._pool_view_3d_fp8 = MagicMock(return_value=torch.zeros(1, 1, 584, dtype=torch.uint8))
                l._should_async_workspace_reads_for_prefill = MagicMock(return_value=False)
                l._flash_mla_sparse_fwd_chunked_projected = MagicMock(return_value=torch.zeros(T_tok, 8, dtype=torch.bfloat16))
                qkv_h = PrefillQKV(
                    qr=torch.zeros(T_tok, 4, dtype=torch.bfloat16),
                    q=torch.zeros(T_tok, 1, 4, dtype=torch.bfloat16),
                    kv_full=torch.zeros(T_tok, 4, dtype=torch.bfloat16),
                )
                l._attn_via_workspace(qkv_h, common_hca_ws, wm, cmp_topk_runtime=None)

        self.assertEqual(mock_combine_cp.call_count, 1)
        self.assertTrue(mock_combine_cp.call_args.kwargs.get("dense_arange_topk"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_combine_topk_cp_mixed_reuse_regular_and_dense_arange(self) -> None:
        device = torch.device("cuda")
        prefixes = [4096, 2048, 0, 1280]
        inputs = [512, 256, 512, 1024]
        win = 512

        # Zigzag-like local positions per request (1/2 of global sequence across cp_size=2).
        gp_list = []
        req_list = []
        for req_id, (sp, ln) in enumerate(zip(prefixes, inputs)):
            half = ln // 2
            gp_req = torch.cat(
                [
                    torch.arange(sp, sp + half, device=device, dtype=torch.int64),
                    torch.arange(sp + ln - half, sp + ln, device=device, dtype=torch.int64),
                ]
            )
            gp_list.append(gp_req)
            req_list.append(torch.full((int(gp_req.numel()),), req_id, device=device, dtype=torch.int32))
        gp = torch.cat(gp_list)
        req_id = torch.cat(req_list)
        prefix_t = torch.tensor(prefixes, device=device, dtype=torch.int64)
        T_tok = int(gp.numel())

        # CSA ratio=4 sparse non-contiguous topk_indices
        N_csa = max((sp + ln) // 4 for sp, ln in zip(prefixes, inputs))
        M_csa = N_csa + max(ln + min(sp, win - 1) for sp, ln in zip(prefixes, inputs))
        topk_csa = 2048
        topk_indices_csa = torch.randint(
            0, N_csa, (T_tok, topk_csa), dtype=torch.int32, device=device
        )
        got_i, got_l = combine_topk_swa_indices_cp(
            topk_indices=topk_indices_csa,
            global_positions=gp,
            sp_int=0,
            window_size=win,
            compress_ratio=4,
            topk=topk_csa,
            M=M_csa,
            N=N_csa,
            req_id_per_token=req_id,
            prefix_lengths=prefix_t,
            flash_mla_indices=True,
        )
        ref_i, ref_l = combine_topk_swa_indices_cp_varlen(
            topk_indices=topk_indices_csa,
            global_positions=gp,
            sp_int=0,
            window_size=win,
            compress_ratio=4,
            topk=topk_csa,
            M=M_csa,
            N=N_csa,
            req_id_per_token=req_id,
            prefix_lengths=prefix_t,
        )
        self.assertEqual(got_i.shape, (T_tok, 1, ref_i.shape[-1]))
        self.assertTrue(torch.equal(got_i.squeeze(1), ref_i))
        self.assertTrue(torch.equal(got_l, ref_l))

        # HCA ratio=128 stride-0 arange(N) fast path + T=1 edge case
        N_hca = max((sp + ln) // 128 for sp, ln in zip(prefixes, inputs))
        M_hca = N_hca + max(ln + min(sp, win - 1) for sp, ln in zip(prefixes, inputs))
        dense_view = torch.arange(N_hca, dtype=torch.int32, device=device).view(1, N_hca).expand(T_tok, N_hca)
        got_hca_i, got_hca_l = combine_topk_swa_indices_cp(
            topk_indices=dense_view,
            global_positions=gp,
            sp_int=0,
            window_size=win,
            compress_ratio=128,
            topk=N_hca,
            M=M_hca,
            N=N_hca,
            req_id_per_token=req_id,
            prefix_lengths=prefix_t,
            dense_arange_topk=True,
        )
        ref_hca_i, ref_hca_l = combine_topk_swa_indices_cp_varlen(
            topk_indices=dense_view,
            global_positions=gp,
            sp_int=0,
            window_size=win,
            compress_ratio=128,
            topk=N_hca,
            M=M_hca,
            N=N_hca,
            req_id_per_token=req_id,
            prefix_lengths=prefix_t,
        )
        self.assertTrue(torch.equal(got_hca_i, ref_hca_i))
        self.assertTrue(torch.equal(got_hca_l, ref_hca_l))

        # Single-token edge case: expand(1, 1) has stride(0) == 1 (not 0), exercising explicit flag.
        t1_view = torch.arange(1, dtype=torch.int32, device=device).view(1, 1).expand(1, 1)
        t1_gp = torch.tensor([127], dtype=torch.int64, device=device)
        t1_req = torch.zeros(1, dtype=torch.int32, device=device)
        t1_pref = torch.zeros(1, dtype=torch.int64, device=device)
        got_t1_i, got_t1_l = combine_topk_swa_indices_cp(
            topk_indices=t1_view,
            global_positions=t1_gp,
            sp_int=0,
            window_size=win,
            compress_ratio=128,
            topk=1,
            M=129,
            N=1,
            req_id_per_token=t1_req,
            prefix_lengths=t1_pref,
            dense_arange_topk=True,
        )
        ref_t1_i, ref_t1_l = combine_topk_swa_indices_cp_varlen(
            topk_indices=t1_view,
            global_positions=t1_gp,
            sp_int=0,
            window_size=win,
            compress_ratio=128,
            topk=1,
            M=129,
            N=1,
            req_id_per_token=t1_req,
            prefix_lengths=t1_pref,
        )
        self.assertTrue(torch.equal(got_t1_i, ref_t1_i))
        self.assertTrue(torch.equal(got_t1_l, ref_t1_l))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_row_scatter_vectorized_block_n_and_alignments(self) -> None:
        device = torch.device("cuda")
        torch.manual_seed(55)
        for n_rows in (1, 7, 8, 9, 17, 25):
            dst = torch.full((64, 512), -7.0, dtype=torch.bfloat16, device=device)
            src = torch.randn((n_rows, 512), dtype=torch.bfloat16, device=device)
            slots = torch.randperm(64, dtype=torch.int64, device=device)[:n_rows]
            if n_rows >= 9:
                # Full 8-row tile of -1 sentinel + partial negative slot.
                slots[:8] = -1
            dst_ref = dst.clone()
            valid = slots >= 0
            if valid.any():
                dst_ref.index_copy_(0, slots[valid], src[valid])
            row_scatter(dst, slots, src, mask_negative=True)
            self.assertTrue(torch.equal(dst, dst_ref), f"mismatch for n_rows={n_rows}")

        # Non-16B-aligned odd-halfword fallback & strided src slice.
        for cols in (17, 68, 514):
            dst_u8 = torch.zeros((20, cols), dtype=torch.uint8, device=device)
            full_u8 = torch.randint(0, 255, (11, cols * 2), dtype=torch.uint8, device=device)
            src_u8 = full_u8[:, cols:]
            slots = torch.tensor([19, 0, -1, 5, 11, -1, 3, 8, 9, -1, 14], dtype=torch.int64, device=device)
            ref_u8 = dst_u8.clone()
            v = slots >= 0
            ref_u8.index_copy_(0, slots[v], src_u8[v])
            row_scatter(dst_u8, slots, src_u8, mask_negative=True)
            self.assertTrue(torch.equal(dst_u8, ref_u8), f"mismatch for cols={cols}")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_flat_1d_swa_dequant_skewed_with_zero_len_cold_requests(self) -> None:
        device = torch.device("cuda")
        torch.manual_seed(77)
        win = 512
        block_size = 64
        # B=4 mixed: two high-reuse requests (4096, 2048 -> gather=511), two cold (0 -> gather=0)
        # plus tiny-prefix case [1280, 0, 5, 0] and all-zero [0, 0, 0, 0].
        for prefixes in ([4096, 2048, 0, 0], [1280, 0, 5, 0], [0, 0, 0, 0]):
            B = len(prefixes)
            gather_lens_list = [min(p, win - 1) for p in prefixes]
            max_gather = max(gather_lens_list)
            offset = 32
            M = offset + max_gather + 16

            # Allocate blocks cleanly per request with valid data in each active slot.
            blocks_per_req = max(1, (max(prefixes) + block_size - 1) // block_size)
            num_blocks = 1 + B * blocks_per_req
            k_cache = torch.zeros(
                (num_blocks, block_size, SWA_ENTRY_BYTES), dtype=torch.uint8, device=device
            )
            block_table = torch.tensor(
                [[1 + b * blocks_per_req + j for j in range(blocks_per_req)] for b in range(B)],
                dtype=torch.int32,
                device=device,
            )
            seq_lens = torch.tensor(prefixes, dtype=torch.int32, device=device)
            gather_lens = torch.tensor(gather_lens_list, dtype=torch.int32, device=device)
            slot_mapping = torch.full(
                (B, max(max_gather, 1)), -1, dtype=torch.int64, device=device
            )
            for b, (sp, gl) in enumerate(zip(prefixes, gather_lens_list)):
                if gl <= 0:
                    continue
                for i in range(gl):
                    pos = sp - gl + i
                    blk_id = int(block_table[b, pos // block_size].item())
                    in_blk = pos % block_size
                    slot_mapping[b, i] = blk_id * block_size + in_blk
                # Exercise in-window -1 sentinel preservation on flat 1D path too.
                if gl > 7:
                    slot_mapping[b, 3] = -1

            all_pos_slots = slot_mapping[slot_mapping >= 0]
            if all_pos_slots.numel() > 0:
                src_k = torch.randn(
                    int(all_pos_slots.numel()), SWA_HEAD_DIM, dtype=torch.bfloat16, device=device
                )
                quantize_and_insert_k_cache(src_k, k_cache, all_pos_slots)

            cu_tiles_host, total_tiles = build_dequant_cu_tiles_from_host(
                gather_lens_list, device=device
            )
            cu_tiles_dev = build_dequant_cu_tiles_device(gather_lens)
            self.assertTrue(torch.equal(cu_tiles_host, cu_tiles_dev))

            # 1) Compare dequantize_and_gather_k_cache_slots flat 1D vs 2D baseline.
            slots_2d_in = slot_mapping[:, :max_gather]
            out_2d = torch.full((B, M, SWA_HEAD_DIM), -3.0, dtype=torch.bfloat16, device=device)
            out_1d = out_2d.clone()
            dequantize_and_gather_k_cache_slots(
                out=out_2d,
                k_cache=k_cache,
                slot_mapping=slots_2d_in,
                gather_lens=gather_lens,
                offset=offset,
            )
            dequantize_and_gather_k_cache_slots(
                out=out_1d,
                k_cache=k_cache,
                slot_mapping=slots_2d_in,
                gather_lens=gather_lens,
                offset=offset,
                cu_tiles=cu_tiles_host,
                total_tiles=total_tiles,
            )
            self.assertTrue(torch.equal(out_1d, out_2d))
            # Untouched zero-len requests and margins keep exact sentinel -3.0.
            for b, gl in enumerate(gather_lens_list):
                self.assertTrue(bool((out_1d[b, :offset] == -3.0).all().item()))
                self.assertTrue(bool((out_1d[b, offset + gl :] == -3.0).all().item()))
                if gl > 7:
                    self.assertTrue(bool((out_1d[b, offset + 3] == 0.0).all().item()))

            # 2) Compare try_dequantize_and_gather_k_cache_slots_to_workspace if supported.
            if direct_triton_fast_path_supported(device) and max_gather > 0:
                out_ws = torch.full(
                    (B, M, SWA_HEAD_DIM), -3.0, dtype=torch.bfloat16, device=device
                )
                ok = try_dequantize_and_gather_k_cache_slots_to_workspace(
                    out=out_ws,
                    k_cache=k_cache,
                    slot_mapping=slots_2d_in,
                    gather_lens=gather_lens,
                    offset=offset,
                    cu_tiles=cu_tiles_dev,
                    total_tiles=total_tiles,
                )
                self.assertTrue(ok)
                self.assertTrue(torch.equal(out_ws, out_2d))

            # 3) Compare dequantize_and_gather_k_cache (compressed-K style seq_lens) flat 1D vs 2D.
            cmp_lens_list = [max(0, p // 4) for p in prefixes]
            cmp_lens = torch.tensor(cmp_lens_list, dtype=torch.int32, device=device)
            cmp_max = max(cmp_lens_list)
            cmp_cu, cmp_total = build_dequant_cu_tiles_from_host(
                cmp_lens_list, device=device
            )
            cmp_M = cmp_max + 8
            cmp_out_2d = torch.full(
                (B, cmp_M, SWA_HEAD_DIM), -5.0, dtype=torch.bfloat16, device=device
            )
            cmp_out_1d = cmp_out_2d.clone()
            # Reuse block_table where entries are written cleanly at [0, cmp_lens[b]).
            for b, cl in enumerate(cmp_lens_list):
                if cl > 0:
                    c_slots = torch.tensor(
                        [
                            int(block_table[b, pos // block_size].item()) * block_size
                            + (pos % block_size)
                            for pos in range(cl)
                        ],
                        dtype=torch.int64,
                        device=device,
                    )
                    quantize_and_insert_k_cache(
                        torch.randn(cl, SWA_HEAD_DIM, dtype=torch.bfloat16, device=device),
                        k_cache,
                        c_slots,
                    )
            dequantize_and_gather_k_cache(
                out=cmp_out_2d,
                k_cache=k_cache,
                seq_lens=cmp_lens,
                gather_lens=None,
                block_table=block_table,
                block_size=block_size,
                offset=0,
            )
            dequantize_and_gather_k_cache(
                out=cmp_out_1d,
                k_cache=k_cache,
                seq_lens=cmp_lens,
                gather_lens=None,
                block_table=block_table,
                block_size=block_size,
                offset=0,
                cu_tiles=cmp_cu,
                total_tiles=cmp_total,
            )
            self.assertTrue(torch.equal(cmp_out_1d, cmp_out_2d))


if __name__ == "__main__":
    unittest.main()
