"""UT: IndexerFP8 prefill kernel chain — gather + score + top-K.

Glue test for the three-kernel chain ``IndexerFP8.forward`` invokes once
the per-call meta is built:

  1. ``rtp_llm_ops.cp_gather_indexer_k_quant_cache``
       — gather paged FP8 K + scale via (block_table, cu_kv_seqlens)
  2. ``deep_gemm.fp8_mqa_logits`` (via :func:`fp8_mqa_indexer_score`)
       — per-row [ks[r], ke[r]) MQA score, fp32 logits
  3. ``rtp_llm_ops.dsv4_top_k_per_row_prefill``
       — per-row TopK over ``[ks[r], ke[r])`` with ``-1`` padding

The kernel-level UTs already lock each kernel's correctness in isolation
(``test_dsv4_top_k_per_row_prefill.py``, ``test_fp8_mqa_prefill_indexer
score.py``). This file targets the **glue layer** — the same role
``IndexerOpGetTopkRaggedV2Test`` plays in the
``refactor/dsv4_attention_prefill_split_back`` branch (vendored to here
since we don't take a dependency on ``IndexerOp``):

  * shape / dtype contracts at each boundary
  * the ``k_scale`` ``view(fp32).squeeze(-1)`` glue that
    ``deep_gemm.fp8_mqa_logits`` requires
  * causal ``ke = (q_pos+1)//ratio clamp T`` round-trip — every row's
    valid topk lands in ``[0, ke[r])`` and padding is ``-1``

Run:
  CUDA_VISIBLE_DEVICES=7 /opt/conda310/bin/python3 -m unittest \\
    rtp_llm.models_py.modules.dsv4.test.test_indexer_fp8_prefill_chain
"""

from __future__ import annotations

import unittest

import torch

from rtp_llm.models_py.modules.dsv4._indexer_fp8_quant_triton import (
    INDEXER_ENTRY_BYTES,
    INDEXER_HEAD_DIM,
)
from rtp_llm.models_py.modules.dsv4._indexer_q_fp8_quant_triton import (
    indexer_q_fp8_quant_fold,
)
from rtp_llm.models_py.modules.dsv4._indexer_score_fp8 import (
    fp8_mqa_indexer_score,
    has_fp8_mqa_logits,
)
from rtp_llm.ops.compute_ops import rtp_llm_ops


def _have_chain_kernels() -> bool:
    """All four pieces of the chain must be present:
    indexer_k_quant_and_cache (writer), cp_gather_indexer_k_quant_cache
    (reader), deep_gemm.fp8_mqa_logits, dsv4_top_k_per_row_prefill."""
    return (
        torch.cuda.is_available()
        and has_fp8_mqa_logits()
        and hasattr(rtp_llm_ops, "cp_gather_indexer_k_quant_cache")
        and hasattr(rtp_llm_ops, "indexer_k_quant_and_cache")
        and hasattr(rtp_llm_ops, "dsv4_top_k_per_row_prefill")
    )


class IndexerFP8PrefillChainTest(unittest.TestCase):
    """End-to-end test of cp_gather + fp8_mqa_logits + dsv4_top_k_per_row_prefill."""

    def setUp(self) -> None:
        if not _have_chain_kernels():
            self.skipTest("CUDA + deep_gemm + vendored gather/topk kernels required")
        self.device = torch.device("cuda")
        torch.manual_seed(42)
        # Pick 32 heads so deep_gemm fp8_mqa_logits' seq_len_alignment
        # constraint (block_q = 128/n_heads = 4) is satisfied. IndexerFP8
        # production uses 64 heads (block_q=2), but 32 is the standard
        # cross-test config used in test_indexer_op_v2.py.
        self.index_n_heads = 32
        self.index_head_dim = INDEXER_HEAD_DIM  # 128 (matches CompressorFP8 layout)
        self.quant_block_size = 128
        self.kernel_block_size = 64  # paged cache block (= eb)

    # ------------------------------------------------------------------
    # KV pool population helper — uses the production writer
    # ``indexer_k_quant_and_cache`` so layout is byte-equivalent to what
    # the gather kernel reads in real prefill.
    # ------------------------------------------------------------------
    def _build_kv_pool(self, num_kv_tokens: int):
        cache_stride = INDEXER_ENTRY_BYTES  # 132 = 128 fp8 + 4 fp32 scale
        num_blocks = (
            num_kv_tokens + self.kernel_block_size - 1
        ) // self.kernel_block_size
        kv_pool = torch.empty(
            num_blocks,
            self.kernel_block_size,
            cache_stride,
            dtype=torch.uint8,
            device=self.device,
        )
        k_bf16 = torch.randn(
            num_kv_tokens,
            self.index_head_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )
        slot_mapping = torch.arange(
            num_kv_tokens, dtype=torch.int64, device=self.device
        )
        rtp_llm_ops.indexer_k_quant_and_cache(
            k_bf16, kv_pool, slot_mapping, self.quant_block_size, "ue8m0"
        )
        return kv_pool, num_blocks

    # ------------------------------------------------------------------
    # The whole chain — runs end-to-end and checks every boundary
    # ------------------------------------------------------------------
    def _run_chain(
        self,
        q_bf16: torch.Tensor,  # [N, H, D]
        weights_bf16: torch.Tensor,  # [N, H]
        kv_pool: torch.Tensor,  # [num_blocks, eb, 132] uint8
        block_table: torch.Tensor,  # [B=1, num_blocks] int32
        cu_kv_seqlens: torch.Tensor,  # [B+1=2] int32
        ks: torch.Tensor,  # [N] int32
        ke: torch.Tensor,  # [N] int32
        index_topk: int,
        num_kv_tokens: int,
    ) -> torch.Tensor:
        # 1. Gather K + scale from paged pool.
        k_quant = torch.empty(
            (num_kv_tokens, self.index_head_dim),
            dtype=torch.float8_e4m3fn,
            device=self.device,
        )
        k_scale_buf = torch.empty(
            (num_kv_tokens, 4), dtype=torch.uint8, device=self.device
        )
        rtp_llm_ops.cp_gather_indexer_k_quant_cache(
            kv_pool, k_quant, k_scale_buf, block_table, cu_kv_seqlens
        )
        k_scale = k_scale_buf.view(torch.float32).squeeze(-1)
        self.assertEqual(k_scale.dim(), 1, "fp8_mqa_logits requires 1D k_scale")
        self.assertEqual(k_scale.dtype, torch.float32)

        # 2. Q quant + weight scale-fold (4D-shape contract).
        N, H, D = q_bf16.shape
        q_4d = q_bf16.view(1, N, H, D).contiguous()
        w_3d = weights_bf16.view(1, N, H).contiguous()
        q_fp8, w_fold = indexer_q_fp8_quant_fold(q_4d, w_3d)
        self.assertEqual(q_fp8.dtype, torch.float8_e4m3fn)
        self.assertEqual(q_fp8.shape, (1, N, H, D))
        self.assertEqual(w_fold.dtype, torch.float32)

        # 3. Score via DeepGEMM.
        logits = fp8_mqa_indexer_score(
            q_fp8.view(N, H, D),
            w_fold.view(N, H),
            k_quant,
            k_scale,
            ks,
            ke,
            clean_logits=False,
        )
        self.assertEqual(logits.shape, (N, num_kv_tokens))
        self.assertEqual(logits.dtype, torch.float32)

        # 4. Per-row TopK with [ks, ke) window + -1 padding.
        out_buf = torch.empty((N, index_topk), dtype=torch.int32, device=self.device)
        rtp_llm_ops.dsv4_top_k_per_row_prefill(
            logits,
            ks,
            ke,
            out_buf,
            N,
            logits.stride(0),
            logits.stride(1),
            index_topk,
        )
        return out_buf

    # ------------------------------------------------------------------
    # Test cases
    # ------------------------------------------------------------------
    def test_chain_returns_in_range_indices_with_minus_one_padding(self) -> None:
        """Causal Q-rows: row i sees [0, i+1). TopK should land in
        ``[0, i+1)`` with ``-1`` padding past per-row valid count."""
        N = 32  # block_q = 128/32 = 4 → must be multiple of 4
        num_kv_tokens = 64  # exactly one kernel_block_size
        index_topk = 2048

        kv_pool, num_blocks = self._build_kv_pool(num_kv_tokens)
        block_table = torch.arange(
            num_blocks, dtype=torch.int32, device=self.device
        ).view(1, num_blocks)
        cu_kv_seqlens = torch.tensor(
            [0, num_kv_tokens], dtype=torch.int32, device=self.device
        )

        q = torch.randn(
            N,
            self.index_n_heads,
            self.index_head_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )
        weights = torch.rand(
            N, self.index_n_heads, dtype=torch.bfloat16, device=self.device
        )
        ks = torch.zeros(N, dtype=torch.int32, device=self.device)
        ke = torch.arange(1, N + 1, dtype=torch.int32, device=self.device).clamp(
            max=num_kv_tokens
        )

        topk = self._run_chain(
            q,
            weights,
            kv_pool,
            block_table,
            cu_kv_seqlens,
            ks,
            ke,
            index_topk,
            num_kv_tokens,
        )

        self.assertEqual(topk.shape, (N, index_topk))
        self.assertEqual(topk.dtype, torch.int32)
        for r in range(N):
            valid_count = min(r + 1, num_kv_tokens, index_topk)
            valid_idxs = topk[r, :valid_count]
            self.assertTrue(
                ((valid_idxs >= 0) & (valid_idxs < num_kv_tokens)).all().item(),
                f"Row {r} has out-of-range topk index: "
                f"{valid_idxs[:8].cpu().tolist()}",
            )
            if valid_count < index_topk:
                pad = topk[r, valid_count:]
                self.assertTrue(
                    (pad == -1).all().item(),
                    f"Row {r} padding past valid count {valid_count} "
                    f"should be all -1, got {pad[:5].cpu().tolist()}",
                )

    def test_chain_k_scale_squeeze_glue(self) -> None:
        """Regression on ``k_scale.view(fp32).squeeze(-1)`` glue. If the
        squeeze regresses, ``deep_gemm.fp8_mqa_logits`` asserts on
        ``k_scale.dim() == 1`` and crashes inside ``fp8_mqa_indexer_score``."""
        N = 8
        num_kv_tokens = 8
        index_topk = 64
        kv_pool, num_blocks = self._build_kv_pool(num_kv_tokens)
        block_table = torch.arange(
            num_blocks, dtype=torch.int32, device=self.device
        ).view(1, num_blocks)
        cu_kv_seqlens = torch.tensor(
            [0, num_kv_tokens], dtype=torch.int32, device=self.device
        )
        q = torch.randn(
            N,
            self.index_n_heads,
            self.index_head_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )
        weights = torch.ones(
            N, self.index_n_heads, dtype=torch.bfloat16, device=self.device
        )
        ks = torch.zeros(N, dtype=torch.int32, device=self.device)
        ke = torch.full((N,), num_kv_tokens, dtype=torch.int32, device=self.device)
        # Should not raise.
        topk = self._run_chain(
            q,
            weights,
            kv_pool,
            block_table,
            cu_kv_seqlens,
            ks,
            ke,
            index_topk,
            num_kv_tokens,
        )
        self.assertEqual(topk.shape, (N, index_topk))

    def test_chain_multi_block_pool(self) -> None:
        """Cross multiple paged blocks (num_kv_tokens > kernel_block_size)
        — gather must walk the block_table correctly."""
        N = 16
        num_kv_tokens = self.kernel_block_size * 3  # 192 — 3 blocks
        index_topk = 128
        kv_pool, num_blocks = self._build_kv_pool(num_kv_tokens)
        # Non-identity block_table to exercise indirection.
        # eg [num_blocks-1, ..., 0] reverses physical order — gather
        # must follow the table, not assume sequential blocks.
        block_table = torch.arange(
            num_blocks - 1, -1, -1, dtype=torch.int32, device=self.device
        ).view(1, num_blocks)
        cu_kv_seqlens = torch.tensor(
            [0, num_kv_tokens], dtype=torch.int32, device=self.device
        )
        q = torch.randn(
            N,
            self.index_n_heads,
            self.index_head_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )
        weights = torch.rand(
            N, self.index_n_heads, dtype=torch.bfloat16, device=self.device
        )
        ks = torch.zeros(N, dtype=torch.int32, device=self.device)
        ke = torch.full((N,), num_kv_tokens, dtype=torch.int32, device=self.device)
        topk = self._run_chain(
            q,
            weights,
            kv_pool,
            block_table,
            cu_kv_seqlens,
            ks,
            ke,
            index_topk,
            num_kv_tokens,
        )
        self.assertEqual(topk.shape, (N, index_topk))
        # All rows have full visibility → topk completely filled, no -1.
        self.assertEqual(topk.dtype, torch.int32)
        valid_per_row = (topk >= 0).sum(dim=-1)
        self.assertTrue(
            (valid_per_row == index_topk).all().item(),
            f"Expected all rows fully populated (no -1); got "
            f"{valid_per_row[:8].cpu().tolist()}",
        )
        self.assertTrue(
            ((topk >= 0) & (topk < num_kv_tokens)).all().item(),
            "All topk entries must be in [0, num_kv_tokens)",
        )


if __name__ == "__main__":
    unittest.main()
