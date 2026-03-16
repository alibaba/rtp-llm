"""
Unit test for IndexerOp._get_topk_ragged_cp.

Tests the CP path for computing TopK indices in prefill with context parallel:
single rank with one chunk so that generate_q_indices yields valid indices
for index_select on local q_fp8.

Also tests the prefix cache (reuse cache) variant where cu_kv_seqlens_global
includes prefix tokens, verifying that buffer allocation uses the full KV
length (prefix + new) rather than just new tokens.
"""

import math
from unittest import SkipTest, TestCase, main

import torch

from rtp_llm.models_py.modules.base.cuda.indexer_op import IndexerOp
from rtp_llm.models_py.modules.factory.attention.cuda_cp_impl.prefill_mha.cp_utils import (
    generate_q_indices,
)
from rtp_llm.ops.compute_ops import (
    KVCache,
    PyAttentionInputs,
    PyContextParallelParams,
    rtp_llm_ops,
)


def _check_cuda_deep_gemm():
    try:
        if not torch.cuda.is_available():
            return False
        import deep_gemm  # noqa: F401

        return True
    except ImportError:
        return False


CUDA_DEEPGEMM_OK = _check_cuda_deep_gemm()
SKIP_REASON = "CUDA and deep_gemm required for IndexerOp._get_topk_ragged_cp"


def _setup_indexer_op_cp(
    index_n_heads,
    index_head_dim,
    index_topk,
    block_size,
    rope_head_dim,
    chunk_lengths,
    cp_rank,
    device,
):
    """Create an IndexerOp and set up CP indices for single-rank testing."""
    op = IndexerOp(
        index_n_heads=index_n_heads,
        index_head_dim=index_head_dim,
        index_topk=index_topk,
        rope_head_dim=rope_head_dim,
        cos_sin_cache=None,
        blocksize=64,
        block_size=block_size,
    )

    q0_idx_list, q1_idx_list = generate_q_indices(chunk_lengths)
    op.q0_idx = torch.tensor(q0_idx_list, device=device, dtype=torch.long)
    op.q1_idx = torch.tensor(q1_idx_list, device=device, dtype=torch.long)
    local_tokens = sum(chunk_lengths)
    op.q0_idx_global = cp_rank * local_tokens + op.q0_idx
    op.q1_idx_global = cp_rank * local_tokens + op.q1_idx
    op.total_local_ids = torch.cat([op.q0_idx, op.q1_idx], dim=0)
    op.total_global_ids = torch.cat([op.q0_idx_global, op.q1_idx_global], dim=0)

    return op


class FmhaParams:
    pass


class GetTopkRaggedCPTest(TestCase):
    """Test IndexerOp._get_topk_ragged_cp with single rank and one chunk."""

    def setUp(self):
        if not CUDA_DEEPGEMM_OK:
            raise SkipTest(SKIP_REASON)
        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)
        torch.manual_seed(42)

    def test_get_topk_ragged_cp_shape_and_no_crash(self):
        """
        Single CP rank, one chunk of 8 tokens, no prefix.
        generate_q_indices([8]) -> q0_idx=[0,1,2,3], q1_idx=[4,5,6,7];
        q_fp8 has 8 rows, so index_select is valid.
        """
        index_n_heads = 32
        index_head_dim = 128
        index_topk = 2048
        block_size = 128
        rope_head_dim = 64
        total_tokens = 8
        chunk_lengths = [8]
        cp_rank = 0
        cp_size = 1
        device = self.device

        op = _setup_indexer_op_cp(
            index_n_heads,
            index_head_dim,
            index_topk,
            block_size,
            rope_head_dim,
            chunk_lengths,
            cp_rank,
            device,
        )
        op.cu_kv_seqlens_global = torch.tensor(
            [0, total_tokens], dtype=torch.int32, device=device
        )
        op.kv_len = total_tokens
        op.total_kv_len = total_tokens

        device = self.device
        op.total_local_ids = torch.arange(total_tokens, device=device, dtype=torch.long)
        op.total_global_ids = torch.arange(
            total_tokens, device=device, dtype=torch.long
        )
        op.cu_kv_seqlens_global = torch.tensor(
            [0, total_tokens], dtype=torch.int32, device=device
        )
        q_fp8 = torch.randn(
            total_tokens,
            index_n_heads,
            index_head_dim,
            dtype=torch.float32,
            device=device,
        ).to(torch.float8_e4m3fn)
        weights = torch.randn(
            total_tokens, index_n_heads, 1, dtype=torch.float32, device=device
        )

        num_blocks = 1
        cache_stride = index_head_dim + (index_head_dim // block_size) * 4
        kv_scale_base = torch.empty(
            num_blocks,
            block_size,
            cache_stride,
            dtype=torch.uint8,
            device=device,
        )
        kv_cache = KVCache()
        kv_cache.kv_scale_base = kv_scale_base

        attn_inputs = PyAttentionInputs()
        attn_inputs.kv_cache_block_id_device = torch.tensor(
            [[0]], dtype=torch.int32, device=device
        )
        attn_inputs.cu_kv_seqlens = torch.tensor(
            [0, total_tokens], dtype=torch.int32, device=device
        )

        ks = torch.arange(total_tokens, dtype=torch.int32, device=device)
        ke = torch.arange(1, total_tokens + 1, dtype=torch.int32, device=device)
        expanded_seq_lens = torch.ones(total_tokens, dtype=torch.int32, device=device)
        topk_indices_offset = torch.zeros(
            total_tokens, dtype=torch.int32, device=device
        )

        fmha_params = FmhaParams()
        fmha_params.ks = ks
        fmha_params.ke = ke
        fmha_params.expanded_seq_lens = expanded_seq_lens
        fmha_params.topk_indices_offset = topk_indices_offset

        topk = op._get_topk_ragged_cp(
            q_fp8,
            weights,
            kv_cache,
            fmha_params,
            attn_inputs,
            cp_rank,
            cp_size,
        )

        self.assertIsInstance(topk, torch.Tensor)
        self.assertEqual(topk.dtype, torch.int32)
        self.assertEqual(topk.device, q_fp8.device)
        self.assertEqual(topk.dim(), 2)
        self.assertEqual(topk.shape[1], index_topk)
        self.assertEqual(
            topk.shape[0],
            total_tokens,
            "total_local_ids covers all 8 tokens from generate_q_indices([8])",
        )

    def test_get_topk_ragged_cp_with_prefix(self):
        """
        Single CP rank, 8 new tokens, 128 prefix tokens (reuse cache).

        Verifies that _get_topk_ragged_cp correctly allocates buffers using
        cu_kv_seqlens_global[-1] = prefix_len + new_tokens, rather than
        the old kv_restore_unpad_indices.shape[0] = new_tokens only.

        topk_result = op._get_topk_ragged_cp(
            q_fp8,
            weights,
            kv_cache,
            fmha_params,
            attn_inputs,
            cp_rank,
            cp_size,
        )
        n0 = op.q0_idx.size(0)
        topk0 = topk_result[:n0]
        topk1 = topk_result[n0:]

        self.assertIsInstance(topk, torch.Tensor)
        self.assertEqual(topk.dtype, torch.int32)
        self.assertEqual(topk.dim(), 2)
        self.assertEqual(topk.shape[1], index_topk)
        self.assertEqual(
            topk.shape[0],
            new_tokens,
            "total_local_ids covers all 8 new tokens",
        )
        # All topk indices should be within the full KV range [0, total_kv)
        self.assertTrue(
            (topk >= 0).all() and (topk < total_kv).all(),
            f"topk indices should be in [0, {total_kv}), "
            f"got min={topk.min().item()}, max={topk.max().item()}",
        )
        # Verify some indices reference prefix positions (< prefix_len),
        # confirming that the buffer was sized correctly to include prefix
        has_prefix_refs = (topk < prefix_len).any().item()
        self.assertTrue(
            has_prefix_refs,
            "With prefix_len=128, topk should reference some prefix positions",
        )

    def test_workspace_buffer_reuse(self):
        """Verify that workspace buffers are reused across calls."""
        index_n_heads = 32
        index_head_dim = 128
        index_topk = 2048
        block_size = 128
        rope_head_dim = 64
        total_tokens = 8
        chunk_lengths = [8]
        cp_rank = 0
        cp_size = 1
        device = self.device

        op = _setup_indexer_op_cp(
            index_n_heads,
            index_head_dim,
            index_topk,
            block_size,
            rope_head_dim,
            chunk_lengths,
            cp_rank,
            device,
        )
        op.cu_kv_seqlens_global = torch.tensor(
            [0, total_tokens], dtype=torch.int32, device=device
        )

        q_fp8 = torch.randn(
            total_tokens,
            index_n_heads,
            index_head_dim,
            dtype=torch.float32,
            device=device,
        ).to(torch.float8_e4m3fn)
        weights = torch.randn(
            total_tokens, index_n_heads, 1, dtype=torch.float32, device=device
        )

        cache_stride = index_head_dim + (index_head_dim // block_size) * 4
        kv_scale_base = torch.empty(
            1,
            block_size,
            cache_stride,
            dtype=torch.uint8,
            device=device,
        )
        kv_cache = KVCache()
        kv_cache.kv_scale_base = kv_scale_base

        attn_inputs = PyAttentionInputs()
        attn_inputs.kv_cache_block_id_device = torch.tensor(
            [[0]], dtype=torch.int32, device=device
        )

        ks = torch.arange(total_tokens, dtype=torch.int32, device=device)
        ke = torch.arange(1, total_tokens + 1, dtype=torch.int32, device=device)

        fmha_params = FmhaParams()
        fmha_params.ks = ks
        fmha_params.ke = ke
        fmha_params.expanded_seq_lens = torch.ones(
            total_tokens, dtype=torch.int32, device=device
        )
        fmha_params.topk_indices_offset = torch.zeros(
            total_tokens, dtype=torch.int32, device=device
        )

        self.assertIsNone(op._workspace_k_fp8)

        op._get_topk_ragged_cp(
            q_fp8,
            weights,
            kv_cache,
            fmha_params,
            attn_inputs,
            cp_rank,
            cp_size,
        )

        self.assertIsNotNone(op._workspace_k_fp8)
        buf1_ptr = op._workspace_k_fp8.data_ptr()

        # Second call should reuse the same buffer
        op._get_topk_ragged_cp(
            q_fp8,
            weights,
            kv_cache,
            fmha_params,
            attn_inputs,
            cp_rank,
            cp_size,
        )

        self.assertEqual(
            op._workspace_k_fp8.data_ptr(),
            buf1_ptr,
            "Workspace buffer should be reused across calls",
        )


if __name__ == "__main__":
    main()
