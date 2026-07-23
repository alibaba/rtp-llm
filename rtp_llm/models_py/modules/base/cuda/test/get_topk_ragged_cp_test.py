"""
Unit test for IndexerOp._get_topk_ragged_cp.

Tests the CP path for computing TopK indices in prefill with context parallel:
single rank with one chunk so that generate_q_indices yields valid indices
for index_select on local q_fp8.
"""

from pathlib import Path
from unittest import SkipTest, TestCase, main

import torch

from rtp_llm.ops.compute_ops import (
    LayerKVCache,
    PyAttentionInputs,
    PyContextParallelParams,
    rtp_llm_ops,
)


def _check_cuda_deep_gemm():
    try:
        if not torch.cuda.is_available():
            return False
        import deep_gemm

        package_dir = Path(deep_gemm.__file__).resolve().parent
        return (package_dir / "include" / "deep_gemm").is_dir()
    except ImportError:
        return False


CUDA_DEEPGEMM_OK = _check_cuda_deep_gemm()
SKIP_REASON = (
    "CUDA and a complete deep_gemm JIT package are required for "
    "IndexerOp._get_topk_ragged_cp"
)


class CPGatherIndexerKQuantCacheTest(TestCase):
    head_dim = 128
    cache_block_size = 4
    cache_stride = 132

    def setUp(self):
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is required")
        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)

    def _make_cache(self):
        cache = torch.zeros(
            (2, self.cache_block_size, self.cache_stride),
            dtype=torch.uint8,
            device=self.device,
        )
        flat = cache.flatten()
        for block in range(cache.size(0)):
            block_base = block * cache.stride(0)
            for token in range(self.cache_block_size):
                value = block * 10 + token + 1
                start = block_base + token * self.head_dim
                flat[start : start + self.head_dim] = value
                scale_byte_offset = (
                    block_base
                    + self.cache_block_size * self.head_dim
                    + token * 4
                )
                flat[scale_byte_offset : scale_byte_offset + 4].view(
                    torch.float32
                ).fill_(float(value) / 10.0)
        return cache

    def _gather(self, block_table, cu_seq_lens, num_tokens):
        dst_k = torch.full(
            (num_tokens, self.head_dim),
            0xA5,
            dtype=torch.uint8,
            device=self.device,
        )
        dst_scale = torch.full(
            (num_tokens, 4),
            0xA5,
            dtype=torch.uint8,
            device=self.device,
        )
        rtp_llm_ops.cp_gather_indexer_k_quant_cache(
            self._make_cache(),
            dst_k,
            dst_scale,
            torch.tensor(block_table, dtype=torch.int32, device=self.device),
            torch.tensor(cu_seq_lens, dtype=torch.int32, device=self.device),
        )
        torch.cuda.synchronize()
        return dst_k, dst_scale.view(torch.float32)

    def test_gathers_logical_pages_from_physical_block_table(self):
        dst_k, dst_scale = self._gather([[1, 0]], [0, 6], num_tokens=6)

        expected_values = torch.tensor(
            [11, 12, 13, 14, 1, 2], dtype=torch.uint8, device=self.device
        )
        torch.testing.assert_close(dst_k[:, 0], expected_values)
        torch.testing.assert_close(
            dst_scale[:, 0], expected_values.to(torch.float32) / 10.0
        )

    def test_invalid_physical_blocks_are_zero_filled(self):
        for block_id in (-1, 2):
            with self.subTest(block_id=block_id):
                dst_k, dst_scale = self._gather(
                    [[block_id]], [0, 2], num_tokens=2
                )
                self.assertEqual(torch.count_nonzero(dst_k).item(), 0)
                self.assertEqual(torch.count_nonzero(dst_scale).item(), 0)

    def test_tokens_beyond_logical_block_table_are_zero_filled(self):
        dst_k, dst_scale = self._gather([[0]], [0, 6], num_tokens=6)

        torch.testing.assert_close(
            dst_k[:4, 0],
            torch.tensor([1, 2, 3, 4], dtype=torch.uint8, device=self.device),
        )
        self.assertEqual(torch.count_nonzero(dst_k[4:]).item(), 0)
        self.assertEqual(torch.count_nonzero(dst_scale[4:]).item(), 0)

    def test_tokens_outside_sequence_ranges_are_zero_filled(self):
        dst_k, dst_scale = self._gather([[0]], [1, 3], num_tokens=3)

        self.assertEqual(torch.count_nonzero(dst_k[0]).item(), 0)
        self.assertEqual(torch.count_nonzero(dst_scale[0]).item(), 0)
        torch.testing.assert_close(
            dst_k[1:, 0],
            torch.tensor([1, 2], dtype=torch.uint8, device=self.device),
        )


class GetTopkRaggedCPTest(TestCase):
    """Test IndexerOp._get_topk_ragged_cp with single rank and one chunk."""

    def setUp(self):
        if not CUDA_DEEPGEMM_OK:
            raise SkipTest(SKIP_REASON)
        from rtp_llm.models_py.modules.base.cuda.indexer_op import IndexerOp
        from rtp_llm.models_py.modules.factory.attention.cuda_cp_impl.prefill_mha.cp_utils import (
            generate_q_indices,
        )

        self.IndexerOp = IndexerOp
        self.generate_q_indices = generate_q_indices
        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)
        torch.manual_seed(42)

    def test_get_topk_ragged_cp_shape_and_no_crash(self):
        """
        Single CP rank, one chunk of 8 tokens.
        generate_q_indices([8]) -> q0_idx=[0,1,2,3], q1_idx=[4,5,6,7];
        q_fp8 has 8 rows, so index_select is valid.
        """
        # deep_gemm fp8_mqa_logits requires seq_len_alignment % block_q == 0 with block_q = 128/num_heads; use 32 heads so block_q=4.
        index_n_heads = 32
        index_head_dim = 128
        index_topk = 2048
        block_size = 128
        rope_head_dim = 64
        total_tokens = 8
        chunk_lengths = [8]

        op = self.IndexerOp(
            index_n_heads=index_n_heads,
            index_head_dim=index_head_dim,
            index_topk=index_topk,
            rope_head_dim=rope_head_dim,
            cos_sin_cache=None,
            blocksize=64,
            block_size=block_size,
        )

        device = self.device
        total_local_ids = torch.arange(total_tokens, device=device, dtype=torch.long)
        total_global_ids = torch.arange(total_tokens, device=device, dtype=torch.long)
        cu_kv_seqlens_global = torch.tensor(
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
        kv_cache = LayerKVCache()
        kv_cache.kv_scale_base = kv_scale_base

        attn_inputs = PyAttentionInputs()
        attn_inputs.kv_cache_kernel_block_id_host = torch.tensor(
            [[0]], dtype=torch.int32, device=torch.device("cpu")
        )
        attn_inputs.kv_cache_kernel_block_id_device = torch.tensor(
            [[0]], dtype=torch.int32, device=device
        )
        attn_inputs.cu_kv_seqlens = torch.tensor(
            [0, total_tokens], dtype=torch.int32, device=device
        )
        cp_params = PyContextParallelParams()
        cp_params.prefill_cp_chunk_lengths = torch.tensor(
            chunk_lengths, dtype=torch.int32, device=device
        )
        attn_inputs.context_parallel_info = cp_params

        ks = torch.arange(total_tokens, dtype=torch.int32, device=device)
        ke = torch.arange(1, total_tokens + 1, dtype=torch.int32, device=device)
        expanded_seq_lens = torch.ones(total_tokens, dtype=torch.int32, device=device)
        topk_indices_offset = torch.zeros(
            total_tokens, dtype=torch.int32, device=device
        )

        class FmhaParams:
            pass

        fmha_params = FmhaParams()
        fmha_params.ks = ks
        fmha_params.ke = ke
        fmha_params.expanded_seq_lens = expanded_seq_lens
        fmha_params.topk_indices_offset = topk_indices_offset

        q0_idx_list, _q1_idx_list = self.generate_q_indices(chunk_lengths)
        n0 = len(q0_idx_list)

        # Precompute indexed params (simulates what create_params does)
        precomputed_ks = fmha_params.ks[total_global_ids]
        precomputed_ke = fmha_params.ke[total_global_ids]
        precomputed_lengths = fmha_params.expanded_seq_lens[total_global_ids]
        precomputed_topk_off = fmha_params.topk_indices_offset[total_global_ids]

        topk_result = op._get_topk_ragged_cp(
            q_fp8,
            weights,
            kv_cache,
            fmha_params,
            attn_inputs,
            total_local_ids,
            cu_kv_seqlens_global,
            total_tokens,
            precomputed_ks,
            precomputed_ke,
            precomputed_lengths,
            precomputed_topk_off,
        )
        topk0 = topk_result[:n0]
        topk1 = topk_result[n0:]

        self.assertIsInstance(topk0, torch.Tensor)
        self.assertIsInstance(topk1, torch.Tensor)
        self.assertEqual(topk0.dtype, torch.int32)
        self.assertEqual(topk1.dtype, torch.int32)
        self.assertEqual(topk0.device, q_fp8.device)
        self.assertEqual(topk1.device, q_fp8.device)
        self.assertEqual(topk0.dim(), 2)
        self.assertEqual(topk1.dim(), 2)
        self.assertEqual(topk0.shape[1], index_topk)
        self.assertEqual(topk1.shape[1], index_topk)
        self.assertEqual(
            topk0.shape[0], 4, "q0 has 4 rows from generate_q_indices([8])"
        )
        self.assertEqual(
            topk1.shape[0], 4, "q1 has 4 rows from generate_q_indices([8])"
        )

    def test_get_topk_ragged_cp_with_prefix_cache(self):
        """
        Single CP rank, one chunk of 8 tokens with prefix_length=64 (page-aligned).
        Total KV = prefix(64) + input(8) = 72 tokens.
        q_fp8 has 8 rows (input only), topk computed over full 72-token KV range.
        ks/ke shifted by prefix_length so each query row sees prefix + its own context.
        """
        index_n_heads = 32
        index_head_dim = 128
        index_topk = 2048
        block_size = 128
        rope_head_dim = 64
        input_tokens = 8
        prefix_length = 64
        total_kv_tokens = input_tokens + prefix_length  # 72
        chunk_lengths = [8]

        op = self.IndexerOp(
            index_n_heads=index_n_heads,
            index_head_dim=index_head_dim,
            index_topk=index_topk,
            rope_head_dim=rope_head_dim,
            cos_sin_cache=None,
            blocksize=64,
            block_size=block_size,
        )

        device = self.device
        total_local_ids = torch.arange(input_tokens, device=device, dtype=torch.long)
        total_global_ids = torch.arange(input_tokens, device=device, dtype=torch.long)
        # cu_kv_seqlens_global includes prefix: [0, prefix + input]
        cu_kv_seqlens_global = torch.tensor(
            [0, total_kv_tokens], dtype=torch.int32, device=device
        )

        q_fp8 = torch.randn(
            input_tokens,
            index_n_heads,
            index_head_dim,
            dtype=torch.float32,
            device=device,
        ).to(torch.float8_e4m3fn)
        weights = torch.randn(
            input_tokens, index_n_heads, 1, dtype=torch.float32, device=device
        )

        # Allocate enough blocks for total_kv_tokens
        import math

        page_size = 64
        num_blocks = math.ceil(total_kv_tokens / page_size)
        cache_stride = index_head_dim + (index_head_dim // block_size) * 4
        kv_scale_base = torch.empty(
            num_blocks,
            block_size,
            cache_stride,
            dtype=torch.uint8,
            device=device,
        )
        kv_cache = LayerKVCache()
        kv_cache.kv_scale_base = kv_scale_base

        attn_inputs = PyAttentionInputs()
        attn_inputs.kv_cache_kernel_block_id_device = torch.arange(
            num_blocks, dtype=torch.int32, device=device
        ).unsqueeze(0)
        attn_inputs.cu_kv_seqlens = torch.tensor(
            [0, total_kv_tokens], dtype=torch.int32, device=device
        )
        cp_params = PyContextParallelParams()
        cp_params.prefill_cp_chunk_lengths = torch.tensor(
            chunk_lengths, dtype=torch.int32, device=device
        )
        attn_inputs.context_parallel_info = cp_params

        # ks/ke shifted by prefix_length: each query token i sees KV[prefix+i : prefix+i+1]
        ks = (
            torch.arange(input_tokens, dtype=torch.int32, device=device) + prefix_length
        )
        ke = (
            torch.arange(1, input_tokens + 1, dtype=torch.int32, device=device)
            + prefix_length
        )
        expanded_seq_lens = torch.ones(input_tokens, dtype=torch.int32, device=device)
        topk_indices_offset = torch.zeros(
            input_tokens, dtype=torch.int32, device=device
        )

        class FmhaParams:
            pass

        fmha_params = FmhaParams()
        fmha_params.ks = ks
        fmha_params.ke = ke
        fmha_params.expanded_seq_lens = expanded_seq_lens
        fmha_params.topk_indices_offset = topk_indices_offset

        q0_idx_list, _q1_idx_list = self.generate_q_indices(chunk_lengths)
        n0 = len(q0_idx_list)

        # Precompute indexed params (simulates what create_params does)
        precomputed_ks = fmha_params.ks[total_global_ids]
        precomputed_ke = fmha_params.ke[total_global_ids]
        precomputed_lengths = fmha_params.expanded_seq_lens[total_global_ids]
        precomputed_topk_off = fmha_params.topk_indices_offset[total_global_ids]

        topk_result = op._get_topk_ragged_cp(
            q_fp8,
            weights,
            kv_cache,
            fmha_params,
            attn_inputs,
            total_local_ids,
            cu_kv_seqlens_global,
            total_kv_tokens,
            precomputed_ks,
            precomputed_ke,
            precomputed_lengths,
            precomputed_topk_off,
        )
        topk0 = topk_result[:n0]
        topk1 = topk_result[n0:]

        self.assertIsInstance(topk0, torch.Tensor)
        self.assertIsInstance(topk1, torch.Tensor)
        self.assertEqual(topk0.dtype, torch.int32)
        self.assertEqual(topk1.dtype, torch.int32)
        self.assertEqual(topk0.device, q_fp8.device)
        self.assertEqual(topk1.device, q_fp8.device)
        self.assertEqual(topk0.dim(), 2)
        self.assertEqual(topk1.dim(), 2)
        self.assertEqual(topk0.shape[1], index_topk)
        self.assertEqual(topk1.shape[1], index_topk)
        self.assertEqual(
            topk0.shape[0], 4, "q0 has 4 rows from generate_q_indices([8])"
        )
        self.assertEqual(
            topk1.shape[0], 4, "q1 has 4 rows from generate_q_indices([8])"
        )


if __name__ == "__main__":
    main()
