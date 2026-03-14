"""
Unit test for IndexerOp._get_topk_ragged_cp.

Tests the CP path for computing TopK indices in prefill with context parallel:
single rank with one chunk so that generate_q_indices yields valid indices
for index_select on local q_fp8.
"""

from unittest import SkipTest, TestCase, main

import torch

from rtp_llm.models_py.modules.base.cuda.indexer_op import IndexerOp
from rtp_llm.models_py.modules.factory.attention.cuda_cp_impl.prefill_mha.cp_utils import (
    generate_q_indices,
)
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
        import deep_gemm  # noqa: F401

        return True
    except ImportError:
        return False


CUDA_DEEPGEMM_OK = _check_cuda_deep_gemm()
SKIP_REASON = "CUDA and deep_gemm required for IndexerOp._get_topk_ragged_cp"


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

        op = IndexerOp(
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

        q0_idx_list, _q1_idx_list = generate_q_indices(chunk_lengths)
        n0 = len(q0_idx_list)

        topk_result = op._get_topk_ragged_cp(
            q_fp8,
            weights,
            kv_cache,
            fmha_params,
            attn_inputs,
            total_local_ids,
            total_global_ids,
            cu_kv_seqlens_global,
            total_tokens,
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
