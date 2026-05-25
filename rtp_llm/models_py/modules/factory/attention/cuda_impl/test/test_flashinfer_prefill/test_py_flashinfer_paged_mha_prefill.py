import logging
import unittest
from typing import List

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
    PyFlashinferPrefillPagedAttnOp,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.test.attention_ref import (
    compute_flashinfer_prefill_reference,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.test.base_attention_test import (
    BaseAttentionTest,
    fill_paged_kv_cache,
)
from rtp_llm.ops import KvCacheDataType
from rtp_llm.ops.compute_ops import LayerKVCache

logging.basicConfig(level=logging.INFO, format="%(message)s")


class TestPyFlashinferPrefillPagedAttnOp(BaseAttentionTest):
    """Test suite for PyFlashinferPrefillPagedAttnOp with paged KV cache"""

    atol = 5e-3

    def setUp(self):
        """Set up test fixtures"""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available - this test requires CUDA")

        # Call parent setUp for common initialization
        super().setUp()

    def _create_paged_kv_cache(
        self,
        k_ragged: torch.Tensor,
        v_ragged: torch.Tensor,
        sequence_lengths: List[int],
        page_size: int,
        num_kv_heads: int,
        head_dim: int,
        block_table: torch.Tensor,
        cache_dtype: torch.dtype,
    ) -> LayerKVCache:
        """
        Convert ragged K, V to paged KV cache format (HND layout)

        Args:
            k_ragged: [total_tokens, num_kv_heads, head_dim]
            v_ragged: [total_tokens, num_kv_heads, head_dim]
            sequence_lengths: List of sequence lengths
            page_size: Page size
            num_kv_heads: Number of KV heads
            head_dim: Head dimension
            block_table: [batch, max_pages] page ids the op will read

        Returns:
            paged_kv_cache: [num_pages, 2, num_kv_heads, page_size, head_dim] (HND layout)
        """
        offsets = [0]
        for seq_len in sequence_lengths:
            offsets.append(offsets[-1] + seq_len)
        per_batch = [
            (
                k_ragged[offsets[i] : offsets[i + 1]],
                v_ragged[offsets[i] : offsets[i + 1]],
            )
            for i in range(len(sequence_lengths))
        ]
        return fill_paged_kv_cache(
            [entry[0] for entry in per_batch],
            [entry[1] for entry in per_batch],
            sequence_lengths,
            block_table,
            page_size,
            num_kv_heads,
            head_dim,
            cache_dtype,
            self.device,
        )

    def _test_prefill_correctness(
        self,
        batch_size: int,
        sequence_lengths: List[int],
        head_num: int = 32,
        head_num_kv: int = 8,
        size_per_head: int = 128,
        page_size: int = 64,
        causal: bool = True,
    ):
        """Test prefill correctness by comparing with flashinfer reference implementation"""

        config = self._create_config(
            head_num=head_num,
            head_num_kv=head_num_kv,
            size_per_head=size_per_head,
            seq_size_per_block=page_size,
        )
        config.attn_configs.is_causal = causal

        attn_inputs = self._create_prefill_attention_inputs(
            batch_size, sequence_lengths, config.seq_size_per_block
        )

        # Create PyFlashinferPrefillPagedAttnOp instance
        attn_op = PyFlashinferPrefillPagedAttnOp(
            config.attn_configs,
            attn_inputs,
        )

        # Check support
        if not attn_op.support(attn_inputs):
            raise RuntimeError(
                "PyFlashinferPrefillPagedAttnOp does not support this configuration"
            )

        # Prepare params
        params = attn_op.prepare(attn_inputs)

        # Create Q input
        total_tokens = sum(sequence_lengths)
        hidden_size_q = config.size_per_head * config.head_num

        # Create Q tensor [total_tokens, hidden_size_q]
        q_flat = torch.randn(
            total_tokens,
            hidden_size_q,
            dtype=torch.float16,
            device=self.device,
        )
        q = q_flat.reshape(total_tokens, config.head_num, config.size_per_head)

        # Create K, V for paged cache
        hidden_size_k = config.size_per_head * config.head_num_kv
        hidden_size_v = config.size_per_head * config.head_num_kv

        k_flat = torch.randn(
            total_tokens,
            hidden_size_k,
            dtype=torch.float16,
            device=self.device,
        )
        v_flat = torch.randn(
            total_tokens,
            hidden_size_v,
            dtype=torch.float16,
            device=self.device,
        )
        k = k_flat.reshape(total_tokens, config.head_num_kv, config.size_per_head)
        v = v_flat.reshape(total_tokens, config.head_num_kv, config.size_per_head)

        cache_dtype = self.cache_dtype(config.attn_configs)
        k = k.to(cache_dtype)
        v = v.to(cache_dtype)

        # Create paged KV cache
        paged_kv_cache = self._create_paged_kv_cache(
            k,
            v,
            sequence_lengths,
            page_size,
            config.head_num_kv,
            config.size_per_head,
            attn_inputs.kv_cache_kernel_block_id,
            cache_dtype,
        )

        # Forward pass through PyFlashinferPrefillPagedAttnOp
        output = attn_op.forward(q, paged_kv_cache)  # Use layer 0

        # Compute reference outputs using flashinfer's reference (with round-trip)
        ref_output = compute_flashinfer_prefill_reference(
            q.to(attn_op.q_dtype).to(q.dtype),
            k.to(q.dtype),
            v.to(q.dtype),
            attn_inputs.cu_seqlens_device,
            causal=causal,
        )

        # Compare outputs
        print(
            f"Testing batch_size={batch_size}, seq_lens={sequence_lengths}, "
            f"head_num={head_num}, kv_head_num={head_num_kv}, "
            f"size_per_head={size_per_head}, page_size={page_size}"
        )

        # Assert closeness (with relaxed tolerance for FP16)
        try:
            self._assert_output_close(output, ref_output, name="Prefill output")
            print("✓ Test passed")
        except AssertionError as e:
            logging.error(f"✗ Test failed: {e}")
            raise

    # ========== Test Cases: Single Batch ==========

    def test_single_sequence_small(self):
        """Test single sequence with small length"""
        self._test_prefill_correctness(
            batch_size=1,
            sequence_lengths=[32],
            head_num=8,
            head_num_kv=2,
            size_per_head=64,
            page_size=16,
        )

    def test_non_causal_prefill(self):
        self._test_prefill_correctness(
            batch_size=2,
            sequence_lengths=[64, 96],
            head_num=8,
            head_num_kv=2,
            size_per_head=64,
            page_size=16,
            causal=False,
        )

    def test_single_sequence_medium(self):
        """Test single sequence with medium length"""
        self._test_prefill_correctness(
            batch_size=1,
            sequence_lengths=[128],
            head_num=32,
            head_num_kv=8,
            size_per_head=128,
            page_size=64,
        )

    def test_single_sequence_large(self):
        """Test single sequence with large length"""
        self._test_prefill_correctness(
            batch_size=1,
            sequence_lengths=[512],
            head_num=32,
            head_num_kv=8,
            size_per_head=128,
            page_size=64,
        )

    # ========== Test Cases: Multi Batch ==========

    def test_multi_batch_uniform(self):
        """Test multiple sequences with uniform lengths"""
        self._test_prefill_correctness(
            batch_size=4,
            sequence_lengths=[64, 64, 64, 64],
            head_num=32,
            head_num_kv=8,
            size_per_head=128,
            page_size=64,
        )

    def test_multi_batch_varied(self):
        """Test multiple sequences with varied lengths"""
        self._test_prefill_correctness(
            batch_size=4,
            sequence_lengths=[32, 64, 128, 256],
            head_num=32,
            head_num_kv=8,
            size_per_head=128,
            page_size=64,
        )

    # ========== Test Cases: Chunked Prefill (Prefix Caching) ==========

    def test_chunked_prefill_single_batch(self):
        """Test chunked prefill with single batch (mimics your real scenario)

        Scenario:
        - Existing KV cache: 4884 tokens
        - New Q input: 5 tokens
        - Total KV: 4889 tokens
        """
        print("\n" + "=" * 70)
        print("Testing CHUNKED PREFILL scenario")
        print("  prefix_length: 4884 (existing KV cache)")
        print("  input_length: 5 (new Q tokens)")
        print("  Expected: Q[i] attends to KV[0:4884+i+1]")
        print("=" * 70)

        batch_size = 1
        prefix_lengths = [4884]
        input_lengths = [5]
        page_size = 64
        head_num = 40
        head_num_kv = 8
        size_per_head = 128

        config = self._create_config(
            head_num=head_num,
            head_num_kv=head_num_kv,
            size_per_head=size_per_head,
            seq_size_per_block=page_size,
        )

        # Create chunked prefill attention inputs
        attn_inputs = self._create_chunked_prefill_attention_inputs(
            batch_size, prefix_lengths, input_lengths, config.seq_size_per_block
        )

        # Create PyFlashinferPrefillPagedAttnOp instance
        attn_op = PyFlashinferPrefillPagedAttnOp(config.attn_configs, attn_inputs)

        # Check support
        if not attn_op.support(attn_inputs):
            raise RuntimeError(
                "PyFlashinferPrefillPagedAttnOp does not support chunked prefill"
            )

        # Prepare params
        params = attn_op.prepare(attn_inputs)

        # Create Q input (only for new tokens)
        total_q_tokens = sum(input_lengths)  # 5

        q = torch.randn(
            total_q_tokens,
            config.head_num,
            config.size_per_head,
            dtype=torch.float16,
            device=self.device,
        )

        # Create K, V for FULL sequence (prefix + input)
        total_kv_tokens = sum(
            [p + i for p, i in zip(prefix_lengths, input_lengths)]
        )  # 4889

        k = torch.randn(
            total_kv_tokens,
            config.head_num_kv,
            config.size_per_head,
            dtype=torch.float16,
            device=self.device,
        )
        v = torch.randn(
            total_kv_tokens,
            config.head_num_kv,
            config.size_per_head,
            dtype=torch.float16,
            device=self.device,
        )

        cache_dtype = self.cache_dtype(config.attn_configs)
        k = k.to(cache_dtype)
        v = v.to(cache_dtype)

        # Create paged KV cache
        sequence_lengths = [p + i for p, i in zip(prefix_lengths, input_lengths)]
        paged_kv_cache = self._create_paged_kv_cache(
            k,
            v,
            sequence_lengths,
            page_size,
            config.head_num_kv,
            config.size_per_head,
            attn_inputs.kv_cache_kernel_block_id,
            cache_dtype,
        )

        # Forward pass through PyFlashinferPrefillPagedAttnOp
        print("\nRunning FlashInfer forward pass...")
        output = attn_op.forward(q, paged_kv_cache)

        print(f"Output shape: {output.shape}")
        print(f"Output has NaN: {torch.isnan(output).any().item()}")
        print(f"Output has Inf: {torch.isinf(output).any().item()}")

        # Try to verify the output is not all NaN
        if torch.isnan(output).all():
            raise RuntimeError(
                "❌ All output is NaN! FlashInfer chunked prefill failed!"
            )

        print("✅ Test completed (output not all NaN)")

        # ========== Correctness Verification (Simplified) ==========
        print("\n" + "=" * 70)
        print("Computing reference output (simplified approach)...")
        print("=" * 70)

        # 简化方法：构造完整的 Q（前面用0填充），然后只取最后几个输出
        # 这样可以直接用标准的 single_prefill_with_kv_cache

        from flashinfer.prefill import single_prefill_with_kv_cache

        # 构造完整长度的 Q（和 K/V 一样长）
        # 前 prefix_len 个位置填0，后 input_len 个位置是真实的 Q
        prefix_len = prefix_lengths[0]
        input_len = input_lengths[0]
        seq_len = prefix_len + input_len

        # Q_full: [seq_len, num_heads, head_dim]
        q_full = torch.zeros(
            seq_len,
            config.head_num,
            config.size_per_head,
            dtype=torch.float16,
            device=self.device,
        )
        # with round-trip
        q_full[prefix_len:] = q.to(attn_op.q_dtype).to(q.dtype)  # 把真实的 Q 放在后面

        print(f"  Q_full shape: {q_full.shape} (padded)")
        print(f"  K shape: {k.shape}")
        print(f"  Prefix: {prefix_len}, Input: {input_len}")

        # 用 FlashInfer 计算完整的 attention
        ref_output_full = single_prefill_with_kv_cache(
            q_full, k.to(q.dtype), v.to(q.dtype), causal=True, kv_layout="NHD"
        )

        # 只取最后 input_len 个输出（对应真实的 Q）
        ref_output = ref_output_full[prefix_len:]

        print(f"\n[Reference Output]")
        print(f"  Shape: {ref_output.shape}")
        print(f"  Has NaN: {torch.isnan(ref_output).any().item()}")
        print(
            f"  Range: [{ref_output.min().item():.4f}, {ref_output.max().item():.4f}]"
        )

        print(f"\n[Test Output]")
        print(f"  Shape: {output.shape}")
        print(f"  Has NaN: {torch.isnan(output).any().item()}")
        if not torch.isnan(output).any():
            print(f"  Range: [{output.min().item():.4f}, {output.max().item():.4f}]")

        # Compare outputs
        print(f"\n[Correctness Check]")
        try:
            self._assert_output_close(
                output,
                ref_output,
                rtol=max(self.rtol, 1e-2),
                atol=max(self.atol, 1e-2),
                name="Chunked prefill output",
            )
            print("✅ Correctness check PASSED!")
        except AssertionError as e:
            print(f"❌ Correctness check FAILED: {e}")

            # Detailed debugging
            diff = (output - ref_output).abs()
            print(f"\n[Debugging Info]")
            print(f"  Max absolute difference: {diff.max().item():.6f}")
            print(f"  Mean absolute difference: {diff.mean().item():.6f}")
            print(f"  Median absolute difference: {diff.median().item():.6f}")

            # Find tokens with largest errors
            max_diff_idx = int(diff.view(-1).argmax().item())
            token_idx = max_diff_idx // (config.head_num * config.size_per_head)
            print(f"  Token with max error: {token_idx}")
            print(f"    Test output: {output.view(-1)[max_diff_idx].item():.6f}")
            print(f"    Ref output: {ref_output.view(-1)[max_diff_idx].item():.6f}")

            raise

    def test_multi_batch_small_lengths(self):
        """Test multiple sequences with small lengths"""
        self._test_prefill_correctness(
            batch_size=3,
            sequence_lengths=[8, 16, 24],
            head_num=8,
            head_num_kv=2,
            size_per_head=64,
            page_size=16,
        )

    # ========== Test Cases: Different Page Sizes ==========

    def test_small_page_size(self):
        """Test with small page size"""
        self._test_prefill_correctness(
            batch_size=2,
            sequence_lengths=[128, 256],
            head_num=32,
            head_num_kv=8,
            size_per_head=128,
            page_size=32,
        )

    def test_large_page_size(self):
        """Test with large page size"""
        self._test_prefill_correctness(
            batch_size=2,
            sequence_lengths=[128, 256],
            head_num=32,
            head_num_kv=8,
            size_per_head=128,
            page_size=128,
        )

    # ========== Test Cases: Different Head Configurations ==========

    def test_many_heads(self):
        """Test with many heads"""
        self._test_prefill_correctness(
            batch_size=2,
            sequence_lengths=[64, 128],
            head_num=64,
            head_num_kv=16,
            size_per_head=128,
            page_size=64,
        )

    def test_gqa_4(self):
        """Test with GQA group size 4"""
        self._test_prefill_correctness(
            batch_size=2,
            sequence_lengths=[64, 128],
            head_num=32,
            head_num_kv=8,  # 32/8 = 4 queries per KV
            size_per_head=128,
            page_size=64,
        )


class TestPyFlashinferPrefillPagedAttnOpFP8(TestPyFlashinferPrefillPagedAttnOp):
    kv_cache_dtype = KvCacheDataType.FP8
    rtol = 4e-2
    atol = 4e-2
    max_mismatch_rate = 1e-5


if __name__ == "__main__":
    unittest.main()
