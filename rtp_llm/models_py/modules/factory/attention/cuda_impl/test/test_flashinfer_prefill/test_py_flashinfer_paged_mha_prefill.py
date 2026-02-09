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
    compare_tensors,
)
from rtp_llm.ops.compute_ops import KVCache, PyAttentionInputs

logging.basicConfig(level=logging.INFO, format="%(message)s")


class TestPyFlashinferPrefillPagedAttnOp(BaseAttentionTest):
    """Test suite for PyFlashinferPrefillPagedAttnOp with paged KV cache"""

    def setUp(self):
        """Set up test fixtures"""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available - this test requires CUDA")

        # Call parent setUp for common initialization
        super().setUp()

    def _create_chunked_prefill_attention_inputs(
        self,
        batch_size: int,
        prefix_lengths: List[int],
        input_lengths: List[int],
        seq_size_per_block: int,
    ) -> PyAttentionInputs:
        """Helper to create PyAttentionInputs for chunked prefill mode

        Args:
            batch_size: Number of sequences in the batch
            prefix_lengths: List of prefix lengths (existing KV cache) for each batch
            input_lengths: List of input lengths (new Q tokens) for each batch
            seq_size_per_block: Number of tokens per block (page size)

        Returns:
            PyAttentionInputs configured for chunked prefill mode
        """
        attn_inputs = PyAttentionInputs()

        # Prefill mode
        attn_inputs.is_prefill = True

        # input_lengths is the length of new Q tokens
        attn_inputs.input_lengths = torch.tensor(
            input_lengths, dtype=torch.int32, device="cpu"
        ).pin_memory()

        # prefix_lengths is the length of existing KV cache
        attn_inputs.prefix_lengths = torch.tensor(
            prefix_lengths, dtype=torch.int32, device="cpu"
        ).pin_memory()

        # sequence_lengths is prefix + input (total KV length)
        sequence_lengths = [p + i for p, i in zip(prefix_lengths, input_lengths)]
        attn_inputs.sequence_lengths = torch.tensor(
            sequence_lengths, dtype=torch.int32, device="cpu"
        ).pin_memory()

        # Create KV cache block IDs for total sequence
        kv_cache_block_id = self._create_kv_cache_block_ids(
            batch_size, sequence_lengths, seq_size_per_block
        )
        attn_inputs.kv_cache_block_id_host = kv_cache_block_id
        attn_inputs.kv_cache_block_id_device = kv_cache_block_id.to(self.device)

        # Create cu_seqlens (cumulative input lengths, NOT sequence lengths!)
        cu_seqlens = [0]
        for input_len in input_lengths:
            cu_seqlens.append(cu_seqlens[-1] + input_len)
        attn_inputs.cu_seqlens = torch.tensor(
            cu_seqlens, dtype=torch.int32, device=self.device
        )

        return attn_inputs

    def _create_paged_kv_cache(
        self,
        k_ragged: torch.Tensor,
        v_ragged: torch.Tensor,
        sequence_lengths: List[int],
        page_size: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> KVCache:
        """
        Convert ragged K, V to paged KV cache format (HND layout)

        Args:
            k_ragged: [total_tokens, num_kv_heads, head_dim]
            v_ragged: [total_tokens, num_kv_heads, head_dim]
            sequence_lengths: List of sequence lengths
            page_size: Page size
            num_kv_heads: Number of KV heads
            head_dim: Head dimension

        Returns:
            paged_kv_cache: [num_layers, num_pages, 2, num_kv_heads, page_size, head_dim] (HND layout)
        """
        total_pages = sum(
            (seq_len + page_size - 1) // page_size for seq_len in sequence_lengths
        )
        num_layers = 1  # Single layer for testing

        # Allocate paged KV cache in HND format: [num_layers, num_pages, 2, num_kv_heads, page_size, head_dim]
        paged_kv_cache = torch.zeros(
            num_layers,
            total_pages,
            2,
            num_kv_heads,
            page_size,
            head_dim,
            dtype=k_ragged.dtype,
            device=self.device,
        )

        page_idx = 0
        token_offset = 0

        for seq_len in sequence_lengths:
            num_pages = (seq_len + page_size - 1) // page_size

            # Fill pages with K, V data
            for i in range(num_pages):
                start_token = i * page_size
                end_token = min(start_token + page_size, seq_len)
                num_tokens_in_page = end_token - start_token

                # Copy K, V to page (layer 0) in HND layout
                # k_ragged/v_ragged shape: [total_tokens, num_kv_heads, head_dim]
                # paged_kv_cache shape: [num_layers, num_pages, 2, num_kv_heads, page_size, head_dim]
                paged_kv_cache[0, page_idx, 0, :, :num_tokens_in_page, :] = k_ragged[
                    token_offset + start_token : token_offset + end_token
                ].transpose(
                    0, 1
                )  # [num_tokens, H, D] -> [H, num_tokens, D]

                paged_kv_cache[0, page_idx, 1, :, :num_tokens_in_page, :] = v_ragged[
                    token_offset + start_token : token_offset + end_token
                ].transpose(
                    0, 1
                )  # [num_tokens, H, D] -> [H, num_tokens, D]

                page_idx += 1

            token_offset += seq_len
        kv_cache = KVCache()
        kv_cache.kv_cache_base = paged_kv_cache[0]
        return kv_cache

    def _test_prefill_correctness(
        self,
        batch_size: int,
        sequence_lengths: List[int],
        head_num: int = 32,
        head_num_kv: int = 8,
        size_per_head: int = 128,
        page_size: int = 64,
    ):
        """Test prefill correctness by comparing with flashinfer reference implementation"""

        config = self._create_config(
            head_num=head_num,
            head_num_kv=head_num_kv,
            size_per_head=size_per_head,
            seq_size_per_block=page_size,
        )

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

        # Create paged KV cache
        paged_kv_cache = self._create_paged_kv_cache(
            k, v, sequence_lengths, page_size, config.head_num_kv, config.size_per_head
        )

        # Forward pass through PyFlashinferPrefillPagedAttnOp
        output = attn_op.forward(q, paged_kv_cache, params)  # Use layer 0

        # Compute reference outputs using flashinfer's reference
        ref_output = compute_flashinfer_prefill_reference(
            q, k, v, attn_inputs.cu_seqlens, causal=True
        )

        # Compare outputs
        print(
            f"Testing batch_size={batch_size}, seq_lens={sequence_lengths}, "
            f"head_num={head_num}, kv_head_num={head_num_kv}, "
            f"size_per_head={size_per_head}, page_size={page_size}"
        )

        # Assert closeness (with relaxed tolerance for FP16)
        try:
            compare_tensors(
                output,
                ref_output,
                rtol=1e-2,
                atol=5e-3,
                name="Prefill output",
            )
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

        # Create paged KV cache
        sequence_lengths = [p + i for p, i in zip(prefix_lengths, input_lengths)]
        paged_kv_cache = self._create_paged_kv_cache(
            k, v, sequence_lengths, page_size, config.head_num_kv, config.size_per_head
        )

        # Forward pass through PyFlashinferPrefillPagedAttnOp
        print("\nRunning FlashInfer forward pass...")
        output = attn_op.forward(q, paged_kv_cache, params)

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
        q_full[prefix_len:] = q  # 把真实的 Q 放在后面

        print(f"  Q_full shape: {q_full.shape} (padded)")
        print(f"  K shape: {k.shape}")
        print(f"  Prefix: {prefix_len}, Input: {input_len}")

        # 用 FlashInfer 计算完整的 attention
        ref_output_full = single_prefill_with_kv_cache(
            q_full, k, v, causal=True, kv_layout="NHD"
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
            compare_tensors(
                output,
                ref_output,
                atol=1e-2,
                rtol=1e-2,
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


if __name__ == "__main__":
    unittest.main()
