import logging
import unittest
from typing import List

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
    PyFlashinferPrefillAttnOp,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.test.attention_ref import (
    compute_flashinfer_prefill_reference,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.test.base_attention_test import (
    BaseAttentionTest,
    compare_tensors,
)
from rtp_llm.ops.compute_ops import KVCache

logging.basicConfig(level=logging.INFO, format="%(message)s")


class TestPyFlashinferPrefillAttnOp(BaseAttentionTest):
    """Test suite for PyFlashinferPrefillAttnOp with correctness verification"""

    def setUp(self):
        """Set up test fixtures"""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available - this test requires CUDA")

        # Call parent setUp for common initialization
        super().setUp()

    def _create_kv_cache(
        self,
        total_blocks: int,
        seq_size_per_block: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
    ):
        """Helper to create empty KV cache for prefill"""
        kv_cache = KVCache()

        # Create combined KV cache with shape [total_blocks, 2, num_kv_heads, seq_size_per_block, head_dim]
        # Initialize with zeros since we'll write to it during prefill
        kv_cache_combined = torch.zeros(
            total_blocks,
            2,  # K and V
            num_kv_heads,
            seq_size_per_block,
            head_dim,
            dtype=dtype,
            device=self.device,
        )

        kv_cache.kv_cache_base = kv_cache_combined

        # Extract separate K and V for reference computation
        k_cache = kv_cache_combined[:, 0, :, :, :]
        v_cache = kv_cache_combined[:, 1, :, :, :]

        return kv_cache, k_cache, v_cache

    def _test_prefill_correctness(
        self,
        batch_size: int,
        sequence_lengths: List[int],
        head_num: int = 32,
        head_num_kv: int = 8,
        size_per_head: int = 128,
        seq_size_per_block: int = 64,
    ):
        """Test prefill correctness by comparing with flashinfer reference implementation"""

        config = self._create_config(
            head_num=head_num,
            head_num_kv=head_num_kv,
            size_per_head=size_per_head,
            seq_size_per_block=seq_size_per_block,
        )

        attn_inputs = self._create_prefill_attention_inputs(
            batch_size, sequence_lengths, config.seq_size_per_block
        )

        # Create PyFlashinferPrefillAttnOp instance
        attn_op = PyFlashinferPrefillAttnOp(config.attn_configs)

        # Check support
        if not attn_op.support(attn_inputs):
            raise RuntimeError(
                "PyFlashinferPrefillAttnOp does not support this configuration"
            )

        # Prepare params
        params = attn_op.prepare(attn_inputs)

        # Create QKV input in the format expected by PyFlashinferPrefillAttnOp
        # Input shape: [total_tokens, hidden_size_q + hidden_size_k + hidden_size_v]
        # where hidden_size_q = head_dim * head_num, hidden_size_k = hidden_size_v = head_dim * kv_head_num
        total_tokens = sum(sequence_lengths)

        hidden_size_q = config.size_per_head * config.head_num
        hidden_size_k = config.size_per_head * config.head_num_kv
        hidden_size_v = config.size_per_head * config.head_num_kv

        # Create QKV tensor in flattened format [total_tokens, hidden_size_q + hidden_size_k + hidden_size_v]
        qkv = torch.randn(
            total_tokens,
            hidden_size_q + hidden_size_k + hidden_size_v,
            dtype=torch.float16,
            device=self.device,
        )

        # Extract Q, K, V for reference computation
        # Split and reshape to [total_tokens, num_heads, head_dim]
        q_flat, k_flat, v_flat = torch.split(
            qkv,
            [hidden_size_q, hidden_size_k, hidden_size_v],
            dim=-1,
        )
        q = q_flat.reshape(total_tokens, config.head_num, config.size_per_head)
        k = k_flat.reshape(total_tokens, config.head_num_kv, config.size_per_head)
        v = v_flat.reshape(total_tokens, config.head_num_kv, config.size_per_head)

        # Create KV cache
        total_blocks = self._calculate_total_blocks(
            sequence_lengths, config.seq_size_per_block
        )
        kv_cache, _, _ = self._create_kv_cache(
            total_blocks,
            config.seq_size_per_block,
            config.head_num_kv,
            config.size_per_head,
            dtype=torch.float16,
        )

        # Forward pass through PyFlashinferPrefillAttnOp
        output = attn_op.forward(qkv, kv_cache, params)

        # Compute reference outputs using flashinfer's single_prefill_with_kv_cache
        ref_output = compute_flashinfer_prefill_reference(
            q, k, v, attn_inputs.cu_seqlens, causal=True
        )

        # Compare outputs
        compare_tensors(
            output,
            ref_output,
            rtol=1e-2,
            atol=1e-2,
            name=f"Prefill output (batch={batch_size}, seq_lens={sequence_lengths})",
        )

        logging.info(
            f"✓ Test passed: batch_size={batch_size}, sequence_lengths={sequence_lengths}"
        )

    def test_single_batch_prefill(self):
        """Test prefill for a single batch"""
        logging.info("\n=== Testing single batch prefill ===")
        for head_dim in [128, 256]:
            logging.info(f"\n--- Testing head_dim={head_dim} ---")
            self._test_prefill_correctness(
                batch_size=1,
                sequence_lengths=[128],
                size_per_head=head_dim,
            )

    def test_multi_batch_prefill(self):
        """Test prefill for multiple batches with varying sequence lengths"""
        logging.info("\n=== Testing multi-batch prefill ===")
        for head_dim in [128, 256]:
            logging.info(f"\n--- Testing head_dim={head_dim} ---")
            self._test_prefill_correctness(
                batch_size=4,
                sequence_lengths=[64, 128, 256, 512],
                size_per_head=head_dim,
            )

    def test_different_block_sizes(self):
        """Test with different block sizes"""
        logging.info("\n=== Testing different block sizes ===")
        for head_dim in [128, 256]:
            for block_size in [16, 32, 64, 128]:
                logging.info(
                    f"\n--- Testing head_dim={head_dim}, block_size={block_size} ---"
                )
                self._test_prefill_correctness(
                    batch_size=2,
                    sequence_lengths=[100, 200],
                    size_per_head=head_dim,
                    seq_size_per_block=block_size,
                )

    def test_different_head_configurations(self):
        """Test with different head configurations (GQA)"""
        logging.info("\n=== Testing different head configurations ===")
        test_cases = [
            (32, 32, "MHA"),  # MHA: head_num == head_num_kv
            (32, 8, "GQA"),  # GQA: head_num > head_num_kv (group_size=4)
            (32, 4, "GQA-4"),  # GQA with group_size=8
        ]

        for head_dim in [128, 256]:
            for head_num, head_num_kv, name in test_cases:
                logging.info(
                    f"\n--- Testing {name}: head_num={head_num}, head_num_kv={head_num_kv}, head_dim={head_dim} ---"
                )
                self._test_prefill_correctness(
                    batch_size=2,
                    sequence_lengths=[100, 200],
                    head_num=head_num,
                    head_num_kv=head_num_kv,
                    size_per_head=head_dim,
                )

    def test_edge_case_sequence_lengths(self):
        """Test edge cases with sequence lengths"""
        logging.info("\n=== Testing edge case sequence lengths ===")

        for head_dim in [128, 256]:
            # Sequence length exactly equal to block size
            logging.info(
                f"\n--- Testing seq_len == block_size, head_dim={head_dim} ---"
            )
            self._test_prefill_correctness(
                batch_size=1,
                sequence_lengths=[64],
                size_per_head=head_dim,
                seq_size_per_block=64,
            )

            # Sequence length slightly more than block size
            logging.info(f"\n--- Testing seq_len > block_size, head_dim={head_dim} ---")
            self._test_prefill_correctness(
                batch_size=1,
                sequence_lengths=[65],
                size_per_head=head_dim,
                seq_size_per_block=64,
            )

            # Very short sequences
            logging.info(f"\n--- Testing short sequences, head_dim={head_dim} ---")
            self._test_prefill_correctness(
                batch_size=2,
                sequence_lengths=[10, 20],
                size_per_head=head_dim,
                seq_size_per_block=64,
            )

    def test_variable_sequence_lengths(self):
        """Test prefill with highly variable sequence lengths

        PyFlashinferPrefillAttnOp uses BatchPrefillWithRaggedKVCacheWrapper which
        handles ragged tensors efficiently. This test verifies it works correctly
        with sequences of very different lengths.

        Note: This implementation uses ragged tensor format (via cu_seqlens), not
        padded format. Padding would be wasteful for highly variable lengths.
        """
        logging.info("\n=== Testing variable sequence lengths (ragged format) ===")

        for head_dim in [128, 256]:
            logging.info(f"\n--- Testing varied lengths, head_dim={head_dim} ---")

            # Test with very different sequence lengths
            self._test_prefill_correctness(
                batch_size=4,
                sequence_lengths=[32, 96, 200, 512],  # Highly variable
                size_per_head=head_dim,
            )

            # Test with extreme variation
            logging.info(f"\n--- Testing extreme variation, head_dim={head_dim} ---")
            self._test_prefill_correctness(
                batch_size=3,
                sequence_lengths=[16, 128, 1024],  # 64x difference
                size_per_head=head_dim,
            )


if __name__ == "__main__":
    unittest.main()
