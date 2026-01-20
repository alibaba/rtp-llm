import logging
import math
import unittest
from typing import List, Tuple

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
    PyFlashinferPrefillPagedAttnOp,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.test.attention_ref import (
    compute_flashinfer_prefill_reference,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.test.base_attention_test import (
    compare_tensors,
    set_seed,
)
from rtp_llm.ops import AttentionConfigs
from rtp_llm.ops.compute_ops import KVCache, PyAttentionInputs

logging.basicConfig(level=logging.INFO, format="%(message)s")


class TestPyFlashinferPrefillPagedAttnOp(unittest.TestCase):
    """Test suite for PyFlashinferPrefillPagedAttnOp with paged KV cache"""

    def setUp(self):
        """Set up test fixtures"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        self.device = torch.device("cuda")
        set_seed(42)

    def _create_config(
        self,
        head_num: int = 32,
        head_num_kv: int = 8,
        size_per_head: int = 128,
        seq_size_per_block: int = 64,
        data_type: str = "fp16",
    ):
        """Helper to create a test config

        Returns a simple namespace with attention configuration.
        No TP-related config needed for unit tests.
        """
        attn_configs = AttentionConfigs()
        attn_configs.head_num = head_num
        attn_configs.kv_head_num = head_num_kv
        attn_configs.size_per_head = size_per_head
        attn_configs.tokens_per_block = seq_size_per_block
        attn_configs.use_mla = False
        # Set dtype based on data_type parameter
        dtype_map = {
            "fp16": torch.float16,
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
        }
        attn_configs.dtype = dtype_map.get(data_type, torch.float16)

        # Return a simple namespace instead of TestConfig
        from types import SimpleNamespace

        return SimpleNamespace(
            attn_configs=attn_configs,
            head_num=head_num,
            head_num_kv=head_num_kv,
            size_per_head=size_per_head,
            seq_size_per_block=seq_size_per_block,
        )

    def _create_attention_inputs(
        self,
        batch_size: int,
        sequence_lengths: List[int],
        seq_size_per_block: int,
    ) -> PyAttentionInputs:
        """Helper to create PyAttentionInputs for prefill"""
        attn_inputs = PyAttentionInputs()

        # Prefill mode
        attn_inputs.is_prefill = True

        # input_lengths is the length of each sequence in the batch
        attn_inputs.input_lengths = torch.tensor(
            sequence_lengths, dtype=torch.int32, device="cpu"
        ).pin_memory()

        # sequence_lengths for prefill is same as input_lengths
        attn_inputs.sequence_lengths = torch.tensor(
            sequence_lengths, dtype=torch.int32, device="cpu"
        ).pin_memory()

        # prefix_lengths is all zeros for pure prefill (no prefix caching)
        attn_inputs.prefix_lengths = torch.zeros(
            batch_size, dtype=torch.int32, device="cpu"
        )

        # Create KV cache block IDs
        max_blocks = max(
            [math.ceil(seq_len / seq_size_per_block) for seq_len in sequence_lengths]
        )
        kv_cache_block_id = torch.zeros(
            [batch_size, max_blocks], dtype=torch.int32, device="cpu"
        )

        # Fill block IDs sequentially for each batch
        block_offset = 0
        for i, seq_len in enumerate(sequence_lengths):
            num_blocks = math.ceil(seq_len / seq_size_per_block)
            kv_cache_block_id[i, :num_blocks] = torch.arange(
                block_offset, block_offset + num_blocks, dtype=torch.int32
            )
            block_offset += num_blocks

        attn_inputs.kv_cache_block_id_host = kv_cache_block_id
        attn_inputs.kv_cache_block_id_device = kv_cache_block_id.to(self.device)

        # Create cu_seqlens (cumulative sequence lengths) for ragged tensor
        cu_seqlens = [0]
        for seq_len in sequence_lengths:
            cu_seqlens.append(cu_seqlens[-1] + seq_len)
        attn_inputs.cu_seqlens = torch.tensor(
            cu_seqlens, dtype=torch.int32, device=self.device
        )

        return attn_inputs

    def _create_paged_kv_cache_params(
        self,
        sequence_lengths: List[int],
        page_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create paged KV cache parameters for FlashInfer

        Args:
            sequence_lengths: List of sequence lengths
            page_size: Page size

        Returns:
            paged_kv_indptr: [batch_size + 1]
            paged_kv_indices: [total_pages]
            paged_kv_last_page_len: [batch_size]
        """
        batch_size = len(sequence_lengths)

        paged_kv_indptr = [0]
        paged_kv_indices = []
        paged_kv_last_page_len = []

        page_idx = 0
        for seq_len in sequence_lengths:
            num_pages = math.ceil(seq_len / page_size)
            last_page_len = (
                seq_len % page_size if seq_len % page_size != 0 else page_size
            )

            # Add page indices for this sequence
            for _ in range(num_pages):
                paged_kv_indices.append(page_idx)
                page_idx += 1

            paged_kv_indptr.append(paged_kv_indptr[-1] + num_pages)
            paged_kv_last_page_len.append(last_page_len)

        return (
            torch.tensor(paged_kv_indptr, dtype=torch.int32, device=self.device),
            torch.tensor(paged_kv_indices, dtype=torch.int32, device=self.device),
            torch.tensor(paged_kv_last_page_len, dtype=torch.int32, device=self.device),
        )

    def _create_paged_kv_cache(
        self,
        k_ragged: torch.Tensor,
        v_ragged: torch.Tensor,
        sequence_lengths: List[int],
        page_size: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> torch.Tensor:
        """
        Convert ragged K, V to paged KV cache format

        Args:
            k_ragged: [total_tokens, num_kv_heads, head_dim]
            v_ragged: [total_tokens, num_kv_heads, head_dim]
            sequence_lengths: List of sequence lengths
            page_size: Page size
            num_kv_heads: Number of KV heads
            head_dim: Head dimension

        Returns:
            paged_kv_cache: [num_layers, num_pages, 2, page_size, num_kv_heads, head_dim]
        """
        total_pages = sum(
            (seq_len + page_size - 1) // page_size for seq_len in sequence_lengths
        )
        num_layers = 1  # Single layer for testing

        # Allocate paged KV cache
        paged_kv_cache = torch.zeros(
            num_layers,
            total_pages,
            2,
            page_size,
            num_kv_heads,
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

                # Copy K, V to page (layer 0)
                paged_kv_cache[0, page_idx, 0, :num_tokens_in_page] = k_ragged[
                    token_offset + start_token : token_offset + end_token
                ]
                paged_kv_cache[0, page_idx, 1, :num_tokens_in_page] = v_ragged[
                    token_offset + start_token : token_offset + end_token
                ]

                page_idx += 1

            token_offset += seq_len

        return paged_kv_cache

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

        attn_inputs = self._create_attention_inputs(
            batch_size, sequence_lengths, config.seq_size_per_block
        )

        # Create PyFlashinferPrefillPagedAttnOp instance
        attn_op = PyFlashinferPrefillPagedAttnOp(
            config.attn_configs,
            page_size=page_size,
        )

        # Check support
        if not attn_op.support(attn_inputs):
            self.skipTest(
                "PyFlashinferPrefillPagedAttnOp does not support this configuration"
            )

        # Create paged KV cache parameters
        paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len = (
            self._create_paged_kv_cache_params(sequence_lengths, page_size)
        )

        # Prepare params
        params = attn_op.prepare(
            attn_inputs,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
        )

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
        output = attn_op.forward(q, paged_kv_cache[0], params)  # Use layer 0

        # Compute reference outputs using flashinfer's reference
        ref_output = compute_flashinfer_prefill_reference(
            q, k, v, attn_inputs.cu_seqlens, causal=True
        )

        # Compare outputs
        logging.info(
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
            logging.info("✓ Test passed")
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
