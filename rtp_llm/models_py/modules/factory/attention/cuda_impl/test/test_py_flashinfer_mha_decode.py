import logging
import math
import os
import sys
import unittest
from typing import List
from unittest.mock import patch

import torch
from attention_ref import compute_flashinfer_decode_reference
from base_attention_test import BaseAttentionTest, compare_tensors

from rtp_llm.models_py.model_desc.minimax_m3 import _target_verify_impl_class
from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
    PyFlashinferDecodeAttnOp,
)
from rtp_llm.ops.compute_ops import PyAttentionInputs, fill_mla_params, get_typemeta

logging.basicConfig(level=logging.INFO, format="%(message)s")


class TestPyFlashinferDecodeAttnOp(BaseAttentionTest):
    """Test suite for PyFlashinferDecodeAttnOp with correctness verification"""

    def _create_attention_inputs(
        self,
        batch_size: int,
        sequence_lengths: List[int],
        seq_size_per_block: int,
        dtype: torch.dtype = torch.float16,
    ) -> PyAttentionInputs:
        """Helper to create PyAttentionInputs for decode"""
        attn_inputs = self._create_attention_inputs_base(
            batch_size=batch_size,
            sequence_lengths=sequence_lengths,
            seq_size_per_block=seq_size_per_block,
        )
        attn_inputs.dtype = get_typemeta(torch.zeros([1], dtype=dtype))
        return attn_inputs

    def _check_params(
        self,
        attn_inputs: PyAttentionInputs,
        batch_size: int,
        sequence_lengths: List[int],
        seq_size_per_block: int,
    ):
        """Check that the prepared parameters match expected values

        This validates that fill_mla_params correctly generates:
        - decode_page_indptr: cumulative count of pages per sequence
        - page_indice: sequential block IDs for all sequences
        - paged_kv_last_page_len: last page length for each sequence
        """
        # Call fill_mla_params to get the actual params
        mla_params = fill_mla_params(
            attn_inputs.prefix_lengths,
            attn_inputs.sequence_lengths,
            attn_inputs.input_lengths,
            attn_inputs.kv_cache_block_id_host,
            seq_size_per_block,
        )

        # Calculate expected values
        expected_page_indptr = [0]
        expected_page_indices = []
        expected_last_page_len = []

        block_offset = 0
        for seq_len in sequence_lengths:
            num_blocks = math.ceil(seq_len / seq_size_per_block)
            expected_page_indptr.append(expected_page_indptr[-1] + num_blocks)

            # Add all block indices for this sequence
            for j in range(num_blocks):
                expected_page_indices.append(block_offset + j)

            # Last page length is the remainder, or full block size if perfectly aligned
            expected_last_page_len.append(
                seq_len % seq_size_per_block or seq_size_per_block
            )
            block_offset += num_blocks

        # Get actual values from mla_params
        actual_page_indptr = mla_params.decode_page_indptr_h.tolist()
        actual_page_indices = mla_params.page_indice_h.tolist()[
            : len(expected_page_indices)
        ]
        actual_last_page_len = mla_params.paged_kv_last_page_len_h.tolist()

        # Verify each parameter
        if actual_page_indptr != expected_page_indptr:
            error_msg = f"page_indptr mismatch:\n  Expected: {expected_page_indptr}\n  Got: {actual_page_indptr}"
            logging.error(error_msg)
            raise AssertionError(error_msg)

        if actual_page_indices != expected_page_indices:
            error_msg = f"page_indices mismatch:\n  Expected: {expected_page_indices}\n  Got: {actual_page_indices}"
            logging.error(error_msg)
            raise AssertionError(error_msg)

        if actual_last_page_len != expected_last_page_len:
            error_msg = f"last_page_len mismatch:\n  Expected: {expected_last_page_len}\n  Got: {actual_last_page_len}"
            logging.error(error_msg)
            raise AssertionError(error_msg)

        # All checks passed
        logging.info(f"✓ fill_mla_params check passed:")
        logging.info(f"  decode_page_indptr: {actual_page_indptr}")
        logging.info(f"  page_indice: {actual_page_indices}")
        logging.info(f"  paged_kv_last_page_len: {actual_last_page_len}")

    def _test_decode_correctness(
        self,
        batch_size: int,
        sequence_lengths: List[int],
        head_num: int = 32,
        head_num_kv: int = 8,
        size_per_head: int = 128,
        seq_size_per_block: int = 64,
    ):
        """Test decode correctness by comparing with flashinfer reference implementation"""

        config = self._create_config(
            head_num=head_num,
            head_num_kv=head_num_kv,
            size_per_head=size_per_head,
            seq_size_per_block=seq_size_per_block,
        )

        attn_inputs = self._create_attention_inputs(
            batch_size, sequence_lengths, config.seq_size_per_block
        )

        # Create PyFlashinferDecodeAttnOp instance
        attn_op = PyFlashinferDecodeAttnOp(config.attn_configs)

        # Check that prepared parameters match expected values BEFORE calling prepare
        # This validates fill_mla_params works correctly with the given inputs
        self._check_params(
            attn_inputs, batch_size, sequence_lengths, config.seq_size_per_block
        )

        # Use the standard prepare method which calls fill_mla_params
        # This will now work correctly because:
        # 1. prefix_lengths is empty tensor -> triggers decode branch
        # 2. sequence_lengths are passed as indices (length - 1)
        params = attn_op.prepare(attn_inputs)

        # Create query input [batch_size, head_num, head_dim]
        local_head_num = config.head_num // config.tp_size
        local_kv_head_num = config.head_num_kv // config.tp_size
        q = self._create_query_tensor(batch_size, local_head_num, config.size_per_head)

        # Create KV cache
        total_blocks = self._calculate_total_blocks(
            sequence_lengths, config.seq_size_per_block
        )
        kv_cache, k_cache, v_cache = self._create_kv_cache(
            total_blocks,
            config.seq_size_per_block,
            local_kv_head_num,
            config.size_per_head,
            dtype=torch.float16,
        )

        # Forward pass through PyFlashinferDecodeAttnOp
        output = attn_op.forward(q, kv_cache, params)

        # Generate block_id_list from attn_inputs for reference computation
        block_id_list = self._generate_block_id_list(
            attn_inputs, sequence_lengths, config.seq_size_per_block
        )

        # Compute reference outputs using flashinfer's single_decode_with_kv_cache
        ref_output_stacked = compute_flashinfer_decode_reference(
            q,
            k_cache,
            v_cache,
            sequence_lengths,
            block_id_list,
            config.seq_size_per_block,
        )

        # Compare outputs
        compare_tensors(
            output,
            ref_output_stacked,
            rtol=1e-2,
            atol=1e-2,
            name=f"Decode output (batch={batch_size}, seq_lens={sequence_lengths})",
        )

        logging.info(
            f"✓ Test passed: batch_size={batch_size}, sequence_lengths={sequence_lengths}"
        )

    def test_single_batch_decode(self):
        """Test decode for a single batch"""
        logging.info("\n=== Testing single batch decode ===")
        for head_dim in [128, 256]:
            logging.info(f"\n--- Testing head_dim={head_dim} ---")
            self._test_decode_correctness(
                batch_size=1,
                sequence_lengths=[128],
                size_per_head=head_dim,
            )

    def test_multi_batch_decode(self):
        """Test decode for multiple batches with varying sequence lengths"""
        logging.info("\n=== Testing multi-batch decode ===")
        for head_dim in [128, 256]:
            logging.info(f"\n--- Testing head_dim={head_dim} ---")
            self._test_decode_correctness(
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
                self._test_decode_correctness(
                    batch_size=2,
                    sequence_lengths=[100, 200],
                    size_per_head=head_dim,
                    seq_size_per_block=block_size,
                )

    def test_different_head_configurations(self):
        """Test with different head configurations (GQA)"""
        logging.info("\n=== Testing different head configurations ===")
        test_cases = [
            (32, 32, "MHA"),  # MHA: head_num == head_num_kv (group_size=1)
            (32, 8, "GQA"),  # GQA: head_num > head_num_kv (group_size=4)
            (32, 4, "GQA-4"),  # GQA with group_size=8
        ]

        for head_dim in [128, 256]:
            for head_num, head_num_kv, name in test_cases:
                logging.info(
                    f"\n--- Testing {name}: head_num={head_num}, head_num_kv={head_num_kv}, head_dim={head_dim} ---"
                )
                self._test_decode_correctness(
                    batch_size=2,
                    sequence_lengths=[100, 200],
                    head_num=head_num,
                    head_num_kv=head_num_kv,
                    size_per_head=head_dim,
                )

    def test_cuda_graph_prepare_uses_minimum_kv_length(self):
        config = self._create_config(
            head_num=32,
            head_num_kv=8,
            size_per_head=128,
            seq_size_per_block=128,
        )
        sequence_lengths = [256, 512]
        attn_inputs = self._create_attention_inputs(
            batch_size=2,
            sequence_lengths=sequence_lengths,
            seq_size_per_block=config.seq_size_per_block,
        )
        attn_inputs.is_cuda_graph = True
        attn_inputs.sequence_lengths_plus_1_d = torch.tensor(
            sequence_lengths, dtype=torch.int32, device="cuda"
        )

        attn_op = PyFlashinferDecodeAttnOp(config.attn_configs)
        params = attn_op.prepare(attn_inputs)
        torch.cuda.synchronize()
        self.assertTrue(attn_op.decode_wrapper._use_cuda_graph)
        self.assertEqual(params.decode_page_indptr_d.cpu().tolist(), [0, 1, 2])
        self.assertEqual(params.paged_kv_last_page_len_d.cpu().tolist(), [1, 1])

        indptr_ptr = attn_op.decode_wrapper._paged_kv_indptr_buf.data_ptr()
        indices_ptr = attn_op.decode_wrapper._paged_kv_indices_buf.data_ptr()
        last_page_len_ptr = (
            attn_op.decode_wrapper._paged_kv_last_page_len_buf.data_ptr()
        )
        original_plan = attn_op.decode_wrapper.plan
        plan_calls = []

        def counted_plan(*args, **kwargs):
            plan_calls.append((args, kwargs))
            return original_plan(*args, **kwargs)

        attn_op.decode_wrapper.plan = counted_plan
        attn_op.prepare_for_cuda_graph_replay(attn_inputs)
        torch.cuda.synchronize()
        self.assertEqual(len(plan_calls), 1)
        self.assertEqual(params.kvlen_d.cpu().tolist(), sequence_lengths)
        self.assertEqual(
            attn_op.decode_wrapper._paged_kv_indptr_buf.data_ptr(), indptr_ptr
        )
        self.assertEqual(
            attn_op.decode_wrapper._paged_kv_indices_buf.data_ptr(), indices_ptr
        )
        self.assertEqual(
            attn_op.decode_wrapper._paged_kv_last_page_len_buf.data_ptr(),
            last_page_len_ptr,
        )

    def test_cuda_graph_long_kv_replan_matches_eager_reference(self):
        """A long-KV replay must use the runtime plan, not the 1-page capture plan."""
        config = self._create_config(
            head_num=32,
            head_num_kv=8,
            size_per_head=128,
            seq_size_per_block=128,
            data_type="bf16",
        )
        sequence_lengths = [638 * config.seq_size_per_block]
        attn_inputs = self._create_attention_inputs(
            batch_size=1,
            sequence_lengths=sequence_lengths,
            seq_size_per_block=config.seq_size_per_block,
            dtype=torch.bfloat16,
        )
        attn_inputs.is_cuda_graph = True
        attn_inputs.sequence_lengths_plus_1_d = torch.tensor(
            sequence_lengths, dtype=torch.int32, device=self.device
        )

        attn_op = PyFlashinferDecodeAttnOp(config.attn_configs)
        params = attn_op.prepare(attn_inputs)
        q = self._create_query_tensor(
            1, config.head_num, config.size_per_head, dtype=torch.bfloat16
        )
        total_blocks = self._calculate_total_blocks(
            sequence_lengths, config.seq_size_per_block
        )
        kv_cache, k_cache, v_cache = self._create_kv_cache(
            total_blocks,
            config.seq_size_per_block,
            config.head_num_kv,
            config.size_per_head,
            dtype=torch.bfloat16,
        )

        # Compile/JIT the captured one-page schedule before graph capture.
        attn_op.forward(q, kv_cache, params)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = attn_op.forward(q, kv_cache, params)

        # Reproduce the old contract: update only the aliased paged-KV tensors
        # while retaining the one-page capture plan.
        params.fill_decode_cuda_graph_params(
            attn_inputs.sequence_lengths_plus_1_d,
            attn_inputs.kv_cache_kernel_block_id_device,
            config.seq_size_per_block,
        )
        graph.replay()
        torch.cuda.synchronize()
        stale_plan_output = graph_output.clone()

        attn_op.prepare_for_cuda_graph_replay(attn_inputs)
        graph.replay()
        torch.cuda.synchronize()
        replay_output = graph_output.clone()

        block_ids = self._generate_block_id_list(
            attn_inputs, sequence_lengths, config.seq_size_per_block
        )
        ref_output = compute_flashinfer_decode_reference(
            q,
            k_cache,
            v_cache,
            sequence_lengths,
            block_ids,
            config.seq_size_per_block,
        )
        self.assertFalse(
            torch.allclose(stale_plan_output, ref_output, rtol=2e-2, atol=2e-2),
            "the one-page capture plan unexpectedly matched the 638-page reference",
        )
        compare_tensors(
            replay_output,
            ref_output,
            rtol=2e-2,
            atol=2e-2,
            name="CUDA graph long-KV replay",
        )

    def test_target_verify_replay_replans_after_metadata_refresh(self):
        """Target verify must replace its MAX_SEQ_LEN plan with the live-KV plan."""
        config = self._create_config(
            head_num=32,
            head_num_kv=8,
            size_per_head=128,
            seq_size_per_block=128,
            data_type="bf16",
        )
        verify_tokens = 4
        capture_prefix = 131072 - verify_tokens
        replay_prefix = 638 * config.seq_size_per_block - verify_tokens
        max_blocks = math.ceil(131072 / config.seq_size_per_block)
        block_table = torch.arange(
            max_blocks, dtype=torch.int32, device=self.device
        ).unsqueeze(0)

        def target_verify_inputs(prefix: int) -> PyAttentionInputs:
            attn_inputs = PyAttentionInputs()
            attn_inputs.is_prefill = True
            attn_inputs.is_target_verify = True
            attn_inputs.is_cuda_graph = True
            attn_inputs.total_tokens = verify_tokens
            attn_inputs.prefix_lengths = torch.tensor(
                [prefix], dtype=torch.int32, device=self.device
            )
            attn_inputs.input_lengths = torch.tensor(
                [verify_tokens], dtype=torch.int32, device=self.device
            )
            attn_inputs.sequence_lengths = torch.empty(
                0, dtype=torch.int32, device=self.device
            )
            attn_inputs.kv_cache_kernel_block_id_device = block_table
            attn_inputs.dtype = get_typemeta(
                torch.empty(1, dtype=torch.bfloat16, device=self.device)
            )
            return attn_inputs

        with patch.dict(
            os.environ,
            {"RTP_LLM_M3_TARGET_VERIFY_BACKEND": "flashinfer"},
        ):
            impl_class = _target_verify_impl_class()
            impl = object.__new__(impl_class)
            attn_op = impl._create_fmha_impl(config.attn_configs)

        capture_inputs = target_verify_inputs(capture_prefix)
        params = attn_op.prepare(capture_inputs)
        torch.cuda.synchronize()
        self.assertTrue(attn_op.decode_wrapper._use_cuda_graph)
        self.assertEqual(attn_op.decode_wrapper._fixed_batch_size, verify_tokens)

        metadata_ptrs = (
            attn_op.decode_wrapper._paged_kv_indptr_buf.data_ptr(),
            attn_op.decode_wrapper._paged_kv_indices_buf.data_ptr(),
            attn_op.decode_wrapper._paged_kv_last_page_len_buf.data_ptr(),
        )
        original_plan = attn_op.decode_wrapper.plan
        planned_indptr = []

        def counted_plan(*args, **kwargs):
            planned_indptr.append(args[0].cpu().tolist())
            return original_plan(*args, **kwargs)

        attn_op.decode_wrapper.plan = counted_plan
        attn_op.prepare_for_cuda_graph_replay(target_verify_inputs(replay_prefix))
        torch.cuda.synchronize()

        self.assertEqual(planned_indptr, [[0, 638, 1276, 1914, 2552]])
        self.assertEqual(
            params.decode_page_indptr_d.cpu().tolist(),
            [0, 638, 1276, 1914, 2552],
        )
        self.assertEqual(
            (
                attn_op.decode_wrapper._paged_kv_indptr_buf.data_ptr(),
                attn_op.decode_wrapper._paged_kv_indices_buf.data_ptr(),
                attn_op.decode_wrapper._paged_kv_last_page_len_buf.data_ptr(),
            ),
            metadata_ptrs,
        )

    def test_edge_case_sequence_lengths(self):
        """Test edge cases with sequence lengths"""
        logging.info("\n=== Testing edge case sequence lengths ===")

        for head_dim in [128, 256]:
            # Sequence length exactly equal to block size
            logging.info(
                f"\n--- Testing seq_len == block_size, head_dim={head_dim} ---"
            )
            self._test_decode_correctness(
                batch_size=1,
                sequence_lengths=[64],
                size_per_head=head_dim,
                seq_size_per_block=64,
            )

            # Sequence length slightly more than block size
            logging.info(f"\n--- Testing seq_len > block_size, head_dim={head_dim} ---")
            self._test_decode_correctness(
                batch_size=1,
                sequence_lengths=[65],
                size_per_head=head_dim,
                seq_size_per_block=64,
            )

            # Very short sequences
            logging.info(f"\n--- Testing short sequences, head_dim={head_dim} ---")
            self._test_decode_correctness(
                batch_size=2,
                sequence_lengths=[10, 20],
                size_per_head=head_dim,
                seq_size_per_block=64,
            )


if __name__ == "__main__":
    unittest.main()
