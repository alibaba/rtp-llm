import logging
import math
import sys
import unittest
from types import SimpleNamespace
from typing import List, NamedTuple, Optional
from unittest import mock

import torch
from attention_ref import compute_flashinfer_decode_reference
from base_attention_test import BaseAttentionTest, compare_tensors

from rtp_llm.models_py.modules.factory.attention.attn_factory import (
    _validate_dynamic_fp8_config,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
    PyFlashinferDecodeAttnOp,
    PyFlashinferDecodeImpl,
    _validate_dynamic_fp8_scale,
)
from rtp_llm.ops import KvCacheDataType, RopeConfig, RopeStyle
from rtp_llm.ops.compute_ops import (
    LayerKVCache,
    PyAttentionInputs,
    fill_mla_params,
    get_typemeta,
    rtp_llm_ops,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")


class PageMetadata(NamedTuple):
    page_indptr: List[int]
    page_indices: List[int]
    last_page_lens: List[int]


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
            attn_inputs.kv_cache_block_id,
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
        attn_op = PyFlashinferDecodeAttnOp(config.attn_configs, attn_inputs)

        # Check that prepared parameters match expected values BEFORE calling prepare
        self._check_params(
            attn_inputs, batch_size, sequence_lengths, config.seq_size_per_block
        )

        fmha_params = rtp_llm_ops.FlashInferMlaAttnParams()
        attn_op.set_params(fmha_params)
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
            dtype=self.cache_dtype(config.attn_configs),
        )

        # Forward pass through PyFlashinferDecodeAttnOp
        output = attn_op.forward(q, kv_cache, params)

        # Generate block_id_list from attn_inputs for reference computation
        block_id_list = self._generate_block_id_list(
            attn_inputs, sequence_lengths, config.seq_size_per_block
        )

        # Compute reference outputs using flashinfer's single_decode_with_kv_cache (with round-trip)
        ref_output_stacked = compute_flashinfer_decode_reference(
            q.to(attn_op.q_dtype).to(q.dtype),
            k_cache.to(q.dtype),
            v_cache.to(q.dtype),
            sequence_lengths,
            block_id_list,
            config.seq_size_per_block,
        )

        # Compare outputs
        self._assert_output_close(
            output,
            ref_output_stacked,
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

    def test_eager_cuda_metadata_plans_on_device_and_matches_reference(self):
        """Eager CUDA-core decode plans on device and matches the reference."""
        config = self._create_config(head_num=32, head_num_kv=32)
        sequence_lengths = [100, 200]
        batch_size = len(sequence_lengths)
        attn_inputs = self._create_attention_inputs(
            batch_size,
            sequence_lengths,
            config.seq_size_per_block,
        )
        attn_inputs.is_cuda_graph = False
        attn_inputs.sequence_lengths = attn_inputs.sequence_lengths.cuda()
        attn_inputs.input_lengths = attn_inputs.input_lengths.cuda()
        attn_inputs.prefix_lengths = attn_inputs.prefix_lengths.cuda()

        attn_op = PyFlashinferDecodeAttnOp(config.attn_configs, attn_inputs)
        self.assertFalse(attn_op.use_tensor_core)
        self.assertFalse(attn_op._uses_cuda_core_graph_plan_cache())
        fmha_params = rtp_llm_ops.FlashInferMlaAttnParams()
        attn_op.set_params(fmha_params)
        with mock.patch.object(
            attn_op.decode_wrapper,
            "plan",
            wraps=attn_op.decode_wrapper.plan,
        ) as plan_mock:
            params = attn_op.prepare(attn_inputs)
            self.assertEqual(plan_mock.call_count, 1)
            plan_call = plan_mock.call_args
            self.assertTrue(plan_call.args[0].is_cuda)
            self.assertTrue(plan_call.args[1].is_cuda)
            self.assertTrue(plan_call.args[2].is_cuda)
            self.assertNotIn("non_blocking", plan_call.kwargs)

        local_head_num = config.head_num // config.tp_size
        local_kv_head_num = config.head_num_kv // config.tp_size
        q = self._create_query_tensor(batch_size, local_head_num, config.size_per_head)
        total_blocks = self._calculate_total_blocks(
            sequence_lengths, config.seq_size_per_block
        )
        kv_cache, k_cache, v_cache = self._create_kv_cache(
            total_blocks,
            config.seq_size_per_block,
            local_kv_head_num,
            config.size_per_head,
            dtype=self.cache_dtype(config.attn_configs),
        )
        output = attn_op.forward(q, kv_cache, params)
        block_id_list = self._generate_block_id_list(
            attn_inputs, sequence_lengths, config.seq_size_per_block
        )
        reference = compute_flashinfer_decode_reference(
            q,
            k_cache,
            v_cache,
            sequence_lengths,
            block_id_list,
            config.seq_size_per_block,
        )
        self._assert_output_close(
            output,
            reference,
            name="Eager CUDA-metadata decode output",
        )


class TestPyFlashinferDecodeCudaGraph(BaseAttentionTest):
    """Test CUDA graph buffer management for PyFlashinferDecodeAttnOp.

    These tests exercise the Python prepare/replay boundary. End-to-end CUDA
    graph capture and replay remains covered by the model smoke test.
    """

    def _create_cuda_graph_inputs(
        self,
        batch_size: int,
        sequence_lengths: List[int],
        seq_size_per_block: int,
        dtype: torch.dtype = torch.float16,
        active_batch_size: Optional[int] = None,
        block_id_offset: int = 0,
        padding_block_id: int = 0,
    ) -> PyAttentionInputs:
        """Create graph inputs with the runner's logical padding lengths."""
        if active_batch_size is None:
            active_batch_size = batch_size
        if len(sequence_lengths) != active_batch_size:
            raise ValueError("sequence_lengths must match active_batch_size")
        if active_batch_size > batch_size:
            raise ValueError("active_batch_size must not exceed batch_size")

        # The runner pads a captured batch with decode slots whose previous
        # sequence length is zero; fill_params() therefore exposes one page
        # with last_page_len=1 for every padding slot.
        logical_sequence_lengths = sequence_lengths + [1] * (
            batch_size - active_batch_size
        )
        attn_inputs = PyAttentionInputs()
        attn_inputs.is_prefill = False
        attn_inputs.is_cuda_graph = True

        seq_t = torch.tensor(logical_sequence_lengths, dtype=torch.int32)
        attn_inputs.sequence_lengths = (seq_t - 1).pin_memory()
        attn_inputs.input_lengths = torch.ones(
            batch_size, dtype=torch.int32
        ).pin_memory()
        attn_inputs.prefix_lengths = torch.empty(0, dtype=torch.int32).pin_memory()

        kv_cache_block_id = self._create_kv_cache_block_ids(
            batch_size, logical_sequence_lengths, seq_size_per_block
        )
        for batch_idx, seq_len in enumerate(sequence_lengths):
            page_count = math.ceil(seq_len / seq_size_per_block)
            kv_cache_block_id[batch_idx, :page_count] += block_id_offset
        if active_batch_size < batch_size:
            kv_cache_block_id[active_batch_size:, 0] = padding_block_id
        attn_inputs.kv_cache_kernel_block_id = kv_cache_block_id
        attn_inputs.kv_cache_kernel_block_id_device = kv_cache_block_id.cuda()

        attn_inputs.cu_seqlens_device = torch.arange(
            0, batch_size + 1, dtype=torch.int32, device="cuda"
        )
        attn_inputs.dtype = get_typemeta(torch.zeros([1], dtype=dtype))
        return attn_inputs

    def test_set_params_invalidates_cuda_core_plan_snapshot(self):
        config = self._create_config(head_num=32, head_num_kv=32)
        inputs = self._create_cuda_graph_inputs(
            2,
            [64, 128],
            config.seq_size_per_block,
        )
        attn_op = PyFlashinferDecodeAttnOp(config.attn_configs, inputs)
        self.assertTrue(attn_op._uses_cuda_core_graph_plan_cache())
        attn_op._cuda_core_plan_page_indptr_h = torch.tensor(
            [0, 1, 3],
            dtype=torch.int32,
        )

        params = rtp_llm_ops.FlashInferMlaAttnParams()
        attn_op.set_params(params)

        self.assertIs(attn_op.fmha_params, params)
        self.assertIsNone(attn_op._cuda_core_plan_page_indptr_h)

    def test_set_params_rejects_replacing_graph_bound_buffers(self):
        config = self._create_config(head_num=32, head_num_kv=32)
        inputs = self._create_cuda_graph_inputs(
            2,
            [64, 128],
            config.seq_size_per_block,
        )
        attn_op = PyFlashinferDecodeAttnOp(config.attn_configs, inputs)
        attn_op.set_params(rtp_llm_ops.FlashInferMlaAttnParams())
        attn_op.prepare(inputs)

        with self.assertRaisesRegex(RuntimeError, "cannot be replaced"):
            attn_op.set_params(rtp_llm_ops.FlashInferMlaAttnParams())

    def _expected_page_metadata(
        self,
        active_sequence_lengths: List[int],
        seq_size_per_block: int,
        batch_size: Optional[int] = None,
        block_id_offset: int = 0,
        padding_block_id: int = 0,
    ) -> PageMetadata:
        active_batch_size = len(active_sequence_lengths)
        if batch_size is None:
            batch_size = active_batch_size
        if active_batch_size > batch_size:
            raise ValueError("active batch size must not exceed batch_size")
        sequence_lengths = active_sequence_lengths + [1] * (
            batch_size - active_batch_size
        )
        page_counts = [
            math.ceil(seq_len / seq_size_per_block) for seq_len in sequence_lengths
        ]
        page_indptr = [0]
        for page_count in page_counts:
            page_indptr.append(page_indptr[-1] + page_count)
        page_indices = []
        next_block_id = block_id_offset
        for batch_idx, page_count in enumerate(page_counts):
            if batch_idx < active_batch_size:
                page_indices.extend(range(next_block_id, next_block_id + page_count))
                next_block_id += page_count
            else:
                page_indices.extend([padding_block_id] * page_count)
        last_page_lens = [
            seq_len % seq_size_per_block or seq_size_per_block
            for seq_len in sequence_lengths
        ]
        return PageMetadata(page_indptr, page_indices, last_page_lens)

    def _assert_page_metadata(
        self,
        fmha_params,
        expected: PageMetadata,
    ) -> None:
        self.assertEqual(
            fmha_params.decode_page_indptr_h.tolist(), expected.page_indptr
        )
        self.assertEqual(fmha_params.page_indice_h.tolist(), expected.page_indices)
        self.assertEqual(
            fmha_params.paged_kv_last_page_len_h.tolist(),
            expected.last_page_lens,
        )
        torch.cuda.synchronize()
        self.assertEqual(
            fmha_params.decode_page_indptr_d.cpu().tolist(), expected.page_indptr
        )
        self.assertEqual(
            fmha_params.page_indice_d.cpu().tolist(), expected.page_indices
        )
        self.assertEqual(
            fmha_params.paged_kv_last_page_len_d.cpu().tolist(),
            expected.last_page_lens,
        )

    def _assert_graph_buffer_pointers(self, attn_op, fmha_params, pointers) -> None:
        current_pointers = (
            fmha_params.decode_page_indptr_d.data_ptr(),
            fmha_params.page_indice_d.data_ptr(),
            fmha_params.paged_kv_last_page_len_d.data_ptr(),
        )
        self.assertEqual(current_pointers, pointers)
        self.assertEqual(
            attn_op.decode_wrapper._paged_kv_indptr_buf.data_ptr(), pointers[0]
        )
        self.assertEqual(
            attn_op.decode_wrapper._paged_kv_indices_buf.data_ptr(), pointers[1]
        )
        self.assertEqual(
            attn_op.decode_wrapper._paged_kv_last_page_len_buf.data_ptr(),
            pointers[2],
        )

    def _assert_active_output_matches_reference(
        self,
        attn_op,
        fmha_params,
        attn_inputs: PyAttentionInputs,
        sequence_lengths: List[int],
        q: torch.Tensor,
        kv_cache,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        seq_size_per_block: int,
    ) -> None:
        active_batch_size = len(sequence_lengths)
        output = attn_op.forward(q, kv_cache, fmha_params)
        block_id_list = []
        for batch_idx, seq_len in enumerate(sequence_lengths):
            page_count = math.ceil(seq_len / seq_size_per_block)
            block_id_list.append(
                attn_inputs.kv_cache_kernel_block_id[batch_idx, :page_count].tolist()
            )
        reference = compute_flashinfer_decode_reference(
            q[:active_batch_size].to(attn_op.q_dtype).to(q.dtype),
            k_cache.to(q.dtype),
            v_cache.to(q.dtype),
            sequence_lengths,
            block_id_list,
            seq_size_per_block,
        )
        self._assert_output_close(
            output[:active_batch_size],
            reference,
            name=f"CUDA-core graph replay output ({sequence_lengths})",
        )

    def test_capture_sets_fixed_batch_size(self):
        """prepare() with is_cuda_graph=True must set _fixed_batch_size."""
        config = self._create_config()
        capture_bs = 4
        seq_lens = [64, 128, 256, 512]
        inputs = self._create_cuda_graph_inputs(
            capture_bs,
            seq_lens,
            config.seq_size_per_block,
        )

        attn_op = PyFlashinferDecodeAttnOp(config.attn_configs, inputs)
        self.assertTrue(attn_op.enable_cuda_graph)
        self.assertEqual(attn_op.decode_wrapper._fixed_batch_size, 0)

        fmha_params = rtp_llm_ops.FlashInferMlaAttnParams()
        attn_op.set_params(fmha_params)
        attn_op.prepare(inputs)

        self.assertEqual(attn_op.decode_wrapper._fixed_batch_size, capture_bs)
        self.assertTrue(attn_op.decode_wrapper._use_cuda_graph)
        logging.info("_fixed_batch_size correctly set after prepare()")

    def test_replay_refreshes_plan_metadata(self):
        """Tensor-core replay replans because its plan consumes KV lengths."""
        config = self._create_config()
        capture_bs = 8
        capture_seq_lens = [64, 128, 256, 512, 64, 128, 256, 512]

        capture_inputs = self._create_cuda_graph_inputs(
            capture_bs,
            capture_seq_lens,
            config.seq_size_per_block,
        )
        attn_op = PyFlashinferDecodeAttnOp(config.attn_configs, capture_inputs)
        self.assertTrue(attn_op.use_tensor_core)
        self.assertTrue(attn_op._tensor_core_cuda_graph_needs_replan())
        fmha_params = rtp_llm_ops.FlashInferMlaAttnParams()
        attn_op.set_params(fmha_params)
        with mock.patch.object(
            attn_op.decode_wrapper,
            "plan",
            wraps=attn_op.decode_wrapper.plan,
        ) as plan_mock:
            attn_op.prepare(capture_inputs)
            self.assertEqual(plan_mock.call_count, 1)
            capture_call = plan_mock.call_args
            self.assertFalse(capture_call.args[0].is_cuda)
            self.assertTrue(capture_call.args[1].is_cuda)
            self.assertEqual(
                capture_call.args[1].data_ptr(), fmha_params.page_indice_d.data_ptr()
            )
            self.assertFalse(capture_call.args[2].is_cuda)
            self.assertTrue(capture_call.kwargs["non_blocking"])
            self.assertTrue(hasattr(attn_op.decode_wrapper, "_qo_indptr_buf"))
            self.assertEqual(
                attn_op.decode_wrapper._qo_indptr_buf.numel(), capture_bs + 1
            )
            graph_buffer_pointers = (
                fmha_params.decode_page_indptr_d.data_ptr(),
                fmha_params.page_indice_d.data_ptr(),
                fmha_params.paged_kv_last_page_len_d.data_ptr(),
            )
            self._assert_graph_buffer_pointers(
                attn_op, fmha_params, graph_buffer_pointers
            )

            plan_mock.reset_mock()
            run_seq_lens = [100, 200, 300, 400, 64, 128, 256, 512]
            run_inputs = self._create_cuda_graph_inputs(
                capture_bs,
                run_seq_lens,
                config.seq_size_per_block,
            )
            attn_op.prepare_for_cuda_graph_replay(run_inputs)
            self.assertEqual(plan_mock.call_count, 1)
            expected = self._expected_page_metadata(
                run_seq_lens,
                config.seq_size_per_block,
                batch_size=capture_bs,
            )
            self._assert_page_metadata(fmha_params, expected)
            self._assert_graph_buffer_pointers(
                attn_op, fmha_params, graph_buffer_pointers
            )
            attn_op.prepare_for_cuda_graph_replay(run_inputs)
            self.assertEqual(plan_mock.call_count, 2)

        self.assertEqual(attn_op.decode_wrapper._fixed_batch_size, capture_bs)

    def test_tensor_core_replay_does_not_wait_for_previous_graph(self):
        config = self._create_config()
        inputs = self._create_cuda_graph_inputs(
            2, [100, 200], config.seq_size_per_block
        )
        attn_op = PyFlashinferDecodeAttnOp(config.attn_configs, inputs)
        self.assertTrue(attn_op.use_tensor_core)
        attn_op.set_params(rtp_llm_ops.FlashInferMlaAttnParams())
        attn_op.prepare(inputs)
        attn_op.prepare_for_cuda_graph_replay(inputs)
        torch.cuda.synchronize()

        # A blocking host-index copy in FlashInfer.plan used to wait for all
        # earlier work on this stream, preventing cross-step graph submission.
        previous_forward = torch.cuda.Event()
        torch.cuda._sleep(200_000_000)
        previous_forward.record()
        try:
            attn_op.prepare_for_cuda_graph_replay(inputs)
            self.assertFalse(previous_forward.query())
        finally:
            torch.cuda.synchronize()

    def test_dynamic_fp8_cuda_graph_plan_and_buffers_are_stable(self):
        config = self._create_config(
            head_num=32,
            head_num_kv=8,
            size_per_head=128,
            seq_size_per_block=64,
            data_type="bf16",
        )
        config.attn_configs.kv_cache_dtype = KvCacheDataType.FP8
        config.attn_configs.fp8_kv_cache_mode = 2

        for capture_bs in (1, 2, 4, 8, 16, 32):
            with self.subTest(capture_bs=capture_bs):
                capture_inputs = self._create_cuda_graph_inputs(
                    capture_bs,
                    [4809] * capture_bs,
                    config.seq_size_per_block,
                )
                attn_op = PyFlashinferDecodeAttnOp(config.attn_configs, capture_inputs)
                fmha_params = rtp_llm_ops.FlashInferMlaAttnParams()
                attn_op.set_params(fmha_params)
                attn_op.prepare(capture_inputs)
                capture_signature = tuple(attn_op.decode_wrapper._plan_info)
                capture_pointers = (
                    attn_op.g_workspace_buffer.data_ptr(),
                    attn_op.decode_wrapper._int_workspace_buffer.data_ptr(),
                    attn_op.decode_wrapper._qo_indptr_buf.data_ptr(),
                    fmha_params.decode_page_indptr_d.data_ptr(),
                    fmha_params.page_indice_d.data_ptr(),
                    fmha_params.paged_kv_last_page_len_d.data_ptr(),
                )

                replay_cases = [
                    ([1] * capture_bs, capture_bs, False),
                    (
                        [63 + index % 3 for index in range(capture_bs)],
                        capture_bs,
                        False,
                    ),
                    (
                        [64 + index % 3 for index in range(capture_bs)],
                        capture_bs,
                        True,
                    ),
                    (
                        [4807 + index % 3 for index in range(capture_bs)],
                        capture_bs,
                        False,
                    ),
                ]
                if capture_bs > 1:
                    active_batch_size = max(1, capture_bs // 2)
                    replay_cases.append(
                        ([65] * active_batch_size, active_batch_size, True)
                    )

                for sequence_lengths, active_batch_size, sparse_ids in replay_cases:
                    replay_inputs = self._create_cuda_graph_inputs(
                        capture_bs,
                        sequence_lengths,
                        config.seq_size_per_block,
                        active_batch_size=active_batch_size,
                        padding_block_id=7,
                    )
                    if sparse_ids:
                        sparse_table = (
                            replay_inputs.kv_cache_kernel_block_id * 7 + 11
                        ) % 97
                        if capture_bs > 1:
                            sparse_table[-1, 0] = sparse_table[0, 0]
                        replay_inputs.kv_cache_kernel_block_id = sparse_table
                        replay_inputs.kv_cache_kernel_block_id_device = (
                            sparse_table.cuda()
                        )
                    attn_op.prepare_for_cuda_graph_replay(replay_inputs)
                    self.assertEqual(
                        tuple(attn_op.decode_wrapper._plan_info),
                        capture_signature,
                        f"plan changed for batch={capture_bs}, lengths={sequence_lengths}",
                    )
                    self.assertEqual(
                        (
                            attn_op.g_workspace_buffer.data_ptr(),
                            attn_op.decode_wrapper._int_workspace_buffer.data_ptr(),
                            attn_op.decode_wrapper._qo_indptr_buf.data_ptr(),
                            fmha_params.decode_page_indptr_d.data_ptr(),
                            fmha_params.page_indice_d.data_ptr(),
                            fmha_params.paged_kv_last_page_len_d.data_ptr(),
                        ),
                        capture_pointers,
                    )

    def test_dynamic_fp8_cuda_graph_rejects_changed_plan(self):
        config = self._create_config(
            head_num=32,
            head_num_kv=8,
            size_per_head=128,
            seq_size_per_block=64,
            data_type="bf16",
        )
        config.attn_configs.kv_cache_dtype = KvCacheDataType.FP8
        config.attn_configs.fp8_kv_cache_mode = 2
        inputs = self._create_cuda_graph_inputs(
            2,
            [64, 65],
            config.seq_size_per_block,
        )
        attn_op = PyFlashinferDecodeAttnOp(config.attn_configs, inputs)
        attn_op.set_params(rtp_llm_ops.FlashInferMlaAttnParams())
        attn_op.prepare(inputs)

        changed_plan = list(attn_op.decode_wrapper._plan_info)
        changed_plan[0] += 1
        attn_op.decode_wrapper._plan_info = changed_plan
        with self.assertRaisesRegex(RuntimeError, "plan or buffer addresses changed"):
            attn_op._validate_dynamic_fp8_cuda_graph_state()

    def test_dynamic_fp8_direct_decode_real_cuda_graph_replay(self):
        config = self._create_config(
            head_num=32,
            head_num_kv=8,
            size_per_head=128,
            seq_size_per_block=64,
            data_type="bf16",
        )
        config.attn_configs.kv_cache_dtype = KvCacheDataType.FP8
        config.attn_configs.fp8_kv_cache_mode = 2
        capture_bs = 4
        capture_lengths = [129, 130, 131, 132]
        capture_inputs = self._create_cuda_graph_inputs(
            capture_bs,
            capture_lengths,
            config.seq_size_per_block,
        )
        attn_op = PyFlashinferDecodeAttnOp(config.attn_configs, capture_inputs)
        fmha_params = rtp_llm_ops.FlashInferMlaAttnParams()
        attn_op.set_params(fmha_params)
        attn_op.prepare(capture_inputs)

        page_count = 64
        local_head_num = config.head_num // config.tp_size
        local_kv_head_num = config.head_num_kv // config.tp_size
        payload = torch.randn(
            page_count,
            2,
            local_kv_head_num,
            config.seq_size_per_block,
            config.size_per_head,
            device="cuda",
            dtype=torch.bfloat16,
        ).to(torch.float8_e4m3fn)
        scales = (
            torch.rand(
                page_count,
                2,
                local_kv_head_num,
                config.seq_size_per_block,
                device="cuda",
                dtype=torch.float32,
            )
            .mul_(0.02)
            .add_(0.01)
        )
        kv_cache = LayerKVCache()
        kv_cache.kv_cache_base = payload
        kv_cache.kv_scale_base = scales.flatten(1)
        q = self._create_query_tensor(
            capture_bs, local_head_num, config.size_per_head
        ).to(config.attn_configs.dtype)

        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            graph_output = attn_op.forward(q, kv_cache, fmha_params)
        torch.cuda.synchronize()

        graph_pointers = (
            attn_op.g_workspace_buffer.data_ptr(),
            attn_op.decode_wrapper._int_workspace_buffer.data_ptr(),
            attn_op.decode_wrapper._qo_indptr_buf.data_ptr(),
            fmha_params.decode_page_indptr_d.data_ptr(),
            fmha_params.page_indice_d.data_ptr(),
            fmha_params.paged_kv_last_page_len_d.data_ptr(),
            q.data_ptr(),
            payload.data_ptr(),
            kv_cache.kv_scale_base.data_ptr(),
            graph_output.data_ptr(),
        )
        capture_signature = tuple(attn_op.decode_wrapper._plan_info)
        replay_cases = (
            ([63, 64, 65, 129], 0),
            ([65, 66, 127, 130], 16),
        )
        scale_view = kv_cache.kv_scale_base.view(
            page_count,
            2,
            local_kv_head_num,
            config.seq_size_per_block,
        )
        restored_k = (payload[:, 0].float() * scale_view[:, 0].unsqueeze(-1)).to(
            q.dtype
        )
        restored_v = (payload[:, 1].float() * scale_view[:, 1].unsqueeze(-1)).to(
            q.dtype
        )

        for sequence_lengths, block_id_offset in replay_cases:
            replay_inputs = self._create_cuda_graph_inputs(
                capture_bs,
                sequence_lengths,
                config.seq_size_per_block,
                block_id_offset=block_id_offset,
            )
            replay_q = torch.randn_like(q)
            q.copy_(replay_q)
            attn_op.prepare_for_cuda_graph_replay(replay_inputs)
            graph.replay()
            torch.cuda.synchronize()

            block_id_list = [
                replay_inputs.kv_cache_kernel_block_id[
                    batch_idx,
                    : math.ceil(sequence_length / config.seq_size_per_block),
                ].tolist()
                for batch_idx, sequence_length in enumerate(sequence_lengths)
            ]
            reference = compute_flashinfer_decode_reference(
                replay_q,
                restored_k,
                restored_v,
                sequence_lengths,
                block_id_list,
                config.seq_size_per_block,
            )
            torch.testing.assert_close(
                graph_output,
                reference,
                rtol=0.06,
                atol=0.04,
            )
            self.assertEqual(
                tuple(attn_op.decode_wrapper._plan_info), capture_signature
            )
            self.assertEqual(
                (
                    attn_op.g_workspace_buffer.data_ptr(),
                    attn_op.decode_wrapper._int_workspace_buffer.data_ptr(),
                    attn_op.decode_wrapper._qo_indptr_buf.data_ptr(),
                    fmha_params.decode_page_indptr_d.data_ptr(),
                    fmha_params.page_indice_d.data_ptr(),
                    fmha_params.paged_kv_last_page_len_d.data_ptr(),
                    q.data_ptr(),
                    payload.data_ptr(),
                    kv_cache.kv_scale_base.data_ptr(),
                    graph_output.data_ptr(),
                ),
                graph_pointers,
            )

    def test_dynamic_fp8_impl_real_cuda_graph_replay(self):
        config = self._create_config(
            head_num=32,
            head_num_kv=8,
            size_per_head=128,
            seq_size_per_block=64,
            data_type="bf16",
        )
        config.attn_configs.kv_cache_dtype = KvCacheDataType.FP8
        config.attn_configs.fp8_kv_cache_mode = 2
        config.attn_configs.need_rope_kv_cache = False
        capture_bs = 2
        capture_inputs = self._create_cuda_graph_inputs(
            capture_bs,
            [130, 131],
            config.seq_size_per_block,
        )
        impl = PyFlashinferDecodeImpl(config.attn_configs, capture_inputs)
        local_head_num = config.head_num // config.tp_size
        local_kv_head_num = config.head_num_kv // config.tp_size
        qkv = torch.randn(
            capture_bs,
            (local_head_num + 2 * local_kv_head_num) * config.size_per_head,
            device="cuda",
            dtype=config.attn_configs.dtype,
        )

        def new_cache() -> LayerKVCache:
            cache = LayerKVCache()
            cache.kv_cache_base = torch.zeros(
                32,
                2,
                local_kv_head_num,
                config.seq_size_per_block,
                config.size_per_head,
                device="cuda",
                dtype=torch.float8_e4m3fn,
            )
            cache.kv_scale_base = torch.ones(
                32,
                2 * local_kv_head_num * config.seq_size_per_block,
                device="cuda",
                dtype=torch.float32,
            )
            return cache

        graph_cache = new_cache()
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            graph_output = impl.forward(qkv, graph_cache)
        torch.cuda.synchronize()

        for sequence_lengths, block_id_offset in (
            ([63, 65], 4),
            ([129, 130], 12),
        ):
            graph_cache.kv_cache_base.zero_()
            graph_cache.kv_scale_base.fill_(1)
            replay_qkv = torch.randn_like(qkv)
            qkv.copy_(replay_qkv)
            replay_inputs = self._create_cuda_graph_inputs(
                capture_bs,
                sequence_lengths,
                config.seq_size_per_block,
                block_id_offset=block_id_offset,
            )
            impl.prepare_cuda_graph(replay_inputs)
            graph.replay()
            torch.cuda.synchronize()

            eager_inputs = self._create_cuda_graph_inputs(
                capture_bs,
                sequence_lengths,
                config.seq_size_per_block,
                block_id_offset=block_id_offset,
            )
            eager_inputs.is_cuda_graph = False
            eager_impl = PyFlashinferDecodeImpl(config.attn_configs, eager_inputs)
            eager_cache = new_cache()
            eager_output = eager_impl.forward(replay_qkv.clone(), eager_cache)
            torch.cuda.synchronize()

            torch.testing.assert_close(
                graph_output,
                eager_output,
                rtol=0.06,
                atol=0.04,
            )
            torch.testing.assert_close(
                graph_cache.kv_cache_base,
                eager_cache.kv_cache_base,
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                graph_cache.kv_scale_base,
                eager_cache.kv_scale_base,
                rtol=0,
                atol=0,
            )

    def test_cuda_core_replay_replans_only_on_page_topology_change(self):
        """CUDA-core replay caches only topology and refreshes graph buffers."""
        config = self._create_config(head_num=32, head_num_kv=32)
        capture_bs = 4
        active_bs = 2
        capture_seq_lens = [64, 128, 256, 512]

        capture_inputs = self._create_cuda_graph_inputs(
            capture_bs,
            capture_seq_lens,
            config.seq_size_per_block,
        )
        # A CUDA-resident base field must not switch graph capture to the
        # device-only fill route: CUDA-core planning consumes host metadata.
        capture_inputs.input_lengths = capture_inputs.input_lengths.cuda()
        attn_op = PyFlashinferDecodeAttnOp(config.attn_configs, capture_inputs)
        self.assertFalse(attn_op.use_tensor_core)
        self.assertFalse(attn_op._tensor_core_cuda_graph_needs_replan())
        self.assertTrue(attn_op._uses_cuda_core_graph_plan_cache())
        fmha_params = rtp_llm_ops.FlashInferMlaAttnParams()
        attn_op.set_params(fmha_params)
        with mock.patch.object(
            attn_op.decode_wrapper,
            "plan",
            wraps=attn_op.decode_wrapper.plan,
        ) as plan_mock:
            attn_op.prepare(capture_inputs)
            graph_buffer_pointers = (
                fmha_params.decode_page_indptr_d.data_ptr(),
                fmha_params.page_indice_d.data_ptr(),
                fmha_params.paged_kv_last_page_len_d.data_ptr(),
            )
            with self.subTest(phase="capture host plan"):
                self.assertEqual(plan_mock.call_count, 1)
                capture_call = plan_mock.call_args
                self.assertFalse(capture_call.args[0].is_cuda)
                self.assertTrue(capture_call.args[1].is_cuda)
                self.assertFalse(capture_call.args[2].is_cuda)
                self.assertTrue(capture_call.kwargs["non_blocking"])
                self.assertTrue(attn_op.decode_wrapper._use_cuda_graph)
                self.assertEqual(attn_op.decode_wrapper._fixed_batch_size, capture_bs)
                self.assertFalse(hasattr(attn_op.decode_wrapper, "_qo_indptr_buf"))
                self._assert_graph_buffer_pointers(
                    attn_op, fmha_params, graph_buffer_pointers
                )

            plan_mock.reset_mock()
            changed_seq_lens = [100, 200]
            changed_inputs = self._create_cuda_graph_inputs(
                capture_bs,
                changed_seq_lens,
                config.seq_size_per_block,
                active_batch_size=active_bs,
            )
            attn_op.prepare_for_cuda_graph_replay(changed_inputs)
            expected = self._expected_page_metadata(
                changed_seq_lens,
                config.seq_size_per_block,
                batch_size=capture_bs,
            )
            with self.subTest(phase="changed topology replans"):
                self.assertEqual(plan_mock.call_count, 1)
                changed_call = plan_mock.call_args
                self.assertFalse(changed_call.args[0].is_cuda)
                self.assertTrue(changed_call.args[1].is_cuda)
                self.assertFalse(changed_call.args[2].is_cuda)
                self.assertTrue(changed_call.kwargs["non_blocking"])
                self._assert_page_metadata(fmha_params, expected)
                self._assert_graph_buffer_pointers(
                    attn_op, fmha_params, graph_buffer_pointers
                )
                self.assertEqual(
                    attn_op._cuda_core_plan_page_indptr_h.tolist(),
                    expected.page_indptr,
                )

            same_topology_seq_lens = [101, 201]
            block_id_offset = 17
            same_topology_inputs = self._create_cuda_graph_inputs(
                capture_bs,
                same_topology_seq_lens,
                config.seq_size_per_block,
                active_batch_size=active_bs,
                block_id_offset=block_id_offset,
            )
            plan_mock.reset_mock()
            attn_op.prepare_for_cuda_graph_replay(same_topology_inputs)
            expected = self._expected_page_metadata(
                same_topology_seq_lens,
                config.seq_size_per_block,
                batch_size=capture_bs,
                block_id_offset=block_id_offset,
            )
            with self.subTest(phase="same topology skips replan"):
                self.assertEqual(plan_mock.call_count, 0)
                self._assert_page_metadata(fmha_params, expected)
                self._assert_graph_buffer_pointers(
                    attn_op, fmha_params, graph_buffer_pointers
                )

            # A skipped topology-only replan must produce the same active-slot
            # output as explicitly refreshing the identical plan.
            local_head_num = config.head_num // config.tp_size
            local_kv_head_num = config.head_num_kv // config.tp_size
            q = self._create_query_tensor(
                capture_bs, local_head_num, config.size_per_head
            )
            total_blocks = max(expected.page_indices) + 1
            kv_cache, k_cache, v_cache = self._create_kv_cache(
                total_blocks,
                config.seq_size_per_block,
                local_kv_head_num,
                config.size_per_head,
                dtype=self.cache_dtype(config.attn_configs),
            )
            skipped_replan_output = attn_op.forward(q, kv_cache, fmha_params)
            self._assert_active_output_matches_reference(
                attn_op,
                fmha_params,
                same_topology_inputs,
                same_topology_seq_lens,
                q,
                kv_cache,
                k_cache,
                v_cache,
                config.seq_size_per_block,
            )
            attn_op._plan_decode_wrapper(same_topology_inputs)
            forced_replan_output = attn_op.forward(q, kv_cache, fmha_params)
            with self.subTest(phase="skipped replan output"):
                torch.testing.assert_close(
                    skipped_replan_output[:active_bs],
                    forced_replan_output[:active_bs],
                    rtol=1e-5,
                    atol=1e-5,
                )

            plan_mock.reset_mock()
            crossed_page_seq_lens = [129, 201]
            crossed_page_inputs = self._create_cuda_graph_inputs(
                capture_bs,
                crossed_page_seq_lens,
                config.seq_size_per_block,
                active_batch_size=active_bs,
            )
            attn_op.prepare_for_cuda_graph_replay(crossed_page_inputs)
            expected = self._expected_page_metadata(
                crossed_page_seq_lens,
                config.seq_size_per_block,
                batch_size=capture_bs,
            )
            with self.subTest(phase="page boundary replans"):
                self.assertEqual(plan_mock.call_count, 1)
                self._assert_page_metadata(fmha_params, expected)
                self._assert_graph_buffer_pointers(
                    attn_op, fmha_params, graph_buffer_pointers
                )

    def test_cuda_core_replay_matches_reference_with_stale_padding_ids(self):
        """Replanned active slots match reference despite stale padding IDs."""
        config = self._create_config(head_num=32, head_num_kv=32)
        capture_bs = 4
        active_bs = 2
        padding_block_id = 7
        capture_inputs = self._create_cuda_graph_inputs(
            capture_bs,
            [64, 128, 256, 512],
            config.seq_size_per_block,
        )
        attn_op = PyFlashinferDecodeAttnOp(config.attn_configs, capture_inputs)
        fmha_params = rtp_llm_ops.FlashInferMlaAttnParams()
        attn_op.set_params(fmha_params)

        local_head_num = config.head_num // config.tp_size
        local_kv_head_num = config.head_num_kv // config.tp_size
        q = self._create_query_tensor(capture_bs, local_head_num, config.size_per_head)
        changed_seq_lens = [100, 200]
        changed_expected = self._expected_page_metadata(
            changed_seq_lens,
            config.seq_size_per_block,
            batch_size=capture_bs,
            padding_block_id=padding_block_id,
        )
        crossed_page_seq_lens = [129, 201]
        crossed_expected = self._expected_page_metadata(
            crossed_page_seq_lens,
            config.seq_size_per_block,
            batch_size=capture_bs,
            padding_block_id=padding_block_id,
        )
        required_block_count = (
            max(
                max(changed_expected.page_indices),
                max(crossed_expected.page_indices),
            )
            + 1
        )
        kv_cache, k_cache, v_cache = self._create_kv_cache(
            required_block_count,
            config.seq_size_per_block,
            local_kv_head_num,
            config.size_per_head,
            dtype=self.cache_dtype(config.attn_configs),
        )

        with mock.patch.object(
            attn_op.decode_wrapper,
            "plan",
            wraps=attn_op.decode_wrapper.plan,
        ) as plan_mock:
            attn_op.prepare(capture_inputs)

            changed_inputs = self._create_cuda_graph_inputs(
                capture_bs,
                changed_seq_lens,
                config.seq_size_per_block,
                active_batch_size=active_bs,
                padding_block_id=padding_block_id,
            )
            plan_mock.reset_mock()
            attn_op.prepare_for_cuda_graph_replay(changed_inputs)
            with self.subTest(phase="changed topology reference"):
                self.assertEqual(plan_mock.call_count, 1)
                self.assertEqual(
                    changed_expected.page_indices[-2:],
                    [padding_block_id, padding_block_id],
                )
                self._assert_page_metadata(fmha_params, changed_expected)
                self._assert_active_output_matches_reference(
                    attn_op,
                    fmha_params,
                    changed_inputs,
                    changed_seq_lens,
                    q,
                    kv_cache,
                    k_cache,
                    v_cache,
                    config.seq_size_per_block,
                )

            crossed_page_inputs = self._create_cuda_graph_inputs(
                capture_bs,
                crossed_page_seq_lens,
                config.seq_size_per_block,
                active_batch_size=active_bs,
                padding_block_id=padding_block_id,
            )
            plan_mock.reset_mock()
            attn_op.prepare_for_cuda_graph_replay(crossed_page_inputs)
            with self.subTest(phase="page boundary reference"):
                self.assertEqual(plan_mock.call_count, 1)
                self._assert_page_metadata(fmha_params, crossed_expected)
                self._assert_active_output_matches_reference(
                    attn_op,
                    fmha_params,
                    crossed_page_inputs,
                    crossed_page_seq_lens,
                    q,
                    kv_cache,
                    k_cache,
                    v_cache,
                    config.seq_size_per_block,
                )


class _NoReleaseDecodeAttnOp(PyFlashinferDecodeAttnOp):
    def __del__(self):
        pass


class TestDynamicFp8DecodeUnit(unittest.TestCase):
    @staticmethod
    def _config() -> SimpleNamespace:
        rope_config = RopeConfig()
        rope_config.style = RopeStyle.Base
        rope_config.dim = 4
        rope_config.base = 10000
        return SimpleNamespace(
            dtype=torch.bfloat16,
            kv_cache_dtype=KvCacheDataType.FP8,
            fp8_kv_cache_mode=2,
            head_num=4,
            kv_head_num=2,
            size_per_head=4,
            tokens_per_block=32,
            kernel_tokens_per_block=8,
            need_rope_kv_cache=True,
            rope_config=rope_config,
            max_seq_len=128,
        )

    def test_mode2_allows_single_token_and_rejects_speculative_decode(self):
        config = self._config()
        config.use_mla = False
        config.use_logn_attn = False
        config.gen_num_per_cycle = 1
        _validate_dynamic_fp8_config(config, is_cuda_graph=False)

        config.gen_num_per_cycle = 2
        with self.assertRaisesRegex(ValueError, "multi-token decode"):
            _validate_dynamic_fp8_config(config, is_cuda_graph=False)

    def test_mode2_constructs_and_caches_versioned_fa2_direct_scale_jit(self):
        wrappers = [SimpleNamespace(_fixed_batch_size=0) for _ in range(2)]
        loaded_module = object()
        spec = SimpleNamespace(
            extra_include_dirs=None,
            build_and_load=mock.Mock(return_value=object()),
        )
        generator = mock.Mock(return_value=spec)
        module_adapter = mock.Mock(return_value=loaded_module)
        module = (
            "rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha"
        )
        with mock.patch(
            f"{module}.get_py_flashinfer_workspace_buffer",
            return_value=torch.empty(0),
        ), mock.patch(
            f"{module}.BatchDecodeWithPagedKVCacheWrapper",
            side_effect=wrappers,
        ) as wrapper_cls, mock.patch(
            f"{module}.flashinfer_decode.gen_customize_batch_prefill_module",
            generator,
        ), mock.patch(
            f"{module}.flashinfer_decode.get_batch_prefill_jit_module",
            module_adapter,
        ), mock.patch.dict(
            f"{module}._g_dynamic_fp8_jit_modules", clear=True
        ):
            op = _NoReleaseDecodeAttnOp(
                self._config(), SimpleNamespace(is_cuda_graph=False)
            )
            cached_op = _NoReleaseDecodeAttnOp(
                self._config(), SimpleNamespace(is_cuda_graph=False)
            )

        self.assertTrue(op.use_tensor_core)
        self.assertEqual(op.q_dtype, torch.bfloat16)
        self.assertEqual(op.kv_dtype, torch.float8_e4m3fn)
        self.assertEqual(wrapper_cls.call_count, 2)
        wrapper_kwargs = wrapper_cls.call_args.kwargs
        self.assertEqual(wrapper_kwargs["backend"], "fa2")
        self.assertTrue(wrapper_kwargs["use_tensor_cores"])
        self.assertNotIn("jit_args", wrapper_kwargs)
        generator.assert_called_once()
        spec.build_and_load.assert_called_once()
        module_adapter.assert_called_once()
        self.assertIs(op.decode_wrapper._jit_module, loaded_module)
        self.assertIs(cached_op.decode_wrapper._jit_module, loaded_module)
        jit_args = generator.call_args.args[1:]
        self.assertIn("direct_scale_v3", jit_args[0])
        self.assertIn("q_bfloat16", jit_args[0])
        self.assertIn("kv_float8_e4m3fn", jit_args[0])
        self.assertIn("o_bfloat16", jit_args[0])
        self.assertIn("idx_int32", jit_args[0])
        self.assertIn("hdq_4_hdv_4", jit_args[0])
        self.assertEqual(
            jit_args[1:7],
            (
                torch.bfloat16,
                torch.float8_e4m3fn,
                torch.bfloat16,
                torch.int32,
                4,
                4,
            ),
        )
        self.assertEqual(jit_args[7], ["kv_scale"])
        self.assertEqual(jit_args[8], ["float"])
        self.assertEqual(jit_args[9:11], ([], []))
        self.assertEqual(jit_args[11], "RtpLlmDynamicFp8Attention<use_custom_mask>")
        self.assertIn("use_per_token_kv_scale = true", jit_args[12])
        self.assertIn("RtpLlmDynamicFp8DefaultParams", jit_args[12])
        self.assertIn("math::rsqrt(float(HEAD_DIM_QK))", jit_args[12])
        self.assertIn("params.paged_kv.page_size.divmod", jit_args[12])
        self.assertNotIn("using Base::Base", jit_args[12])
        self.assertNotIn("physical_page_size", jit_args)
        self.assertNotIn("subdivision", jit_args)

    @staticmethod
    def _decode_op() -> _NoReleaseDecodeAttnOp:
        op = _NoReleaseDecodeAttnOp.__new__(_NoReleaseDecodeAttnOp)
        op.g_workspace_buffer = torch.empty(0)
        op.local_head_num = 4
        op.local_kv_head_num = 2
        op.head_dim_qk = 4
        op.dynamic_fp8 = True
        op.physical_page_size = 32
        op.seq_size_per_block = 8
        op.subdivision = 4
        op.use_tensor_core = True
        op.dtype = torch.bfloat16
        op.q_dtype = torch.bfloat16
        op.kv_dtype = torch.float8_e4m3fn
        op.enable_cuda_graph = False
        op._cuda_core_plan_page_indptr_h = None
        page_indptr = torch.tensor([0, 2, 3], dtype=torch.int32)
        page_indices = torch.tensor([12, 4, 9, 77], dtype=torch.int32)
        last_page_len = torch.tensor([8, 1], dtype=torch.int32)
        op.fmha_params = SimpleNamespace(
            decode_page_indptr_h=page_indptr,
            decode_page_indptr_d=page_indptr,
            page_indice_h=page_indices,
            page_indice_d=page_indices,
            paged_kv_last_page_len_h=last_page_len,
            paged_kv_last_page_len_d=last_page_len,
        )
        op.decode_wrapper = SimpleNamespace(plan=mock.Mock(), run=mock.Mock())
        return op

    def test_direct_scale_plan_and_forward_keep_original_page_ids(self):
        op = self._decode_op()
        op._plan_decode_wrapper(SimpleNamespace())

        plan_args = op.decode_wrapper.plan.call_args.args
        torch.testing.assert_close(
            plan_args[1], torch.tensor([12, 4, 9], dtype=torch.int32)
        )
        torch.testing.assert_close(
            op.fmha_params.page_indice_d,
            torch.tensor([12, 4, 9, 77], dtype=torch.int32),
        )
        self.assertEqual(
            op.decode_wrapper.plan.call_args.kwargs["q_data_type"], torch.bfloat16
        )
        self.assertEqual(
            op.decode_wrapper.plan.call_args.kwargs["kv_data_type"],
            torch.float8_e4m3fn,
        )

        q = torch.randn(2, 4, 4, dtype=torch.bfloat16)
        cache = SimpleNamespace(
            kv_cache_base=torch.empty(16, 2, 2, 8, 4, dtype=torch.float8_e4m3fn),
            kv_scale_base=torch.empty(16, 2 * 2 * 8, dtype=torch.float32),
        )
        op.decode_wrapper.run.side_effect = lambda query, _, __: query
        with mock.patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha."
            "_validate_dynamic_fp8_scale",
            return_value=cache.kv_scale_base,
        ) as validate_mock, mock.patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha."
            "rtp_llm_ops.gather_and_dequantize_fp8_kv_cache",
            create=True,
        ) as gather_mock, mock.patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha."
            "quantize_to_fp8_if_needed",
            side_effect=AssertionError("unit-scale cast must not run"),
        ):
            output = op.forward(q, cache, op.fmha_params)

        torch.testing.assert_close(output, q)
        validate_mock.assert_called_once_with(cache, 2, 8)
        gather_mock.assert_not_called()
        run_args = op.decode_wrapper.run.call_args.args
        self.assertEqual(run_args[0].dtype, torch.bfloat16)
        self.assertIs(run_args[1], cache.kv_cache_base)
        self.assertEqual(run_args[1].dtype, torch.float8_e4m3fn)
        self.assertIs(run_args[2], cache.kv_scale_base)

    def test_mode2_eager_prepare_uses_device_decode_metadata_fill(self):
        op = self._decode_op()
        op.decode_wrapper._fixed_batch_size = 0
        sequence_lengths = torch.tensor([8, 12], dtype=torch.int32).pin_memory()
        sequence_lengths_plus_1 = torch.tensor(
            [9, 13], dtype=torch.int32, device="cuda"
        )
        block_table = torch.tensor([[12, 4], [9, 7]], dtype=torch.int32, device="cuda")
        fill_decode = mock.Mock()
        op.fmha_params = SimpleNamespace(
            fill_decode_params_device=fill_decode,
            fill_params=mock.Mock(side_effect=AssertionError("host fill must not run")),
            fill_params_mha_device=mock.Mock(
                side_effect=AssertionError("generic device fill must not run")
            ),
            decode_page_indptr_h=torch.tensor([0, 2, 4], dtype=torch.int32),
            decode_page_indptr_d=torch.tensor(
                [0, 2, 4], dtype=torch.int32, device="cuda"
            ),
            page_indice_h=torch.empty(0, dtype=torch.int32),
            page_indice_d=block_table.flatten(),
            paged_kv_last_page_len_h=torch.tensor([1, 5], dtype=torch.int32),
            paged_kv_last_page_len_d=torch.tensor(
                [1, 5], dtype=torch.int32, device="cuda"
            ),
        )
        inputs = SimpleNamespace(
            input_lengths=torch.ones(2, dtype=torch.int32),
            sequence_lengths=sequence_lengths,
            sequence_lengths_plus_1_device=sequence_lengths_plus_1,
            kv_cache_kernel_block_id=block_table.cpu(),
            kv_cache_kernel_block_id_device=block_table,
        )

        params = op.prepare(inputs)

        self.assertIs(params, op.fmha_params)
        fill_decode.assert_called_once_with(
            sequence_lengths, sequence_lengths_plus_1, block_table, 8
        )
        plan_args = op.decode_wrapper.plan.call_args.args
        self.assertIs(plan_args[0], op.fmha_params.decode_page_indptr_h)
        self.assertTrue(plan_args[1].is_cuda)
        self.assertIs(plan_args[2], op.fmha_params.paged_kv_last_page_len_h)

    def test_direct_scale_validates_scale_storage_contract(self):
        device = torch.device("cuda")
        payload = torch.empty(16, 2, 2, 8, 4, device=device, dtype=torch.float8_e4m3fn)
        valid_scale = torch.empty(16, 2 * 2 * 8, device=device, dtype=torch.float32)
        cache = SimpleNamespace(
            kv_cache_base=payload,
            kv_scale_base=valid_scale,
        )
        self.assertIs(_validate_dynamic_fp8_scale(cache, 2, 8), valid_scale)

        invalid_cases = (
            (None, payload, "requires a kv_scale_base tensor"),
            (
                torch.empty(16, 32, dtype=torch.float32),
                payload,
                "requires kv_scale_base to be CUDA",
            ),
            (
                torch.empty(16, 32, device=device, dtype=torch.float16),
                payload,
                "dtype torch.float32",
            ),
            (
                torch.empty(32, 16, device=device, dtype=torch.float32).t(),
                payload,
                "contiguous kv_scale_base",
            ),
            (
                valid_scale,
                torch.empty(16, 2, 2, 8, 4, dtype=torch.float8_e4m3fn),
                "requires kv_cache_base to be CUDA",
            ),
            (
                torch.empty(16 * 32, device=device, dtype=torch.float32),
                payload,
                "shape",
            ),
            (
                torch.empty(16, 31, device=device, dtype=torch.float32),
                payload,
                "shape",
            ),
        )
        for scale, case_payload, message in invalid_cases:
            with self.subTest(message=message):
                invalid_cache = SimpleNamespace(
                    kv_cache_base=case_payload,
                    kv_scale_base=scale,
                )
                with self.assertRaisesRegex(ValueError, message):
                    _validate_dynamic_fp8_scale(invalid_cache, 2, 8)

    def test_mode2_decode_impl_with_rope_matches_base_reference_with_subdivision(
        self,
    ):
        for subdivision in (1, 2, 4):
            with self.subTest(subdivision=subdivision):
                self._check_mode2_decode_with_subdivision(subdivision)

    def _check_mode2_decode_with_subdivision(self, subdivision: int):
        device = torch.device("cuda")
        torch.manual_seed(2026)
        harness = BaseAttentionTest()
        harness.device = device
        sequence_lengths = [9, 13]
        kernel_page_size = 4
        physical_page_size = kernel_page_size * subdivision
        config = harness._create_config(
            head_num=4,
            head_num_kv=2,
            size_per_head=64,
            seq_size_per_block=kernel_page_size,
            data_type="bf16",
        )
        config.attn_configs.tokens_per_block = physical_page_size
        config.attn_configs.kv_cache_dtype = KvCacheDataType.FP8
        config.attn_configs.fp8_kv_cache_mode = 2
        config.attn_configs.need_rope_kv_cache = True
        config.attn_configs.rope_config.style = RopeStyle.Base
        config.attn_configs.rope_config.dim = 64
        config.attn_configs.rope_config.base = 10000
        config.attn_configs.rope_config.max_pos = 128
        config.attn_configs.max_seq_len = 128
        inputs = harness._create_attention_inputs_base(
            len(sequence_lengths), sequence_lengths, kernel_page_size
        )

        block_table = torch.tensor([[8, 3, 9, 0], [8, 6, 3, 11]], dtype=torch.int32)
        kernel_page_count = 12
        inputs.kv_cache_kernel_block_id = block_table
        inputs.kv_cache_kernel_block_id_device = block_table.to(device)

        impl = PyFlashinferDecodeImpl(config.attn_configs, inputs)
        torch.testing.assert_close(
            impl.fmha_params.positions_d[:2].cpu(),
            torch.tensor([8, 12], dtype=torch.int32),
        )
        qkv = torch.randn(
            2,
            (4 + 2 * 2) * 64,
            device=device,
            dtype=config.attn_configs.dtype,
        )
        q_flat, k_flat, v_flat = torch.split(qkv, [4 * 64, 2 * 64, 2 * 64], dim=-1)
        raw_q = q_flat.reshape(2, 4, 64)
        raw_k = k_flat.reshape(2, 2, 64)
        expected_v = v_flat.reshape(2, 2, 64)
        positions = torch.tensor([8, 12], device=device, dtype=torch.float32)
        inv_freq = torch.exp(
            -math.log(10000.0)
            * torch.arange(0, 64, 2, device=device, dtype=torch.float32)
            / 64
        )
        angles = positions[:, None] * inv_freq[None, :]
        cos = angles.cos()[:, None, :]
        sin = angles.sin()[:, None, :]

        def rope_reference(tensor: torch.Tensor) -> torch.Tensor:
            first = tensor[..., :32].float()
            second = tensor[..., 32:].float()
            return torch.cat(
                [first * cos - second * sin, second * cos + first * sin], dim=-1
            ).to(tensor.dtype)

        expected_q = rope_reference(raw_q)
        expected_k = rope_reference(raw_k)

        k_cache = torch.zeros(
            kernel_page_count,
            2,
            kernel_page_size,
            64,
            device=device,
            dtype=config.attn_configs.dtype,
        )
        v_cache = torch.zeros_like(k_cache)
        k_tokens = []
        v_tokens = []
        target_pages = []
        target_offsets = []
        block_id_list = []
        history_tokens = {}
        for batch_idx, sequence_length in enumerate(sequence_lengths):
            block_ids = block_table[
                batch_idx, : math.ceil(sequence_length / kernel_page_size)
            ].tolist()
            block_id_list.append(block_ids)
            for position in range(sequence_length - 1):
                kernel_page = block_ids[position // kernel_page_size]
                kernel_offset = position % kernel_page_size
                storage_location = (kernel_page, kernel_offset)
                if storage_location not in history_tokens:
                    key = torch.randn(
                        2, 64, device=device, dtype=config.attn_configs.dtype
                    )
                    value = torch.randn_like(key)
                    history_tokens[storage_location] = (key, value)
                    k_tokens.append(key)
                    v_tokens.append(value)
                    k_cache[kernel_page, :, kernel_offset] = key
                    v_cache[kernel_page, :, kernel_offset] = value
                    target_pages.append(kernel_page // subdivision)
                    target_offsets.append(
                        kernel_page % subdivision * kernel_page_size + kernel_offset
                    )

            current_position = sequence_length - 1
            current_kernel_page = block_ids[current_position // kernel_page_size]
            current_kernel_offset = current_position % kernel_page_size
            k_cache[current_kernel_page, :, current_kernel_offset] = expected_k[
                batch_idx
            ]
            v_cache[current_kernel_page, :, current_kernel_offset] = expected_v[
                batch_idx
            ]

        cache = LayerKVCache()
        cache.kv_cache_base = torch.zeros(
            kernel_page_count,
            2,
            2,
            kernel_page_size,
            64,
            device=device,
            dtype=torch.float8_e4m3fn,
        )
        cache.kv_scale_base = torch.ones(
            kernel_page_count,
            2 * 2 * kernel_page_size,
            device=device,
            dtype=torch.float32,
        )
        rtp_llm_ops.quantize_and_write_fp8_kv_cache(
            torch.stack(k_tokens),
            torch.stack(v_tokens),
            cache.kv_cache_base,
            cache.kv_scale_base,
            torch.tensor(target_pages, device=device, dtype=torch.int32),
            torch.tensor(target_offsets, device=device, dtype=torch.int32),
            physical_page_size,
            kernel_page_size,
            subdivision,
        )

        output = impl.forward(qkv.clone(), cache)
        scale_view = cache.kv_scale_base.view(kernel_page_count, 2, 2, kernel_page_size)
        self.assertFalse(torch.equal(scale_view[:, 0], scale_view[:, 1]))
        restored_k_cache = (
            cache.kv_cache_base[:, 0].float() * scale_view[:, 0].unsqueeze(-1)
        ).to(expected_q.dtype)
        restored_v_cache = (
            cache.kv_cache_base[:, 1].float() * scale_view[:, 1].unsqueeze(-1)
        ).to(expected_q.dtype)
        reference = compute_flashinfer_decode_reference(
            expected_q,
            restored_k_cache,
            restored_v_cache,
            sequence_lengths,
            block_id_list,
            kernel_page_size,
        )
        torch.testing.assert_close(output, reference, rtol=0.06, atol=0.04)

        for batch_idx, sequence_length in enumerate(sequence_lengths):
            position = sequence_length - 1
            kernel_page = int(
                block_table[batch_idx, position // kernel_page_size].item()
            )
            storage_token = position % kernel_page_size
            for kv, expected in enumerate((expected_k, expected_v)):
                restored = cache.kv_cache_base[
                    kernel_page, kv, :, storage_token
                ].float() * scale_view[kernel_page, kv, :, storage_token].unsqueeze(-1)
                torch.testing.assert_close(
                    restored,
                    expected[batch_idx].float(),
                    rtol=0.13,
                    atol=0.02,
                )

    def test_impl_uses_one_fused_prepare_before_direct_decode(self):
        config = self._config()
        events = []
        query = torch.randn(2, 4, 4, dtype=torch.bfloat16)
        qkv = torch.empty(2, 32, dtype=torch.bfloat16)
        payload = object()
        scales = object()
        cache = SimpleNamespace(kv_cache_base=payload, kv_scale_base=scales)
        decode = SimpleNamespace(
            set_params=mock.Mock(),
            prepare=mock.Mock(),
            prepare_for_cuda_graph_replay=mock.Mock(),
            forward=mock.Mock(
                side_effect=lambda *args: events.append("attention") or query
            ),
        )
        fused_prepare = mock.Mock(
            side_effect=lambda *args: events.append("fused_prepare") or query
        )

        with mock.patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha."
            "PyFlashinferDecodeAttnOp",
            return_value=decode,
        ), mock.patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha."
            "MhaRotaryEmbeddingOp"
        ) as rope_op, mock.patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha."
            "KVCacheWriteOp"
        ) as writer_op, mock.patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha."
            "FusedRopeKVCacheDecodeOp"
        ) as legacy_fused_op, mock.patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha."
            "get_rope_cache_once",
            return_value=SimpleNamespace(data=None),
        ), mock.patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha."
            "check_rope_cache",
            return_value=False,
        ), mock.patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha."
            "_validate_dynamic_fp8_scale",
            return_value=scales,
        ), mock.patch.object(
            rtp_llm_ops,
            "fused_rope_quantize_and_write_fp8_kv_cache",
            fused_prepare,
        ), mock.patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha."
            "common.create_write_cache_store_impl",
            return_value=None,
        ), mock.patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha."
            "common.apply_write_cache_store"
        ):
            impl = PyFlashinferDecodeImpl(config, SimpleNamespace())
            graph_inputs = SimpleNamespace()
            impl.prepare_cuda_graph(graph_inputs)
            output = impl.forward(qkv, cache)

        decode.prepare_for_cuda_graph_replay.assert_called_once_with(graph_inputs)
        rope_op.assert_not_called()
        writer_op.assert_not_called()
        legacy_fused_op.assert_not_called()
        fused_prepare.assert_called_once()
        self.assertTrue(impl.support_cuda_graph())
        self.assertIs(output, query)
        self.assertEqual(events, ["fused_prepare", "attention"])
        fused_args = fused_prepare.call_args.args
        self.assertIs(fused_args[0], qkv)
        self.assertIs(fused_args[1], payload)
        self.assertIs(fused_args[2], scales)
        self.assertIs(decode.forward.call_args.args[0], query)
        self.assertIs(decode.forward.call_args.args[1], cache)
        self.assertIs(decode.forward.call_args.args[2], impl.fmha_params)

    def test_impl_uses_fused_prepare_with_no_rope_style_when_rope_is_disabled(self):
        config = self._config()
        config.need_rope_kv_cache = False
        qkv = torch.arange(64, dtype=torch.bfloat16).reshape(2, 32)
        query = qkv[:, :16].reshape(2, 4, 4)
        payload = object()
        scales = object()
        cache = SimpleNamespace(kv_cache_base=payload, kv_scale_base=scales)
        decode = SimpleNamespace(
            set_params=mock.Mock(),
            prepare=mock.Mock(),
            forward=mock.Mock(side_effect=lambda query, *_: query),
        )
        fused_prepare = mock.Mock(return_value=query)

        with mock.patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha."
            "PyFlashinferDecodeAttnOp",
            return_value=decode,
        ), mock.patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha."
            "MhaRotaryEmbeddingOp"
        ) as rope_op, mock.patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha."
            "KVCacheWriteOp"
        ) as writer_op, mock.patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha."
            "_validate_dynamic_fp8_scale",
            return_value=scales,
        ), mock.patch.object(
            rtp_llm_ops,
            "fused_rope_quantize_and_write_fp8_kv_cache",
            fused_prepare,
        ), mock.patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha."
            "common.create_write_cache_store_impl",
            return_value=None,
        ), mock.patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha."
            "common.apply_write_cache_store"
        ):
            impl = PyFlashinferDecodeImpl(config, SimpleNamespace())
            output = impl.forward(qkv, cache)

        rope_op.assert_not_called()
        writer_op.assert_not_called()
        fused_prepare.assert_called_once()
        self.assertEqual(fused_prepare.call_args.args[10].style, RopeStyle.No)
        self.assertIsNone(fused_prepare.call_args.args[11])
        self.assertIs(output, query)


class TestPyFlashinferDecodeAttnOpFP8(TestPyFlashinferDecodeAttnOp):
    kv_cache_dtype = KvCacheDataType.FP8
    rtol = 4e-2
    atol = 4e-2
    max_mismatch_rate = 1e-5


class TestPyFlashinferDecodeCudaGraphFP8(TestPyFlashinferDecodeCudaGraph):
    kv_cache_dtype = KvCacheDataType.FP8
    rtol = 4e-2
    atol = 4e-2
    max_mismatch_rate = 1e-5


if __name__ == "__main__":
    unittest.main()
