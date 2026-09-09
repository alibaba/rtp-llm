import logging
import math
import sys
import unittest
from typing import List, NamedTuple, Optional
from unittest import mock

import torch
from attention_ref import compute_flashinfer_decode_reference
from base_attention_test import BaseAttentionTest, compare_tensors

from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
    PyFlashinferDecodeAttnOp,
    partition_cascade_decode_pages,
)
from rtp_llm.models_py.utils.arch import is_rtx_pro_5000_blackwell
from rtp_llm.ops import KvCacheDataType
from rtp_llm.ops.compute_ops import (
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


class TestCascadeDecodeMetadata(unittest.TestCase):
    @staticmethod
    def _partition(rows, kv_lens, last_page_lens, page_size=64):
        indptr = [0]
        indices = []
        for row in rows:
            indices.extend(row)
            indptr.append(len(indices))
        return partition_cascade_decode_pages(
            torch.tensor(indptr, dtype=torch.int32),
            torch.tensor(indices, dtype=torch.int32),
            torch.tensor(last_page_lens, dtype=torch.int32),
            torch.tensor(kv_lens, dtype=torch.int32),
            page_size,
        )

    def test_partitions_shared_full_pages_and_variable_suffixes(self):
        metadata = self._partition(
            [
                [11, 12, 101],
                [11, 12, 201],
                [11, 12, 301, 302, 303],
                [11, 12, 401, 402, 403],
            ],
            [129, 191, 257, 320],
            [1, 63, 1, 64],
        )

        self.assertEqual(metadata.shared_page_count, 2)
        self.assertEqual(metadata.live_batch_size, 4)
        self.assertEqual(metadata.shared_page_indices.tolist(), [11, 12])
        self.assertEqual(metadata.suffix_page_indptr.tolist(), [0, 1, 2, 5, 8])
        self.assertEqual(
            metadata.suffix_page_indices.tolist(),
            [101, 201, 301, 302, 303, 401, 402, 403],
        )
        self.assertEqual(metadata.suffix_last_page_len.tolist(), [1, 63, 1, 64])
        self.assertEqual(metadata.merge_mask.tolist(), [True, True, True, True])

    def test_current_token_page_is_never_shared(self):
        expected_shared_pages = {64: 0, 65: 1, 128: 1, 129: 2}
        for kv_len, expected in expected_shared_pages.items():
            page_count = math.ceil(kv_len / 64)
            common = list(range(11, 11 + max(page_count - 1, 0)))
            rows = [common + [101], common + [201]]
            with self.subTest(kv_len=kv_len):
                metadata = self._partition(
                    rows,
                    [kv_len, kv_len],
                    [kv_len % 64 or 64] * 2,
                )
                self.assertEqual(metadata.shared_page_count, expected)
                self.assertGreaterEqual(
                    metadata.suffix_page_indptr[1] - metadata.suffix_page_indptr[0],
                    1,
                )

    def test_shared_prefix_requires_ordered_physical_identity(self):
        metadata = self._partition(
            [[11, 12, 101], [11, 13, 201], [11, 12, 301]],
            [129, 129, 129],
            [1, 1, 1],
        )
        self.assertEqual(metadata.shared_page_indices.tolist(), [11])
        self.assertEqual(metadata.shared_page_count, 1)

    def test_padding_rows_do_not_participate(self):
        metadata = self._partition(
            [[11, 12, 101], [11, 12, 201], [0], [0]],
            [129, 191, 1, 1],
            [1, 63, 1, 1],
        )
        self.assertEqual(metadata.live_batch_size, 2)
        self.assertEqual(metadata.shared_page_indices.tolist(), [11, 12])
        self.assertEqual(metadata.suffix_page_indices.tolist(), [101, 201, 0, 0])
        self.assertEqual(metadata.merge_mask.tolist(), [True, True, False, False])


def require_cascade_test(test_case: BaseAttentionTest) -> None:
    if test_case.kv_cache_dtype != KvCacheDataType.BASE:
        test_case.skipTest("cascade decode intentionally excludes FP8 KV cache")
    if not is_rtx_pro_5000_blackwell():
        test_case.skipTest("cascade decode is enabled only on RTX PRO 5000")


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

    def test_cascade_is_disabled_on_other_products(self):
        config = self._create_config(
            head_num=32,
            head_num_kv=8,
            size_per_head=128,
            seq_size_per_block=64,
            data_type="bf16",
        )
        attn_inputs = self._create_attention_inputs(
            4, [129, 129, 129, 129], 64, dtype=torch.bfloat16
        )
        with mock.patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha.is_rtx_pro_5000_blackwell",
            return_value=False,
        ):
            attn_op = PyFlashinferDecodeAttnOp(config.attn_configs, attn_inputs)
        self.assertFalse(attn_op.cascade_enabled)

    def test_cascade_rejects_batch_below_break_even(self):
        require_cascade_test(self)
        config = self._create_config(
            head_num=32,
            head_num_kv=8,
            size_per_head=128,
            seq_size_per_block=64,
            data_type="bf16",
        )
        attn_inputs = self._create_attention_inputs(
            8, [129] * 8, 64, dtype=torch.bfloat16
        )
        attn_op = PyFlashinferDecodeAttnOp(config.attn_configs, attn_inputs)
        self.assertFalse(attn_op.cascade_enabled)

    @mock.patch(
        "rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha.MIN_CASCADE_BATCH_SIZE",
        4,
    )
    def test_cascade_rollback_switch(self):
        require_cascade_test(self)
        config = self._create_config(
            head_num=32,
            head_num_kv=8,
            size_per_head=128,
            seq_size_per_block=64,
            data_type="bf16",
        )
        attn_inputs = self._create_attention_inputs(
            4, [129, 129, 129, 129], 64, dtype=torch.bfloat16
        )
        with mock.patch.dict("os.environ", {"RTP_LLM_DISABLE_FLASHINFER_CASCADE": "1"}):
            attn_op = PyFlashinferDecodeAttnOp(config.attn_configs, attn_inputs)
        self.assertFalse(attn_op.cascade_enabled)

    @mock.patch(
        "rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha.MIN_CASCADE_BATCH_SIZE",
        4,
    )
    def test_cascade_rejects_device_metadata_path(self):
        require_cascade_test(self)
        config = self._create_config(
            head_num=32,
            head_num_kv=8,
            size_per_head=128,
            seq_size_per_block=64,
            data_type="bf16",
        )
        attn_inputs = self._create_attention_inputs(
            4, [129, 129, 129, 129], 64, dtype=torch.bfloat16
        )
        attn_inputs.sequence_lengths = attn_inputs.sequence_lengths.cuda()
        attn_op = PyFlashinferDecodeAttnOp(config.attn_configs, attn_inputs)
        self.assertFalse(attn_op.cascade_enabled)

    @mock.patch(
        "rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha.MIN_CASCADE_BATCH_SIZE",
        4,
    )
    def test_cascade_rejects_unprofiled_attention_shape(self):
        require_cascade_test(self)
        config = self._create_config(
            head_num=16,
            head_num_kv=2,
            size_per_head=256,
            seq_size_per_block=64,
            data_type="bf16",
        )
        attn_inputs = self._create_attention_inputs(
            4, [129, 129, 129, 129], 64, dtype=torch.bfloat16
        )
        attn_op = PyFlashinferDecodeAttnOp(config.attn_configs, attn_inputs)
        self.assertFalse(attn_op.cascade_enabled)

    @mock.patch(
        "rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha.MIN_CASCADE_BATCH_SIZE",
        4,
    )
    def test_bf16_no_shared_prefix_uses_standard_decode(self):
        require_cascade_test(self)
        config = self._create_config(
            head_num=32,
            head_num_kv=8,
            size_per_head=128,
            seq_size_per_block=64,
            data_type="bf16",
        )
        attn_inputs = self._create_attention_inputs(
            4, [129, 129, 129, 129], 64, dtype=torch.bfloat16
        )
        block_ids = torch.tensor(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
            dtype=torch.int32,
        )
        attn_inputs.kv_cache_block_id = block_ids
        attn_inputs.kv_cache_block_id_device = block_ids.cuda()
        attn_inputs.kv_cache_kernel_block_id = block_ids
        attn_inputs.kv_cache_kernel_block_id_device = block_ids.cuda()

        attn_op = PyFlashinferDecodeAttnOp(config.attn_configs, attn_inputs)
        params = rtp_llm_ops.FlashInferMlaAttnParams()
        attn_op.set_params(params)
        attn_op.prepare(attn_inputs)
        self.assertTrue(attn_op.cascade_enabled)
        self.assertFalse(attn_op._cascade_active)
        self.assertIsNone(attn_op._cascade_state)

    @mock.patch(
        "rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha.MIN_CASCADE_BATCH_SIZE",
        4,
    )
    def test_bf16_shared_prefix_cascade_matches_unsplit_reference(self):
        require_cascade_test(self)
        config = self._create_config(
            head_num=32,
            head_num_kv=8,
            size_per_head=128,
            seq_size_per_block=64,
            data_type="bf16",
        )
        sequence_lengths = [129, 191, 257, 320]
        attn_inputs = self._create_attention_inputs(
            len(sequence_lengths),
            sequence_lengths,
            config.seq_size_per_block,
            dtype=torch.bfloat16,
        )
        block_ids = torch.tensor(
            [
                [1, 2, 3, 0, 0],
                [1, 2, 4, 0, 0],
                [1, 2, 5, 6, 7],
                [1, 2, 8, 9, 10],
            ],
            dtype=torch.int32,
        )
        attn_inputs.kv_cache_block_id = block_ids
        attn_inputs.kv_cache_block_id_device = block_ids.cuda()
        attn_inputs.kv_cache_kernel_block_id = block_ids
        attn_inputs.kv_cache_kernel_block_id_device = block_ids.cuda()

        attn_op = PyFlashinferDecodeAttnOp(config.attn_configs, attn_inputs)
        self.assertTrue(attn_op.cascade_enabled)
        params = rtp_llm_ops.FlashInferMlaAttnParams()
        attn_op.set_params(params)
        attn_op.prepare(attn_inputs)
        self.assertTrue(attn_op._cascade_active)
        self.assertEqual(attn_op._cascade_shared_page_count, 2)

        q = self._create_query_tensor(
            len(sequence_lengths), 32, 128, dtype=torch.bfloat16
        )
        kv_cache, k_cache, v_cache = self._create_kv_cache(
            11,
            config.seq_size_per_block,
            8,
            128,
            dtype=torch.bfloat16,
        )
        output = attn_op.forward(q, kv_cache, params)
        reference = compute_flashinfer_decode_reference(
            q,
            k_cache,
            v_cache,
            sequence_lengths,
            [
                [1, 2, 3],
                [1, 2, 4],
                [1, 2, 5, 6, 7],
                [1, 2, 8, 9, 10],
            ],
            config.seq_size_per_block,
        )
        self._assert_output_close(
            output,
            reference,
            name="BF16 shared-prefix cascade output",
            rtol=2e-2,
            atol=2e-2,
        )

    def _run_bf16_long_shared_prefix_batch32(self, disable_cascade: bool):
        require_cascade_test(self)
        config = self._create_config(
            head_num=32,
            head_num_kv=8,
            size_per_head=128,
            seq_size_per_block=64,
            data_type="bf16",
        )
        batch_size = 32
        sequence_lengths = [4097] * batch_size
        attn_inputs = self._create_attention_inputs(
            batch_size, sequence_lengths, 64, dtype=torch.bfloat16
        )
        shared_pages = list(range(1, 64))
        block_lists = [
            shared_pages + [64 + 2 * row, 65 + 2 * row] for row in range(batch_size)
        ]
        block_ids = torch.tensor(block_lists, dtype=torch.int32)
        attn_inputs.kv_cache_block_id = block_ids
        attn_inputs.kv_cache_block_id_device = block_ids.cuda()
        attn_inputs.kv_cache_kernel_block_id = block_ids
        attn_inputs.kv_cache_kernel_block_id_device = block_ids.cuda()

        with mock.patch.dict(
            "os.environ",
            {"RTP_LLM_DISABLE_FLASHINFER_CASCADE": "1" if disable_cascade else "0"},
        ):
            attn_op = PyFlashinferDecodeAttnOp(config.attn_configs, attn_inputs)
        params = rtp_llm_ops.FlashInferMlaAttnParams()
        attn_op.set_params(params)
        attn_op.prepare(attn_inputs)
        self.assertEqual(attn_op._cascade_active, not disable_cascade)
        self.assertEqual(
            attn_op._cascade_shared_page_count, 0 if disable_cascade else 63
        )

        q = self._create_query_tensor(batch_size, 32, 128, dtype=torch.bfloat16)
        kv_cache, k_cache, v_cache = self._create_kv_cache(
            128, 64, 8, 128, dtype=torch.bfloat16
        )
        output = attn_op.forward(q, kv_cache, params)
        reference = compute_flashinfer_decode_reference(
            q,
            k_cache,
            v_cache,
            sequence_lengths,
            block_lists,
            64,
        )
        self._assert_output_close(
            output,
            reference,
            name="BF16 batch-32 long shared-prefix output",
            rtol=2e-2,
            atol=2e-2,
        )

    def test_bf16_long_shared_prefix_batch32(self):
        self._run_bf16_long_shared_prefix_batch32(disable_cascade=False)

    def test_bf16_long_shared_prefix_standard_control_batch32(self):
        self._run_bf16_long_shared_prefix_batch32(disable_cascade=True)

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

    @mock.patch(
        "rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha.MIN_CASCADE_BATCH_SIZE",
        4,
    )
    def test_cascade_graph_replay_shared_no_shared_shared(self):
        require_cascade_test(self)
        config = self._create_config(
            head_num=32,
            head_num_kv=8,
            size_per_head=128,
            seq_size_per_block=64,
            data_type="bf16",
        )
        sequence_lengths = [129] * 4
        inputs = self._create_cuda_graph_inputs(
            4,
            sequence_lengths,
            config.seq_size_per_block,
            dtype=torch.bfloat16,
        )
        q = self._create_query_tensor(4, 32, 128, dtype=torch.bfloat16)
        kv_cache, k_cache, v_cache = self._create_kv_cache(
            24,
            config.seq_size_per_block,
            8,
            128,
            dtype=torch.bfloat16,
        )
        attn_op = PyFlashinferDecodeAttnOp(config.attn_configs, inputs)
        self.assertTrue(attn_op.cascade_enabled)
        params = rtp_llm_ops.FlashInferMlaAttnParams()
        attn_op.set_params(params)

        def install_pages(rows):
            pages = torch.tensor(rows, dtype=torch.int32)
            if inputs.kv_cache_kernel_block_id_device.shape != pages.shape:
                raise AssertionError("graph page-table shape must stay fixed")
            inputs.kv_cache_kernel_block_id = pages
            inputs.kv_cache_block_id = pages
            inputs.kv_cache_kernel_block_id_device.copy_(pages)
            inputs.kv_cache_block_id_device = inputs.kv_cache_kernel_block_id_device

        def assert_reference(rows, output, active_count=4):
            active_lengths = sequence_lengths[:active_count]
            reference = compute_flashinfer_decode_reference(
                q[:active_count],
                k_cache,
                v_cache,
                active_lengths,
                rows[:active_count],
                config.seq_size_per_block,
            )
            self._assert_output_close(
                output[:active_count],
                reference,
                name="CUDA graph cascade replay output",
                rtol=2e-2,
                atol=2e-2,
            )

        shared_a = [[1, 2, 3], [1, 2, 4], [1, 2, 5], [1, 2, 6]]
        no_shared = [[1, 2, 3], [7, 8, 4], [9, 10, 5], [11, 12, 6]]
        shared_b = [[13, 14, 3], [13, 14, 4], [13, 14, 5], [13, 14, 6]]
        padded = [[15, 16, 3], [15, 16, 4], [0, 0, 0], [0, 0, 0]]

        install_pages(shared_a)
        attn_op.prepare(inputs)
        self.assertEqual(attn_op._cascade_shared_page_count, 2)
        state = attn_op._cascade_state
        self.assertIsNotNone(state)
        pointers = (
            state.shared_wrapper._paged_kv_indptr_buf.data_ptr(),
            state.shared_page_indices_d.data_ptr(),
            state.suffix_wrapper._paged_kv_indptr_buf.data_ptr(),
            state.suffix_page_indices_d.data_ptr(),
            state.merge_mask_d.data_ptr(),
            state.suffix_out.data_ptr(),
        )

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            attn_op.forward(q, kv_cache, params)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = attn_op.forward(q, kv_cache, params)
        graph.replay()
        torch.cuda.synchronize()
        assert_reference(shared_a, graph_output)

        for rows, expected_shared in ((no_shared, 0), (shared_b, 2)):
            install_pages(rows)
            attn_op.prepare_for_cuda_graph_replay(inputs)
            self.assertEqual(attn_op._cascade_shared_page_count, expected_shared)
            graph.replay()
            torch.cuda.synchronize()
            assert_reference(rows, graph_output)
            self.assertEqual(
                pointers,
                (
                    state.shared_wrapper._paged_kv_indptr_buf.data_ptr(),
                    state.shared_page_indices_d.data_ptr(),
                    state.suffix_wrapper._paged_kv_indptr_buf.data_ptr(),
                    state.suffix_page_indices_d.data_ptr(),
                    state.merge_mask_d.data_ptr(),
                    state.suffix_out.data_ptr(),
                ),
            )

        install_pages(padded)
        inputs.sequence_lengths.copy_(torch.tensor([128, 128, 0, 0]))
        attn_op.prepare_for_cuda_graph_replay(inputs)
        self.assertEqual(attn_op._cascade_shared_page_count, 2)
        self.assertEqual(state.merge_mask_h.tolist(), [True, True, False, False])
        graph.replay()
        torch.cuda.synchronize()
        assert_reference(padded, graph_output, active_count=2)
        self.assertEqual(
            pointers,
            (
                state.shared_wrapper._paged_kv_indptr_buf.data_ptr(),
                state.shared_page_indices_d.data_ptr(),
                state.suffix_wrapper._paged_kv_indptr_buf.data_ptr(),
                state.suffix_page_indices_d.data_ptr(),
                state.merge_mask_d.data_ptr(),
                state.suffix_out.data_ptr(),
            ),
        )

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
