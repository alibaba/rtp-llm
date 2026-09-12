import logging
import math
import sys
import unittest
from typing import List, NamedTuple, Optional
from unittest import mock

import torch
from attention_ref import compute_flashinfer_decode_reference
from base_attention_test import BaseAttentionTest, TestConfig

from rtp_llm.models_py.modules.factory.attention.cuda_impl import py_flashinfer_mha
from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
    PyFlashinferDecodeAttnOp,
)
from rtp_llm.models_py.utils.arch import is_sm90
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


class _CudaGraphTestDecodeAttnOp(PyFlashinferDecodeAttnOp):
    """Keep actual-graph test workspaces out of the process-global pool."""

    def __del__(self):
        # A failed assertion can keep the captured graph alive through its
        # traceback. Releasing that workspace to the global pool would let a
        # later test reuse storage that the graph still references.
        pass


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
        self.assertFalse(attn_op._is_cuda_core_graph_mode())
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
            q.to(attn_op.q_dtype).to(q.dtype),
            k_cache.to(q.dtype),
            v_cache.to(q.dtype),
            sequence_lengths,
            block_id_list,
            config.seq_size_per_block,
        )
        self._assert_output_close(
            output,
            reference,
            name="Eager CUDA-metadata decode output",
        )


class TestDecodeHostFillValidation(BaseAttentionTest):
    """Validate CUDA-core graph host metadata before it reaches C++."""

    def test_normalizes_cuda_sequence_metadata(self):
        """CUDA-resident base metadata is copied to validated host tensors."""
        config = self._create_config(head_num=32, head_num_kv=32)
        sequence_lengths = [64, 129]
        inputs = self._create_attention_inputs_base(
            batch_size=2,
            sequence_lengths=sequence_lengths,
            seq_size_per_block=config.seq_size_per_block,
        )
        inputs.is_prefill = False
        inputs.is_cuda_graph = True
        inputs.sequence_lengths = torch.tensor(
            [sequence_length - 1 for sequence_length in sequence_lengths],
            dtype=torch.int32,
            device="cuda",
        )

        fill_inputs = py_flashinfer_mha._decode_graph_host_fill_inputs(
            inputs,
            config.seq_size_per_block,
        )
        self.assertFalse(fill_inputs.sequence_lengths.is_cuda)
        self.assertFalse(fill_inputs.input_lengths.is_cuda)
        self.assertFalse(fill_inputs.kv_cache_block_id_host.is_cuda)
        self.assertEqual(
            fill_inputs.sequence_lengths.tolist(),
            [sequence_length - 1 for sequence_length in sequence_lengths],
        )

        attn_op = PyFlashinferDecodeAttnOp(config.attn_configs, inputs)
        self.assertTrue(attn_op._is_cuda_core_graph_mode())
        fmha_params = rtp_llm_ops.FlashInferMlaAttnParams()
        attn_op.set_params(fmha_params)
        attn_op.prepare(inputs)
        self.assertEqual(
            fmha_params.paged_kv_last_page_len_h.tolist(),
            [
                sequence_length % config.seq_size_per_block or config.seq_size_per_block
                for sequence_length in sequence_lengths
            ],
        )

    def test_validates_required_lengths(self):
        """Compatibility fallback rejects missing or mismatched lengths."""
        config = self._create_config()

        def make_inputs() -> PyAttentionInputs:
            inputs = self._create_attention_inputs_base(
                batch_size=2,
                sequence_lengths=[64, 128],
                seq_size_per_block=config.seq_size_per_block,
            )
            inputs.is_prefill = False
            inputs.is_cuda_graph = True
            return inputs

        def make_partial_inputs(
            include_input_lengths: bool = True,
            include_prefix_lengths: bool = True,
        ) -> PyAttentionInputs:
            source = make_inputs()
            inputs = PyAttentionInputs()
            if include_input_lengths:
                inputs.input_lengths = source.input_lengths
            if include_prefix_lengths:
                inputs.prefix_lengths = source.prefix_lengths
            inputs.sequence_lengths = source.sequence_lengths
            inputs.kv_cache_kernel_block_id = source.kv_cache_kernel_block_id
            inputs.kv_cache_kernel_block_id_device = (
                source.kv_cache_kernel_block_id_device
            )
            inputs.is_cuda_graph = True
            return inputs

        with self.subTest(case="missing sequence lengths"):
            inputs = make_inputs()
            inputs.sequence_lengths = torch.empty(0, dtype=torch.int32)
            with self.assertRaisesRegex(
                ValueError, "requires non-empty sequence_lengths"
            ):
                py_flashinfer_mha._decode_graph_host_fill_inputs(
                    inputs,
                    config.seq_size_per_block,
                )

        with self.subTest(case="missing input lengths"):
            inputs = make_partial_inputs(include_input_lengths=False)
            with self.assertRaisesRegex(ValueError, "requires input lengths"):
                py_flashinfer_mha._decode_graph_host_fill_inputs(
                    inputs,
                    config.seq_size_per_block,
                )

        with self.subTest(case="missing prefix lengths"):
            inputs = make_partial_inputs(include_prefix_lengths=False)
            fill_inputs = py_flashinfer_mha._decode_graph_host_fill_inputs(
                inputs,
                config.seq_size_per_block,
            )
            self.assertEqual(fill_inputs.prefix_lengths.numel(), 0)

        with self.subTest(case="mismatched sequence batch"):
            inputs = make_inputs()
            inputs.input_lengths = torch.ones(1, dtype=torch.int32)
            with self.assertRaisesRegex(ValueError, "batch size mismatch"):
                py_flashinfer_mha._decode_graph_host_fill_inputs(
                    inputs,
                    config.seq_size_per_block,
                )

    def _create_prefix_graph_inputs(self) -> tuple[PyAttentionInputs, TestConfig]:
        config = self._create_config(head_num=32, head_num_kv=32)
        inputs = self._create_attention_inputs_base(
            batch_size=2,
            sequence_lengths=[64, 128],
            seq_size_per_block=config.seq_size_per_block,
        )
        inputs.is_prefill = False
        inputs.is_cuda_graph = True
        inputs.prefix_lengths = torch.tensor([63, 127], dtype=torch.int32)
        inputs.sequence_lengths = torch.tensor([7, 7], dtype=torch.int32)
        return inputs, config

    def test_accepts_prefix_lengths(self):
        """Prefix metadata is preserved and takes the C++ prefix path."""
        inputs, config = self._create_prefix_graph_inputs()
        fill_inputs = py_flashinfer_mha._decode_graph_host_fill_inputs(
            inputs,
            config.seq_size_per_block,
        )
        self.assertEqual(fill_inputs.prefix_lengths.tolist(), [63, 127])
        self.assertEqual(
            fill_inputs.input_lengths.tolist(), inputs.input_lengths.tolist()
        )
        attn_op = PyFlashinferDecodeAttnOp(config.attn_configs, inputs)
        self.assertTrue(attn_op._is_cuda_core_graph_mode())
        fmha_params = rtp_llm_ops.FlashInferMlaAttnParams()
        attn_op.set_params(fmha_params)
        attn_op.prepare(inputs)
        self.assertEqual(fmha_params.decode_page_indptr_h.tolist(), [0, 1, 3])
        self.assertEqual(fmha_params.paged_kv_last_page_len_h.tolist(), [64, 64])

    def test_accepts_prefix_lengths_without_sequence_lengths(self):
        """The existing C++ prefix contract does not require sequence lengths."""
        inputs, config = self._create_prefix_graph_inputs()
        # C++ fillParams ignores sequence_lengths when prefix lengths exist;
        # preserve that pre-existing prefix-only input contract.
        inputs.sequence_lengths = torch.empty(0, dtype=torch.int32)
        prefix_only_fill_inputs = py_flashinfer_mha._decode_graph_host_fill_inputs(
            inputs,
            config.seq_size_per_block,
        )
        self.assertEqual(prefix_only_fill_inputs.sequence_lengths.numel(), 0)
        prefix_only_op = PyFlashinferDecodeAttnOp(config.attn_configs, inputs)
        prefix_only_params = rtp_llm_ops.FlashInferMlaAttnParams()
        prefix_only_op.set_params(prefix_only_params)
        prefix_only_op.prepare(inputs)
        self.assertEqual(prefix_only_params.decode_page_indptr_h.tolist(), [0, 1, 3])
        self.assertEqual(prefix_only_params.paged_kv_last_page_len_h.tolist(), [64, 64])

    def test_rejects_prefix_input_batch_mismatch(self):
        """Prefix and input metadata must describe the same graph batch."""
        inputs, config = self._create_prefix_graph_inputs()
        inputs.prefix_lengths = torch.tensor([63], dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, "prefix_lengths=1"):
            py_flashinfer_mha._decode_graph_host_fill_inputs(
                inputs,
                config.seq_size_per_block,
            )


class TestPyFlashinferDecodeCudaGraph(BaseAttentionTest):
    """Test CUDA graph buffer management for PyFlashinferDecodeAttnOp.

    These tests exercise the Python prepare/replay boundary for both decode
    backends and actual CUDA-core and tensor-core graph capture/replay.
    """

    def _assert_cuda_core_graph_mode(self, attn_op) -> None:
        self.assertTrue(attn_op._is_cuda_core_graph_mode())

    def _create_cuda_graph_inputs(
        self,
        batch_size: int,
        sequence_lengths: List[int],
        seq_size_per_block: int,
        dtype: torch.dtype = torch.float16,
        active_batch_size: Optional[int] = None,
        block_id_offset: int = 0,
        padding_block_id: int = 0,
        use_prefix_lengths: bool = False,
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
        attn_inputs.input_lengths = torch.ones(
            batch_size, dtype=torch.int32
        ).pin_memory()
        if use_prefix_lengths:
            attn_inputs.prefix_lengths = (seq_t - 1).pin_memory()
            attn_inputs.sequence_lengths = torch.empty(
                0, dtype=torch.int32
            ).pin_memory()
        else:
            attn_inputs.prefix_lengths = torch.empty(0, dtype=torch.int32).pin_memory()
            attn_inputs.sequence_lengths = (seq_t - 1).pin_memory()

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

    def _expected_page_metadata(
        self,
        active_sequence_lengths: List[int],
        seq_size_per_block: int,
        batch_size: Optional[int] = None,
        block_id_offset: int = 0,
        padding_block_id: int = 0,
    ) -> PageMetadata:
        """Derive page layout independently using the runner padding contract."""
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

    def test_padding_page_metadata_oracle_matches_runner_contract(self):
        """Anchor the independently calculated oracle to literal padding values."""
        inputs = self._create_cuda_graph_inputs(
            4,
            [100, 200],
            seq_size_per_block=64,
            active_batch_size=2,
        )
        self.assertEqual(inputs.sequence_lengths.tolist(), [99, 199, 0, 0])
        expected = self._expected_page_metadata(
            [100, 200],
            seq_size_per_block=64,
            batch_size=4,
        )
        self.assertEqual(expected.page_indptr, [0, 2, 6, 7, 8])
        self.assertEqual(expected.page_indices, [0, 1, 2, 3, 4, 5, 0, 0])
        self.assertEqual(expected.last_page_lens, [36, 8, 1, 1])

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

    def _assert_graph_buffer_pointers(
        self,
        attn_op,
        fmha_params,
        pointers,
        qo_indptr_ptr: Optional[int] = None,
    ) -> None:
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
        if qo_indptr_ptr is not None:
            self.assertEqual(
                attn_op.decode_wrapper._qo_indptr_buf.data_ptr(), qo_indptr_ptr
            )

    def _assert_plan_dtypes(self, plan_call, attn_op) -> None:
        self.assertEqual(plan_call.kwargs["q_data_type"], attn_op.q_dtype)
        self.assertEqual(plan_call.kwargs["kv_data_type"], attn_op.kv_dtype)
        self.assertEqual(plan_call.kwargs["o_data_type"], attn_op.dtype)

    def _assert_active_output_matches_reference(
        self,
        output: torch.Tensor,
        attn_inputs: PyAttentionInputs,
        sequence_lengths: List[int],
        q: torch.Tensor,
        q_dtype: torch.dtype,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        seq_size_per_block: int,
    ) -> None:
        active_batch_size = len(sequence_lengths)
        block_id_list = []
        for batch_idx, seq_len in enumerate(sequence_lengths):
            page_count = math.ceil(seq_len / seq_size_per_block)
            block_id_list.append(
                attn_inputs.kv_cache_kernel_block_id[batch_idx, :page_count].tolist()
            )
        reference_q = q[:active_batch_size].to(q_dtype).to(q.dtype)
        reference = compute_flashinfer_decode_reference(
            reference_q,
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
        config = self._create_config(head_num=32, head_num_kv=32)
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
        self.assertTrue(attn_op._graph_buffers_bound)
        self.assertEqual(attn_op._graph_batch_size, capture_bs)
        self.assertTrue(attn_op.decode_wrapper._use_cuda_graph)
        logging.info("_fixed_batch_size correctly set after prepare()")

    def test_disable_split_kv_capability_probe_is_cached(self):
        """Repeated decode-op construction probes a wrapper type only once."""
        config = self._create_config()
        inputs = self._create_cuda_graph_inputs(
            2,
            [64, 128],
            config.seq_size_per_block,
        )
        capability_probe = py_flashinfer_mha._decode_plan_supports_disable_split_kv
        capability_probe.cache_clear()
        try:
            first_op = PyFlashinferDecodeAttnOp(config.attn_configs, inputs)
            second_op = PyFlashinferDecodeAttnOp(config.attn_configs, inputs)
            cache_info = capability_probe.cache_info()
            self.assertEqual(cache_info.misses, 1)
            self.assertEqual(cache_info.hits, 1)
            self.assertEqual(
                first_op._decode_plan_supports_disable_split_kv,
                second_op._decode_plan_supports_disable_split_kv,
            )
        finally:
            capability_probe.cache_clear()

    def test_tensor_core_graph_omits_unsupported_disable_split_kv(self):
        """Legacy FlashInfer plans keep their native CUDA graph signature."""
        config = self._create_config()
        inputs = self._create_cuda_graph_inputs(
            2,
            [64, 128],
            config.seq_size_per_block,
        )
        attn_op = PyFlashinferDecodeAttnOp(config.attn_configs, inputs)
        self.assertTrue(attn_op.use_tensor_core)
        attn_op._decode_plan_supports_disable_split_kv = False
        attn_op.set_params(rtp_llm_ops.FlashInferMlaAttnParams())

        with mock.patch.object(
            attn_op.decode_wrapper,
            "plan",
            wraps=attn_op.decode_wrapper.plan,
        ) as plan_mock:
            attn_op.prepare(inputs)

        self.assertNotIn("disable_split_kv", plan_mock.call_args.kwargs)
        self.assertIsNotNone(attn_op._cuda_graph_plan_info)

    def test_set_params_rejects_rebind_after_cuda_graph_prepare(self):
        """Graph wrapper buffers cannot be rebound after prepare()."""
        config = self._create_config(head_num=32, head_num_kv=32)
        inputs = self._create_cuda_graph_inputs(
            2,
            [64, 128],
            config.seq_size_per_block,
        )
        attn_op = PyFlashinferDecodeAttnOp(config.attn_configs, inputs)
        original_params = rtp_llm_ops.FlashInferMlaAttnParams()
        attn_op.set_params(original_params)
        attn_op.prepare(inputs)

        with self.assertRaisesRegex(RuntimeError, "must be set before prepare"):
            attn_op.set_params(rtp_llm_ops.FlashInferMlaAttnParams())
        self.assertIs(attn_op.fmha_params, original_params)

    def test_prepare_rejects_rebind_after_cuda_graph_prepare(self):
        """A second prepare() cannot silently rebind graph wrapper buffers."""
        config = self._create_config(head_num=32, head_num_kv=32)
        inputs = self._create_cuda_graph_inputs(
            2,
            [64, 128],
            config.seq_size_per_block,
        )
        attn_op = PyFlashinferDecodeAttnOp(config.attn_configs, inputs)
        fmha_params = rtp_llm_ops.FlashInferMlaAttnParams()
        attn_op.set_params(fmha_params)
        attn_op.prepare(inputs)
        graph_buffer_pointers = (
            fmha_params.decode_page_indptr_d.data_ptr(),
            fmha_params.page_indice_d.data_ptr(),
            fmha_params.paged_kv_last_page_len_d.data_ptr(),
        )

        with self.assertRaisesRegex(RuntimeError, "use prepare_for_cuda_graph_replay"):
            attn_op.prepare(inputs)
        self._assert_graph_buffer_pointers(attn_op, fmha_params, graph_buffer_pointers)

    def test_replay_rejects_device_only_sequence_length_mirror(self):
        """Replay requires the host mirror supplied by CudaGraphRunner."""
        config = self._create_config(head_num=32, head_num_kv=32)
        capture_bs = 4
        capture_inputs = self._create_cuda_graph_inputs(
            capture_bs,
            [64, 128, 256, 512],
            config.seq_size_per_block,
        )
        attn_op = PyFlashinferDecodeAttnOp(config.attn_configs, capture_inputs)
        self.assertFalse(attn_op.use_tensor_core)
        fmha_params = rtp_llm_ops.FlashInferMlaAttnParams()
        attn_op.set_params(fmha_params)
        attn_op.prepare(capture_inputs)

        run_seq_lens = [100, 200]
        run_inputs = self._create_cuda_graph_inputs(
            capture_bs,
            run_seq_lens,
            config.seq_size_per_block,
            active_batch_size=len(run_seq_lens),
        )
        run_inputs.sequence_lengths_plus_1_device = (
            run_inputs.sequence_lengths.to(device="cuda", dtype=torch.int32) + 1
        )
        run_inputs.sequence_lengths = torch.empty(0, dtype=torch.int32, device="cuda")
        # The graph contract belongs to the captured op, not mutable replay input.
        run_inputs.is_cuda_graph = False
        with self.assertRaisesRegex(
            ValueError, "requires non-empty sequence_lengths covering all padding slots"
        ):
            attn_op.prepare_for_cuda_graph_replay(run_inputs)

    def test_replay_rejects_unbound_cuda_core_graph_op(self):
        """CUDA-core graph replay requires a completed initial prepare()."""
        config = self._create_config(head_num=32, head_num_kv=32)
        inputs = self._create_cuda_graph_inputs(
            2,
            [64, 128],
            config.seq_size_per_block,
        )
        graph_op = PyFlashinferDecodeAttnOp(config.attn_configs, inputs)
        graph_op.set_params(rtp_llm_ops.FlashInferMlaAttnParams())
        with self.assertRaisesRegex(RuntimeError, r"requires prepare\(\)"):
            graph_op.prepare_for_cuda_graph_replay(inputs)

    def test_replay_rejects_changed_graph_buffer_alias(self):
        """Replay fails fast if a dependency stops preserving bound storage."""
        config = self._create_config(head_num=32, head_num_kv=32)
        inputs = self._create_cuda_graph_inputs(
            2,
            [64, 128],
            config.seq_size_per_block,
        )
        attn_op = PyFlashinferDecodeAttnOp(config.attn_configs, inputs)
        attn_op.set_params(rtp_llm_ops.FlashInferMlaAttnParams())
        attn_op.prepare(inputs)
        original_last_page_lengths = (
            attn_op.fmha_params.paged_kv_last_page_len_h.clone()
        )
        original_plan_topology = attn_op._cuda_core_plan_page_indptr_h.clone()
        attn_op.decode_wrapper._paged_kv_indices_buf = (
            attn_op.decode_wrapper._paged_kv_indices_buf.clone()
        )
        same_topology_inputs = self._create_cuda_graph_inputs(
            2,
            [63, 127],
            config.seq_size_per_block,
        )
        with self.assertRaisesRegex(RuntimeError, "page indices buffer no longer"):
            attn_op.prepare_for_cuda_graph_replay(same_topology_inputs)
        self.assertTrue(
            torch.equal(
                attn_op.fmha_params.paged_kv_last_page_len_h,
                original_last_page_lengths,
            )
        )
        self.assertTrue(
            torch.equal(
                attn_op._cuda_core_plan_page_indptr_h,
                original_plan_topology,
            )
        )

    def test_replay_reports_missing_graph_wrapper_buffer(self):
        """A dependency contract mismatch reports a graph-specific error."""
        config = self._create_config(head_num=32, head_num_kv=32)
        inputs = self._create_cuda_graph_inputs(
            2,
            [64, 128],
            config.seq_size_per_block,
        )
        attn_op = PyFlashinferDecodeAttnOp(config.attn_configs, inputs)
        attn_op.set_params(rtp_llm_ops.FlashInferMlaAttnParams())
        attn_op.prepare(inputs)
        del attn_op.decode_wrapper._paged_kv_indices_buf

        with self.assertRaisesRegex(RuntimeError, "_paged_kv_indices_buf"):
            attn_op.prepare_for_cuda_graph_replay(inputs)

    def test_replay_rejects_batch_size_change_before_refresh(self):
        """A wrong replay batch cannot resize graph-bound metadata buffers."""
        config = self._create_config(head_num=32, head_num_kv=32)
        capture_inputs = self._create_cuda_graph_inputs(
            2,
            [64, 128],
            config.seq_size_per_block,
        )
        attn_op = PyFlashinferDecodeAttnOp(config.attn_configs, capture_inputs)
        fmha_params = rtp_llm_ops.FlashInferMlaAttnParams()
        attn_op.set_params(fmha_params)
        attn_op.prepare(capture_inputs)
        original_indptr = fmha_params.decode_page_indptr_h.clone()

        wrong_batch_inputs = self._create_cuda_graph_inputs(
            1,
            [129],
            config.seq_size_per_block,
        )
        with self.assertRaisesRegex(RuntimeError, "replay=1, bound=2"):
            attn_op.prepare_for_cuda_graph_replay(wrong_batch_inputs)
        self.assertTrue(torch.equal(fmha_params.decode_page_indptr_h, original_indptr))

        replay_inputs = self._create_cuda_graph_inputs(
            2,
            [129, 193],
            config.seq_size_per_block,
        )
        attn_op.prepare_for_cuda_graph_replay(replay_inputs)
        self.assertEqual(fmha_params.decode_page_indptr_h.numel(), 3)

    def test_replay_rejects_non_vector_input_lengths(self):
        """A tensor with the right element count cannot masquerade as a batch."""
        config = self._create_config(head_num=32, head_num_kv=32)
        capture_inputs = self._create_cuda_graph_inputs(
            2,
            [64, 128],
            config.seq_size_per_block,
        )
        attn_op = PyFlashinferDecodeAttnOp(config.attn_configs, capture_inputs)
        fmha_params = rtp_llm_ops.FlashInferMlaAttnParams()
        attn_op.set_params(fmha_params)
        attn_op.prepare(capture_inputs)
        original_indptr = fmha_params.decode_page_indptr_h.clone()

        invalid_inputs = self._create_cuda_graph_inputs(
            2,
            [129, 193],
            config.seq_size_per_block,
        )
        invalid_inputs.input_lengths = invalid_inputs.input_lengths.reshape(1, 2)
        with self.assertRaisesRegex(ValueError, "input_lengths must be 1-D"):
            attn_op.prepare_for_cuda_graph_replay(invalid_inputs)
        self.assertTrue(torch.equal(fmha_params.decode_page_indptr_h, original_indptr))

    def test_host_fill_validates_kv_cache_block_table_shape(self):
        """Malformed block tables fail before reaching the C++ fill helper."""
        config = self._create_config(head_num=32, head_num_kv=32)
        with self.subTest(shape="missing"):
            inputs = PyAttentionInputs()
            with self.assertRaisesRegex(ValueError, "requires a KV cache block table"):
                py_flashinfer_mha._validated_decode_host_block_table(
                    inputs,
                    2,
                    torch.tensor([64, 128], dtype=torch.int32),
                    config.seq_size_per_block,
                )

        with self.subTest(shape="empty"):
            inputs = self._create_cuda_graph_inputs(
                2,
                [64, 128],
                config.seq_size_per_block,
            )
            inputs.kv_cache_kernel_block_id = torch.empty(0, dtype=torch.int32)
            inputs.kv_cache_kernel_block_id_device = torch.empty(
                0, dtype=torch.int32, device="cuda"
            )
            attn_op = PyFlashinferDecodeAttnOp(config.attn_configs, inputs)
            with self.assertRaisesRegex(ValueError, "requires a KV cache block table"):
                attn_op.prepare(inputs)

        with self.subTest(shape="one-dimensional"):
            inputs = self._create_cuda_graph_inputs(
                2,
                [64, 128],
                config.seq_size_per_block,
            )
            inputs.kv_cache_kernel_block_id = torch.arange(4, dtype=torch.int32)
            attn_op = PyFlashinferDecodeAttnOp(config.attn_configs, inputs)
            with self.assertRaisesRegex(ValueError, "must be 2-D, got 1-D"):
                attn_op.prepare(inputs)

        with self.subTest(shape="insufficient rows"):
            inputs = self._create_cuda_graph_inputs(
                2,
                [64, 128],
                config.seq_size_per_block,
            )
            inputs.kv_cache_kernel_block_id = torch.zeros((1, 4), dtype=torch.int32)
            attn_op = PyFlashinferDecodeAttnOp(config.attn_configs, inputs)
            with self.assertRaisesRegex(
                ValueError, "fewer rows than the batch: rows=1, input_lengths=2"
            ):
                attn_op.prepare(inputs)

        with self.subTest(shape="insufficient columns"):
            inputs = self._create_cuda_graph_inputs(
                2,
                [65, 128],
                config.seq_size_per_block,
            )
            inputs.kv_cache_kernel_block_id = torch.zeros((2, 1), dtype=torch.int32)
            attn_op = PyFlashinferDecodeAttnOp(config.attn_configs, inputs)
            with self.assertRaisesRegex(
                ValueError, "fewer columns.*columns=1, required=2"
            ):
                attn_op.prepare(inputs)

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
            self.assertFalse(capture_call.args[1].is_cuda)
            self.assertFalse(capture_call.args[2].is_cuda)
            self.assertTrue(capture_call.kwargs["non_blocking"])
            self.assertEqual(
                "disable_split_kv" in capture_call.kwargs,
                attn_op._decode_plan_supports_disable_split_kv,
            )
            if attn_op._decode_plan_supports_disable_split_kv:
                self.assertTrue(capture_call.kwargs["disable_split_kv"])
            self.assertTrue(hasattr(attn_op.decode_wrapper, "_qo_indptr_buf"))
            self.assertEqual(
                attn_op.decode_wrapper._qo_indptr_buf.numel(), capture_bs + 1
            )
            qo_indptr_ptr = attn_op.decode_wrapper._qo_indptr_buf.data_ptr()
            self._assert_plan_dtypes(capture_call, attn_op)
            graph_buffer_pointers = (
                fmha_params.decode_page_indptr_d.data_ptr(),
                fmha_params.page_indice_d.data_ptr(),
                fmha_params.paged_kv_last_page_len_d.data_ptr(),
            )
            self._assert_graph_buffer_pointers(
                attn_op,
                fmha_params,
                graph_buffer_pointers,
                qo_indptr_ptr=qo_indptr_ptr,
            )

            plan_mock.reset_mock()
            run_seq_lens = [100, 200, 300, 400, 64, 128, 256, 512]
            run_inputs = self._create_cuda_graph_inputs(
                capture_bs,
                run_seq_lens,
                config.seq_size_per_block,
            )
            # Tensor-core graph replay must normalize CUDA-resident metadata to
            # the host mirrors consumed by plan().
            run_inputs.sequence_lengths = run_inputs.sequence_lengths.cuda()
            attn_op.prepare_for_cuda_graph_replay(run_inputs)
            self.assertEqual(plan_mock.call_count, 1)
            self._assert_plan_dtypes(plan_mock.call_args, attn_op)
            expected = self._expected_page_metadata(
                run_seq_lens,
                config.seq_size_per_block,
                batch_size=capture_bs,
            )
            self._assert_page_metadata(fmha_params, expected)
            self._assert_graph_buffer_pointers(
                attn_op,
                fmha_params,
                graph_buffer_pointers,
                qo_indptr_ptr=qo_indptr_ptr,
            )
            attn_op.prepare_for_cuda_graph_replay(run_inputs)
            self.assertEqual(plan_mock.call_count, 2)
            self._assert_plan_dtypes(plan_mock.call_args, attn_op)
            self._assert_graph_buffer_pointers(
                attn_op,
                fmha_params,
                graph_buffer_pointers,
                qo_indptr_ptr=qo_indptr_ptr,
            )

        self.assertEqual(attn_op.decode_wrapper._fixed_batch_size, capture_bs)

    def test_tensor_core_graph_rejects_buffer_rebind(self):
        """Tensor-core graph buffers share the same one-time binding contract."""
        config = self._create_config()
        inputs = self._create_cuda_graph_inputs(
            2,
            [64, 128],
            config.seq_size_per_block,
        )
        attn_op = PyFlashinferDecodeAttnOp(config.attn_configs, inputs)
        self.assertTrue(attn_op.use_tensor_core)
        original_params = rtp_llm_ops.FlashInferMlaAttnParams()
        attn_op.set_params(original_params)
        attn_op.prepare(inputs)

        with self.assertRaisesRegex(RuntimeError, "must be set before prepare"):
            attn_op.set_params(rtp_llm_ops.FlashInferMlaAttnParams())
        with self.assertRaisesRegex(RuntimeError, "use prepare_for_cuda_graph_replay"):
            attn_op.prepare(inputs)
        self.assertIs(attn_op.fmha_params, original_params)

        attn_op.decode_wrapper._qo_indptr_buf = (
            attn_op.decode_wrapper._qo_indptr_buf.clone()
        )
        replay_inputs = self._create_cuda_graph_inputs(
            2,
            [129, 193],
            config.seq_size_per_block,
        )
        with self.assertRaisesRegex(RuntimeError, "query indptr buffer no longer"):
            attn_op.prepare_for_cuda_graph_replay(replay_inputs)

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
        self._assert_cuda_core_graph_mode(attn_op)
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
                capture_expected = self._expected_page_metadata(
                    capture_seq_lens,
                    config.seq_size_per_block,
                    batch_size=capture_bs,
                )
                self._assert_page_metadata(fmha_params, capture_expected)
                self.assertNotEqual(
                    attn_op._cuda_core_plan_page_indptr_h.data_ptr(),
                    fmha_params.decode_page_indptr_h.data_ptr(),
                )
                capture_call = plan_mock.call_args
                self._assert_plan_dtypes(capture_call, attn_op)
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
                self._assert_plan_dtypes(changed_call, attn_op)
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

            # A skipped topology-only replan must still match the reference.
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
            skipped_replan_output = attn_op.forward(q, kv_cache, fmha_params).clone()
            self._assert_active_output_matches_reference(
                skipped_replan_output,
                same_topology_inputs,
                same_topology_seq_lens,
                q,
                attn_op.q_dtype,
                k_cache,
                v_cache,
                config.seq_size_per_block,
            )

    def test_cuda_core_failed_plan_invalidates_topology_snapshot(self):
        """A failed topology replan cannot leave the old snapshot reusable."""
        config = self._create_config(head_num=32, head_num_kv=32)
        capture_inputs = self._create_cuda_graph_inputs(
            2,
            [64, 128],
            config.seq_size_per_block,
        )
        attn_op = PyFlashinferDecodeAttnOp(config.attn_configs, capture_inputs)
        attn_op.set_params(rtp_llm_ops.FlashInferMlaAttnParams())
        attn_op.prepare(capture_inputs)
        self.assertIsNotNone(attn_op._cuda_core_plan_page_indptr_h)

        replay_inputs = self._create_cuda_graph_inputs(
            2,
            [129, 193],
            config.seq_size_per_block,
        )
        with mock.patch.object(
            attn_op.decode_wrapper,
            "plan",
            side_effect=RuntimeError("injected plan failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "injected plan failure"):
                attn_op.prepare_for_cuda_graph_replay(replay_inputs)
        self.assertIsNone(attn_op._cuda_core_plan_page_indptr_h)

        with mock.patch.object(
            attn_op.decode_wrapper,
            "plan",
            wraps=attn_op.decode_wrapper.plan,
        ) as plan_mock:
            attn_op.prepare_for_cuda_graph_replay(replay_inputs)
        self.assertEqual(plan_mock.call_count, 1)
        self.assertIsNotNone(attn_op._cuda_core_plan_page_indptr_h)

    def test_replay_rejects_changed_flashinfer_launch_plan(self):
        """Replay fails if replanning selects a launch incompatible with capture."""
        config = self._create_config(head_num=32, head_num_kv=32)
        capture_inputs = self._create_cuda_graph_inputs(
            2,
            [64, 128],
            config.seq_size_per_block,
        )
        attn_op = PyFlashinferDecodeAttnOp(config.attn_configs, capture_inputs)
        attn_op.set_params(rtp_llm_ops.FlashInferMlaAttnParams())
        attn_op.prepare(capture_inputs)
        captured_plan_info = attn_op._cuda_graph_plan_info
        original_plan = attn_op.decode_wrapper.plan

        def incompatible_plan(*args, **kwargs):
            original_plan(*args, **kwargs)
            raw_plan_info = attn_op.decode_wrapper._plan_info
            plan_info = list(raw_plan_info)
            self.assertGreater(len(plan_info), 0)
            plan_info[0] += 1
            attn_op.decode_wrapper._plan_info = plan_info

        replay_inputs = self._create_cuda_graph_inputs(
            2,
            [129, 193],
            config.seq_size_per_block,
        )
        with mock.patch.object(
            attn_op.decode_wrapper,
            "plan",
            side_effect=incompatible_plan,
        ):
            with self.assertRaisesRegex(RuntimeError, "launch plan changed"):
                attn_op.prepare_for_cuda_graph_replay(replay_inputs)
        self.assertEqual(attn_op._cuda_graph_plan_info, captured_plan_info)
        self.assertIsNone(attn_op._cuda_core_plan_page_indptr_h)

    def test_prefix_replay_rejects_more_reuse_pages_than_capture(self):
        """Prefix graph replay cannot grow its capture-time reuse workspace."""
        config = self._create_config(head_num=32, head_num_kv=32)
        capture_inputs = self._create_cuda_graph_inputs(
            2,
            [64, 128],
            config.seq_size_per_block,
            use_prefix_lengths=True,
        )
        attn_op = PyFlashinferDecodeAttnOp(config.attn_configs, capture_inputs)
        attn_op.set_params(rtp_llm_ops.FlashInferMlaAttnParams())
        attn_op.prepare(capture_inputs)

        replay_inputs = self._create_cuda_graph_inputs(
            2,
            [193, 193],
            config.seq_size_per_block,
            use_prefix_lengths=True,
        )
        with self.assertRaisesRegex(RuntimeError, "Buffer reallocation required"):
            attn_op.prepare_for_cuda_graph_replay(replay_inputs)

    def _run_actual_graph_replay_matches_reference(
        self,
        use_prefix_lengths: bool,
        expect_tensor_core: bool,
        head_num_kv: int = 32,
        active_batch_size: Optional[int] = None,
        padding_block_id: int = 0,
    ) -> None:
        if not is_sm90():
            self.skipTest(
                "FlashInfer 0.6.9 real decode CUDA graph capture is not "
                "supported by the sm8x or sm12x test targets; metadata and "
                "replanning tests remain enabled on those platforms"
            )
        config = self._create_config(head_num=32, head_num_kv=head_num_kv)
        capture_bs = 2
        if use_prefix_lengths:
            # Prefix metadata also sizes FlashInfer's reuse-page workspace.
            capture_sequence_lengths = [64, 384]
        else:
            # CudaGraphRunner captures with near-max lengths, then replays the
            # graph with shorter live sequences.
            capture_sequence_lengths = [512, 512]
        capture_inputs = self._create_cuda_graph_inputs(
            capture_bs,
            capture_sequence_lengths,
            config.seq_size_per_block,
            use_prefix_lengths=use_prefix_lengths,
        )
        # Actual capture fixtures must not recycle a workspace still referenced
        # by another graph created earlier in this process.
        graph_workspace = torch.zeros(
            py_flashinfer_mha.DEFAULT_PY_FLASHINFER_WORKSPACE_SIZE_MB * 1024 * 1024,
            dtype=torch.uint8,
            device="cuda",
        )
        graph = None
        graph_output = None
        attn_op = None

        def cleanup_graph_resources():
            nonlocal graph, graph_output, attn_op, graph_workspace
            graph_output = None
            graph = None
            attn_op = None
            graph_workspace = None
            torch.cuda.synchronize()

        self.addCleanup(cleanup_graph_resources)
        with mock.patch.object(
            py_flashinfer_mha,
            "get_py_flashinfer_workspace_buffer",
            return_value=graph_workspace,
        ):
            attn_op = _CudaGraphTestDecodeAttnOp(config.attn_configs, capture_inputs)
        self.assertEqual(attn_op.use_tensor_core, expect_tensor_core)
        fmha_params = rtp_llm_ops.FlashInferMlaAttnParams()
        attn_op.set_params(fmha_params)
        attn_op.prepare(capture_inputs)

        replay_sequence_lengths = [129, 193]
        if active_batch_size is not None:
            replay_sequence_lengths = replay_sequence_lengths[:active_batch_size]
        replay_inputs = self._create_cuda_graph_inputs(
            capture_bs,
            replay_sequence_lengths,
            config.seq_size_per_block,
            active_batch_size=active_batch_size,
            padding_block_id=padding_block_id,
            use_prefix_lengths=use_prefix_lengths,
        )
        same_topology_sequence_lengths = [130, 194]
        if active_batch_size is not None:
            same_topology_sequence_lengths = same_topology_sequence_lengths[
                :active_batch_size
            ]
        same_topology_inputs = self._create_cuda_graph_inputs(
            capture_bs,
            same_topology_sequence_lengths,
            config.seq_size_per_block,
            active_batch_size=active_batch_size,
            padding_block_id=padding_block_id,
            use_prefix_lengths=use_prefix_lengths,
        )
        local_head_num = config.head_num // config.tp_size
        local_kv_head_num = config.head_num_kv // config.tp_size
        static_q = self._create_query_tensor(
            capture_bs,
            local_head_num,
            config.size_per_head,
        )
        total_blocks = max(
            int(inputs.kv_cache_kernel_block_id.max().item()) + 1
            for inputs in (capture_inputs, replay_inputs, same_topology_inputs)
        )
        kv_cache, k_cache, v_cache = self._create_kv_cache(
            total_blocks,
            config.seq_size_per_block,
            local_kv_head_num,
            config.size_per_head,
            dtype=self.cache_dtype(config.attn_configs),
        )

        torch.cuda.synchronize()
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            attn_op.forward(static_q, kv_cache, fmha_params)
        torch.cuda.current_stream().wait_stream(warmup_stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = attn_op.forward(static_q, kv_cache, fmha_params)

        attn_op.prepare_for_cuda_graph_replay(replay_inputs)
        graph.replay()
        torch.cuda.synchronize()
        self._assert_active_output_matches_reference(
            graph_output,
            replay_inputs,
            replay_sequence_lengths,
            static_q,
            attn_op.q_dtype,
            k_cache,
            v_cache,
            config.seq_size_per_block,
        )

        with mock.patch.object(
            attn_op.decode_wrapper,
            "plan",
            wraps=attn_op.decode_wrapper.plan,
        ) as plan_mock:
            attn_op.prepare_for_cuda_graph_replay(same_topology_inputs)
        self.assertEqual(plan_mock.call_count, 1 if expect_tensor_core else 0)
        graph.replay()
        torch.cuda.synchronize()
        self._assert_active_output_matches_reference(
            graph_output,
            same_topology_inputs,
            same_topology_sequence_lengths,
            static_q,
            attn_op.q_dtype,
            k_cache,
            v_cache,
            config.seq_size_per_block,
        )

    def test_cuda_core_actual_graph_replay_matches_reference(self):
        """Capture and replay CUDA-core decode after crossing a page boundary."""
        self._run_actual_graph_replay_matches_reference(
            use_prefix_lengths=False,
            expect_tensor_core=False,
            active_batch_size=1,
            padding_block_id=7,
        )

    def test_actual_tensor_core_graph_replay_matches_reference(self):
        """Capture and replay GQA tensor-core decode across a page boundary."""
        self._run_actual_graph_replay_matches_reference(
            use_prefix_lengths=False,
            expect_tensor_core=True,
            head_num_kv=8,
            active_batch_size=1,
            padding_block_id=7,
        )

    def test_cuda_core_prefix_actual_graph_replay_matches_reference(self):
        """Capture and replay the compatible prefix-length metadata contract."""
        if self.kv_cache_dtype == KvCacheDataType.FP8:
            self.skipTest("Prefix metadata is dtype-independent and covered by FP16")
        self._run_actual_graph_replay_matches_reference(
            use_prefix_lengths=True,
            expect_tensor_core=False,
            active_batch_size=1,
            padding_block_id=7,
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
                self._assert_page_metadata(fmha_params, changed_expected)
                output = attn_op.forward(q, kv_cache, fmha_params).clone()
                self._assert_active_output_matches_reference(
                    output,
                    changed_inputs,
                    changed_seq_lens,
                    q,
                    attn_op.q_dtype,
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
                output = attn_op.forward(q, kv_cache, fmha_params).clone()
                self._assert_active_output_matches_reference(
                    output,
                    crossed_page_inputs,
                    crossed_page_seq_lens,
                    q,
                    attn_op.q_dtype,
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

    def test_cuda_core_fp8_kv_replay_preserves_plan_dtype(self):
        """CUDA-core FP8 replay retains plan dtype and numerical accuracy."""
        config = self._create_config(head_num=32, head_num_kv=32)
        capture_inputs = self._create_cuda_graph_inputs(
            2,
            [64, 128],
            config.seq_size_per_block,
        )
        attn_op = PyFlashinferDecodeAttnOp(config.attn_configs, capture_inputs)
        self._assert_cuda_core_graph_mode(attn_op)
        fmha_params = rtp_llm_ops.FlashInferMlaAttnParams()
        attn_op.set_params(fmha_params)
        with mock.patch.object(
            attn_op.decode_wrapper,
            "plan",
            wraps=attn_op.decode_wrapper.plan,
        ) as plan_mock:
            attn_op.prepare(capture_inputs)
            self._assert_plan_dtypes(plan_mock.call_args, attn_op)
            self.assertEqual(attn_op.kv_dtype, torch.float8_e4m3fn)
            replay_sequence_lengths = [129, 193]
            replay_inputs = self._create_cuda_graph_inputs(
                2,
                replay_sequence_lengths,
                config.seq_size_per_block,
            )
            plan_mock.reset_mock()
            attn_op.prepare_for_cuda_graph_replay(replay_inputs)
            self.assertEqual(plan_mock.call_count, 1)
            self._assert_plan_dtypes(plan_mock.call_args, attn_op)

        local_head_num = config.head_num // config.tp_size
        local_kv_head_num = config.head_num_kv // config.tp_size
        q = self._create_query_tensor(
            len(replay_sequence_lengths),
            local_head_num,
            config.size_per_head,
        )
        total_blocks = self._calculate_total_blocks(
            replay_sequence_lengths,
            config.seq_size_per_block,
        )
        kv_cache, k_cache, v_cache = self._create_kv_cache(
            total_blocks,
            config.seq_size_per_block,
            local_kv_head_num,
            config.size_per_head,
            dtype=self.cache_dtype(config.attn_configs),
        )
        output = attn_op.forward(q, kv_cache, fmha_params)
        self._assert_active_output_matches_reference(
            output,
            replay_inputs,
            replay_sequence_lengths,
            q,
            attn_op.q_dtype,
            k_cache,
            v_cache,
            config.seq_size_per_block,
        )


if __name__ == "__main__":
    unittest.main()
