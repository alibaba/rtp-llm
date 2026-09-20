import gc
import logging
import math
import os
import sys
import unittest
from types import SimpleNamespace
from typing import List, NamedTuple, Optional
from unittest import mock
from unittest.mock import patch

import torch
from attention_ref import compute_flashinfer_decode_reference
from base_attention_test import BaseAttentionTest, compare_tensors

from rtp_llm.models_py.model_desc.minimax_m3 import _target_verify_impl_class
from rtp_llm.models_py.modules.factory.attention.cuda_impl import py_flashinfer_mha
from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
    PyFlashinferDecodeAttnOp,
    PyFlashinferDecodeImpl,
)
from rtp_llm.ops import KvCacheDataType
from rtp_llm.ops.compute_ops import (
    PyAttentionInputs,
    fill_mla_params,
    get_typemeta,
    rtp_llm_ops,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")


class TestPyFlashinferWorkspacePlanning(unittest.TestCase):
    def test_fa2_graph_plan_uses_exact_split_k_workspace(self) -> None:
        heads = 12
        padded_batch_size = 80
        cta_tile_q = 16
        head_dim = 256
        partial_rows = heads * padded_batch_size * cta_tile_q
        value_offset = 256
        value_bytes = partial_rows * head_dim * 4
        lse_offset = value_offset + value_bytes

        plan_info = [0] * 15
        plan_info[0] = padded_batch_size
        plan_info[3] = cta_tile_q
        plan_info[10] = value_offset
        plan_info[11] = lse_offset
        plan_info[14] = 1
        wrapper = SimpleNamespace(_backend="fa2", _plan_info=plan_info)

        expected = py_flashinfer_mha._round_workspace_size(
            lse_offset + partial_rows * 4
        )
        self.assertEqual(
            py_flashinfer_mha._planned_flashinfer_workspace_size(
                wrapper, heads, head_dim
            ),
            expected,
        )

    def test_fa3_graph_plan_needs_no_float_scratch(self) -> None:
        wrapper = SimpleNamespace(_backend="fa3", _plan_info=[0] * 8)
        self.assertEqual(
            py_flashinfer_mha._planned_flashinfer_workspace_size(wrapper, 12, 256),
            py_flashinfer_mha._FLASHINFER_WORKSPACE_ALIGNMENT,
        )

    def test_unknown_fa3_plan_keeps_flashinfer_default(self) -> None:
        wrapper = SimpleNamespace(_backend="fa3", _plan_info=[0] * 9)
        self.assertIsNone(
            py_flashinfer_mha._planned_flashinfer_workspace_size(wrapper, 12, 256)
        )

    def test_unknown_backend_keeps_flashinfer_default(self) -> None:
        wrapper = SimpleNamespace(_backend="future_backend", _plan_info=[])
        self.assertIsNone(
            py_flashinfer_mha._planned_flashinfer_workspace_size(wrapper, 12, 256)
        )

    def test_workspace_pool_is_size_aware(self) -> None:
        first = py_flashinfer_mha.get_py_flashinfer_workspace_buffer("cpu", 513)
        self.assertEqual(first.numel(), 768)
        py_flashinfer_mha.release_py_flashinfer_workspace_buffer(first)
        reused = py_flashinfer_mha.get_py_flashinfer_workspace_buffer("cpu", 513)
        self.assertIs(reused, first)
        py_flashinfer_mha.release_py_flashinfer_workspace_buffer(reused)


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

    def test_cuda_graph_prepare_uses_minimum_kv_length(self):
        config = self._create_config(
            head_num=32,
            head_num_kv=8,
            size_per_head=128,
            seq_size_per_block=128,
        )
        sequence_lengths = [256, 512]
        # The block table and the paged-KV buffers are sized for the real
        # sequences, but capture plans the smallest possible schedule: one page
        # per request. Replay below has to replace that plan.
        attn_inputs = self._create_attention_inputs(
            batch_size=2,
            sequence_lengths=sequence_lengths,
            seq_size_per_block=config.seq_size_per_block,
        )
        attn_inputs.is_cuda_graph = True
        # sequence_lengths holds the cached length, so zero means a single live
        # KV token: exactly one page whose last page holds one token.
        attn_inputs.sequence_lengths = torch.zeros(
            2, dtype=torch.int32, device="cpu"
        ).pin_memory()

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
        # Replay carries the real KV lengths; the captured one-page plan must be
        # replaced while the graph-owned paged-KV buffers keep their storage.
        attn_inputs.sequence_lengths = (
            torch.tensor(sequence_lengths, dtype=torch.int32, device="cpu") - 1
        ).pin_memory()
        attn_op.prepare_for_cuda_graph_replay(attn_inputs)
        torch.cuda.synchronize()
        self.assertEqual(len(plan_calls), 1)
        self.assertEqual(
            params.decode_page_indptr_d.cpu().tolist()[: len(sequence_lengths) + 1],
            [0, 2, 6],
        )
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
        attn_inputs.sequence_lengths_plus_1_device = torch.tensor(
            sequence_lengths, dtype=torch.int32, device=self.device
        )
        real_cached_lengths = attn_inputs.sequence_lengths
        # Capture the one-page schedule: the block table stays sized for the
        # full 638-page sequence, so replay can widen the plan in place.
        attn_inputs.sequence_lengths = torch.zeros(
            1, dtype=torch.int32, device="cpu"
        ).pin_memory()

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
            dtype=self.cache_dtype(config.attn_configs),
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
            attn_inputs.sequence_lengths_plus_1_device,
            attn_inputs.kv_cache_kernel_block_id_device,
            config.seq_size_per_block,
        )
        graph.replay()
        torch.cuda.synchronize()
        stale_plan_output = graph_output.clone()

        attn_inputs.sequence_lengths = real_cached_lengths
        attn_op.prepare_for_cuda_graph_replay(attn_inputs)
        graph.replay()
        torch.cuda.synchronize()
        replay_output = graph_output.clone()

        block_ids = self._generate_block_id_list(
            attn_inputs, sequence_lengths, config.seq_size_per_block
        )
        ref_output = compute_flashinfer_decode_reference(
            q,
            k_cache.to(q.dtype),
            v_cache.to(q.dtype),
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

    def test_target_verify_fa4_is_independent_of_flashinfer(self):
        try:
            from flash_attn.cute import flash_attn_varlen_func as _fa4_varlen  # noqa: F401
        except (ImportError, OSError):
            self.skipTest("flash-attn-4 is not installed")
        _target_verify_impl_class.cache_clear()
        config = self._create_config(
            head_num=32,
            head_num_kv=8,
            size_per_head=128,
            seq_size_per_block=128,
            data_type="bf16",
        )
        config.attn_configs.kv_cache_dtype = KvCacheDataType.FP8

        with (
            patch.dict(
                os.environ,
                {"RTP_LLM_M3_TARGET_VERIFY_BACKEND": "fa4"},
            ),
            patch.object(torch.cuda, "get_device_capability", return_value=(10, 0)),
            patch.object(
                py_flashinfer_mha,
                "get_py_flashinfer_workspace_buffer",
                side_effect=AssertionError(
                    "FA4 must not allocate FlashInfer workspace"
                ),
            ) as get_workspace,
            patch.object(
                py_flashinfer_mha,
                "BatchDecodeWithPagedKVCacheWrapper",
                side_effect=AssertionError("FA4 must not construct decode wrapper"),
            ) as wrapper_class,
        ):
            impl_class = _target_verify_impl_class()
            impl = object.__new__(impl_class)
            attn_op = impl._create_fmha_impl(config.attn_configs)

        self.assertEqual(attn_op._backend, "fa4")
        self.assertNotIsInstance(attn_op, PyFlashinferDecodeAttnOp)
        self.assertFalse(issubclass(impl_class, PyFlashinferDecodeImpl))
        self.assertFalse(hasattr(attn_op, "g_workspace_buffer"))
        self.assertFalse(hasattr(attn_op, "decode_wrapper"))
        get_workspace.assert_not_called()
        wrapper_class.assert_not_called()

    def test_target_verify_flashinfer_keeps_flashinfer_resources(self):
        config = self._create_config(
            head_num=32,
            head_num_kv=8,
            size_per_head=128,
            seq_size_per_block=128,
            data_type="bf16",
        )
        workspace = torch.empty(1, dtype=torch.uint8, device=self.device)
        wrapper = object()

        with (
            patch.dict(
                os.environ,
                {"RTP_LLM_M3_TARGET_VERIFY_BACKEND": "flashinfer"},
            ),
            patch.object(
                py_flashinfer_mha,
                "get_py_flashinfer_workspace_buffer",
                return_value=workspace,
            ) as get_workspace,
            patch.object(
                py_flashinfer_mha,
                "BatchDecodeWithPagedKVCacheWrapper",
                return_value=wrapper,
            ) as wrapper_class,
            patch.object(
                py_flashinfer_mha,
                "release_py_flashinfer_workspace_buffer",
            ) as release_workspace,
        ):
            impl_class = _target_verify_impl_class()
            impl = object.__new__(impl_class)
            attn_op = impl._create_fmha_impl(config.attn_configs)

            self.assertEqual(attn_op._backend, "flashinfer")
            self.assertIsInstance(attn_op, PyFlashinferDecodeAttnOp)
            self.assertIs(attn_op.g_workspace_buffer, workspace)
            self.assertIs(attn_op.decode_wrapper, wrapper)
            get_workspace.assert_called_once_with()
            wrapper_class.assert_called_once_with(
                workspace,
                "HND",
                use_tensor_cores=attn_op.use_tensor_core,
            )
            del attn_op
            gc.collect()
            release_workspace.assert_called_once_with(workspace)

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


class TestPyFlashinferDecodeCudaGraph(BaseAttentionTest):
    """Test CUDA graph buffer management for PyFlashinferDecodeAttnOp.

    These tests exercise the Python prepare/replay boundary. End-to-end CUDA
    graph capture and replay remains covered by the model smoke test.
    """

    def test_prepare_cuda_graph_supports_legacy_rope_binding(self):
        class FakeFmhaImpl:
            def __init__(self):
                self.prepare_calls = 0

            def prepare_for_cuda_graph_replay(self, attn_inputs):
                self.prepare_calls += 1

        class LegacyRopeImpl:
            def __init__(self, kv_cache_offset):
                self.kv_cache_offset = kv_cache_offset
                self.prepare_calls = 0

            def prepare(self, attn_inputs):
                self.prepare_calls += 1
                return SimpleNamespace(kv_cache_offset=self.kv_cache_offset)

        old_offset = torch.zeros((2, 2), dtype=torch.int32)
        new_offset = torch.arange(4, dtype=torch.int32).reshape(2, 2)
        impl = PyFlashinferDecodeImpl.__new__(PyFlashinferDecodeImpl)
        impl.need_rope_kv_cache = True
        impl.fmha_impl = FakeFmhaImpl()
        impl.rope_impl = LegacyRopeImpl(new_offset)
        impl.rope_params = SimpleNamespace(kv_cache_offset=old_offset)

        impl.prepare_cuda_graph(PyAttentionInputs())

        self.assertEqual(impl.fmha_impl.prepare_calls, 1)
        self.assertEqual(impl.rope_impl.prepare_calls, 1)
        self.assertTrue(torch.equal(impl.rope_params.kv_cache_offset, new_offset))
        self.assertFalse(hasattr(impl.rope_params, "sequence_lengths"))

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
            # Initial graph preparation replans after workspace compaction.
            self.assertEqual(plan_mock.call_count, 2)
            capture_call = plan_mock.call_args
            self.assertFalse(capture_call.args[0].is_cuda)
            self.assertFalse(capture_call.args[1].is_cuda)
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
                # Initial graph preparation replans after workspace compaction.
                self.assertEqual(plan_mock.call_count, 2)
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
