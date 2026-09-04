import logging
import unittest
from types import SimpleNamespace
from typing import List
from unittest.mock import patch

import torch
from attention_ref import compute_flashinfer_decode_reference
from base_attention_test import BaseAttentionTest, compare_tensors

from rtp_llm.models_py.modules.factory.attention import attn_factory
from rtp_llm.models_py.modules.factory.attention.cuda_impl.xqa import XQAImpl
from rtp_llm.ops import RopeStyle
from rtp_llm.ops.compute_ops import PyAttentionInputs, XQAAttnOp, XQAParams
from rtp_llm.ops.fused_rope_kvcache_op import (
    DecodeRopeContractError,
    FusedRopeAttnParams,
    FusedRopeKVCacheDecodeOp,
)
from rtp_llm.test.utils.cuda_graph_util import record_cuda_graph

logging.basicConfig(level=logging.INFO, format="%(message)s")


class TestXQAAttnOp(BaseAttentionTest):
    """Test suite for XQAAttnOp with correctness verification and support testing"""

    def _create_attention_inputs(
        self,
        batch_size: int,
        sequence_lengths: List[int],
        seq_size_per_block: int,
    ) -> PyAttentionInputs:
        """Helper to create PyAttentionInputs for decode"""
        return self._create_attention_inputs_base(
            batch_size=batch_size,
            sequence_lengths=sequence_lengths,
            seq_size_per_block=seq_size_per_block,
        )

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

        # Create XQAAttnOp instance
        attn_op = XQAAttnOp(config.attn_configs)

        # Test support function
        is_supported = attn_op.support(attn_inputs)
        logging.info(f"XQAAttnOp support check: {is_supported}")

        if not is_supported:
            logging.warning(
                f"XQAAttnOp does not support this configuration, skipping correctness test"
            )
            return

        # Prepare parameters
        params_base = attn_op.prepare(attn_inputs)
        # Cast to XQAParams for forward call
        params = XQAParams() if not isinstance(params_base, XQAParams) else params_base

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

        # Forward pass through XQAAttnOp
        output = attn_op.forward(q, kv_cache, params)

        # XQA output shape: [batch_size, head_num * head_dim]
        # Need to reshape to [batch_size, head_num, head_dim] to match reference
        output = output.reshape(batch_size, local_head_num, config.size_per_head)

        # Generate block_id_list from attn_inputs for reference computation
        block_id_list = self._generate_block_id_list(
            attn_inputs, sequence_lengths, config.seq_size_per_block
        )

        # Compute reference outputs using flashinfer
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
            name=f"XQA Decode output (batch={batch_size}, seq_lens={sequence_lengths})",
        )

        logging.info(
            f"✓ Test passed: batch_size={batch_size}, sequence_lengths={sequence_lengths}"
        )

    def test_support(self):
        """Test XQAAttnOp support function comprehensively

        Based on CudaXqa.cc supportXqa function:
        - input_type: BF16 or FP16
        - output_type: BF16, FP16, or FP8_E4M3
        - kv_cache_type: BF16, FP16, or FP8_E4M3
        - group_size <= 16
        - head_dim: 64, 128, or 256
        - page_size: 16, 32, 64, or 128
        """
        logging.info("\n=== Testing XQAAttnOp support() functionality ===")

        # Test SUPPORTED configurations
        logging.info("\n--- Testing SUPPORTED configurations ---")

        supported_cases = [
            # (head_num, head_num_kv, size_per_head, seq_size_per_block, data_type, description)
            (
                32,
                8,
                128,
                64,
                "fp16",
                "Standard config: GQA, head_dim=128, page_size=64",
            ),
            (32, 8, 256, 64, "fp16", "Large head_dim: 256"),
            (32, 8, 64, 64, "fp16", "Small head_dim: 64"),
            (32, 32, 128, 64, "fp16", "MHA: group_size=1"),
            (32, 2, 128, 64, "fp16", "Large group_size: 16"),
            (32, 8, 128, 16, "fp16", "Small page_size: 16"),
            (32, 8, 128, 32, "fp16", "page_size: 32"),
            (32, 8, 128, 128, "fp16", "Large page_size: 128"),
            (64, 4, 128, 64, "fp16", "GQA-16: group_size=16"),
            (128, 8, 256, 128, "fp16", "Large head_num with head_dim=256"),
        ]

        supported_count = 0
        for (
            head_num,
            head_num_kv,
            size_per_head,
            seq_size_per_block,
            data_type,
            desc,
        ) in supported_cases:
            config = self._create_config(
                head_num=head_num,
                head_num_kv=head_num_kv,
                size_per_head=size_per_head,
                seq_size_per_block=seq_size_per_block,
                data_type=data_type,
            )
            attn_inputs = self._create_attention_inputs(
                batch_size=1,
                sequence_lengths=[128],
                seq_size_per_block=seq_size_per_block,
            )

            attn_op = XQAAttnOp(config.attn_configs)
            is_supported = attn_op.support(attn_inputs)

            group_size = head_num // head_num_kv
            logging.info(
                f"  {desc}\n"
                f"    head_num={head_num}, head_num_kv={head_num_kv}, group_size={group_size}\n"
                f"    head_dim={size_per_head}, page_size={seq_size_per_block}, dtype={data_type}\n"
                f"    → Support: {is_supported} {'✓' if is_supported else '✗ UNEXPECTED'}"
            )

            if is_supported:
                supported_count += 1
            else:
                logging.warning(f"    ⚠️  Expected SUPPORTED but got NOT SUPPORTED")

        logging.info(f"\nSupported cases: {supported_count}/{len(supported_cases)}")

        # Test UNSUPPORTED configurations
        logging.info("\n--- Testing UNSUPPORTED configurations ---")

        unsupported_cases = [
            # (head_num, head_num_kv, size_per_head, seq_size_per_block, data_type, description)
            (32, 1, 128, 64, "fp16", "group_size=32 > 16: UNSUPPORTED"),
            (64, 1, 128, 64, "fp16", "group_size=64 > 16: UNSUPPORTED"),
            (32, 8, 96, 64, "fp16", "head_dim=96 not in {64,128,256}: UNSUPPORTED"),
            (32, 8, 192, 64, "fp16", "head_dim=192 not in {64,128,256}: UNSUPPORTED"),
            (32, 8, 512, 64, "fp16", "head_dim=512 not in {64,128,256}: UNSUPPORTED"),
            (32, 8, 128, 8, "fp16", "page_size=8 not in {16,32,64,128}: UNSUPPORTED"),
            (32, 8, 128, 48, "fp16", "page_size=48 not in {16,32,64,128}: UNSUPPORTED"),
            (
                32,
                8,
                128,
                256,
                "fp16",
                "page_size=256 not in {16,32,64,128}: UNSUPPORTED",
            ),
        ]

        unsupported_count = 0
        for (
            head_num,
            head_num_kv,
            size_per_head,
            seq_size_per_block,
            data_type,
            desc,
        ) in unsupported_cases:
            config = self._create_config(
                head_num=head_num,
                head_num_kv=head_num_kv,
                size_per_head=size_per_head,
                seq_size_per_block=seq_size_per_block,
                data_type=data_type,
            )
            attn_inputs = self._create_attention_inputs(
                batch_size=1,
                sequence_lengths=[128],
                seq_size_per_block=seq_size_per_block,
            )

            attn_op = XQAAttnOp(config.attn_configs)
            is_supported = attn_op.support(attn_inputs)

            group_size = head_num // head_num_kv
            logging.info(
                f"  {desc}\n"
                f"    head_num={head_num}, head_num_kv={head_num_kv}, group_size={group_size}\n"
                f"    head_dim={size_per_head}, page_size={seq_size_per_block}, dtype={data_type}\n"
                f"    → Support: {is_supported} {'✗ UNEXPECTED' if is_supported else '✓'}"
            )

            if not is_supported:
                unsupported_count += 1
            else:
                logging.warning(f"    ⚠️  Expected UNSUPPORTED but got SUPPORTED")

        logging.info(
            f"\nUnsupported cases correctly rejected: {unsupported_count}/{len(unsupported_cases)}"
        )

        # Test boundary cases
        logging.info("\n--- Testing BOUNDARY cases ---")

        boundary_cases = [
            # Edge cases for group_size
            (
                32,
                2,
                128,
                64,
                "fp16",
                "group_size=16 (max allowed): SHOULD BE SUPPORTED",
            ),
            (
                34,
                2,
                128,
                64,
                "fp16",
                "group_size=17 (just over limit): SHOULD BE UNSUPPORTED",
            ),
            # Edge cases for head_dim
            (32, 8, 64, 64, "fp16", "head_dim=64 (min): SHOULD BE SUPPORTED"),
            (32, 8, 256, 64, "fp16", "head_dim=256 (max): SHOULD BE SUPPORTED"),
            # Edge cases for page_size
            (32, 8, 128, 16, "fp16", "page_size=16 (min): SHOULD BE SUPPORTED"),
            (32, 8, 128, 128, "fp16", "page_size=128 (max): SHOULD BE SUPPORTED"),
        ]

        for (
            head_num,
            head_num_kv,
            size_per_head,
            seq_size_per_block,
            data_type,
            desc,
        ) in boundary_cases:
            config = self._create_config(
                head_num=head_num,
                head_num_kv=head_num_kv,
                size_per_head=size_per_head,
                seq_size_per_block=seq_size_per_block,
                data_type=data_type,
            )
            attn_inputs = self._create_attention_inputs(
                batch_size=1,
                sequence_lengths=[128],
                seq_size_per_block=seq_size_per_block,
            )

            attn_op = XQAAttnOp(config.attn_configs)
            is_supported = attn_op.support(attn_inputs)

            group_size = head_num // head_num_kv
            logging.info(
                f"  {desc}\n"
                f"    head_num={head_num}, head_num_kv={head_num_kv}, group_size={group_size}\n"
                f"    head_dim={size_per_head}, page_size={seq_size_per_block}\n"
                f"    → Support: {is_supported}"
            )

        logging.info("\n=== XQAAttnOp support() testing completed ===")

    def test_support_functionality(self):
        """Test XQAAttnOp support function with various configurations"""
        logging.info("\n=== Testing XQAAttnOp support functionality ===")

        test_cases = [
            # (head_num, head_num_kv, size_per_head, batch_size, seq_lens, description)
            (32, 8, 128, 1, [128], "Single batch, standard config"),
            (32, 8, 128, 4, [64, 128, 256, 512], "Multi-batch, varying lengths"),
            (32, 32, 128, 2, [100, 200], "MHA config"),
            (32, 4, 128, 2, [100, 200], "GQA-8 config"),
            (32, 8, 256, 2, [100, 200], "256 head dim"),
            (32, 8, 64, 2, [100, 200], "64 head dim"),
        ]

        for (
            head_num,
            head_num_kv,
            size_per_head,
            batch_size,
            seq_lens,
            desc,
        ) in test_cases:
            logging.info(f"\n--- Testing: {desc} ---")
            config = self._create_config(
                head_num=head_num,
                head_num_kv=head_num_kv,
                size_per_head=size_per_head,
            )
            attn_inputs = self._create_attention_inputs(
                batch_size, seq_lens, config.seq_size_per_block
            )

            attn_op = XQAAttnOp(config.attn_configs)
            is_supported = attn_op.support(attn_inputs)

            logging.info(
                f"  Config: head_num={head_num}, head_num_kv={head_num_kv}, "
                f"size_per_head={size_per_head}"
            )
            logging.info(f"  Batch size: {batch_size}, seq_lens: {seq_lens}")
            logging.info(f"  Support result: {is_supported}")

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


class TestFusedRopeKVCacheDecodeSequenceLengths(BaseAttentionTest):
    """Test the shared decode-RoPE sequence-length buffer contract."""

    def _real_kernel_case(self, sequence_lengths: List[int]):
        config = self._create_config()
        config.attn_configs.need_rope_kv_cache = True
        config.attn_configs.max_seq_len = 128
        config.attn_configs.rope_config.style = RopeStyle.Base
        config.attn_configs.rope_config.dim = config.size_per_head
        config.attn_configs.rope_config.base = 10000
        config.attn_configs.rope_config.max_pos = 128

        inputs = self._create_attention_inputs_base(
            len(sequence_lengths), sequence_lengths, config.seq_size_per_block
        )
        inputs.is_cuda_graph = True

        op = FusedRopeKVCacheDecodeOp(config.attn_configs)
        params = op.prepare(inputs)
        qkv_width = (config.head_num + 2 * config.head_num_kv) * config.size_per_head
        qkv = torch.randn(
            len(sequence_lengths),
            qkv_width,
            dtype=torch.float16,
            device=self.device,
        )
        kv_cache, _, _ = self._create_kv_cache(
            self._calculate_total_blocks(sequence_lengths, config.seq_size_per_block),
            config.seq_size_per_block,
            config.head_num_kv,
            config.size_per_head,
        )
        return config, inputs, op, params, qkv, kv_cache

    @staticmethod
    def _base_rope_reference(
        qkv: torch.Tensor,
        positions: torch.Tensor,
        head_num: int,
        head_dim: int,
        rope_base: int,
    ) -> torch.Tensor:
        q = qkv[:, : head_num * head_dim].reshape(-1, head_num, head_dim)
        q_float = q.float()
        half_dim = head_dim // 2
        inv_freq = 1.0 / (
            rope_base
            ** (torch.arange(0, head_dim, 2, device=q.device).float() / head_dim)
        )
        freqs = positions.to(device=q.device, dtype=torch.float32)[:, None] * inv_freq
        cos = torch.cat((freqs.cos(), freqs.cos()), dim=-1)[:, None, :]
        sin = torch.cat((freqs.sin(), freqs.sin()), dim=-1)[:, None, :]
        rotated_half = torch.cat(
            (-q_float[..., half_dim:], q_float[..., :half_dim]), dim=-1
        )
        return (q_float * cos + rotated_half * sin).to(q.dtype)

    def _inputs(
        self,
        values: List[int],
        *,
        is_cuda_graph: bool = True,
        pinned: bool = True,
        dtype: torch.dtype = torch.int32,
        with_target: bool = True,
    ) -> PyAttentionInputs:
        inputs = PyAttentionInputs()
        inputs.is_cuda_graph = is_cuda_graph
        source = torch.tensor(values, dtype=dtype)
        inputs.sequence_lengths = source.pin_memory() if pinned else source
        if with_target:
            # Deliberately unrelated values: the op must mirror the authoritative
            # host tensor. CudaGraphRunner, not this op, owns padding-tail zeroing.
            inputs.sequence_lengths_plus_1_device = torch.full(
                (len(values),), 901, dtype=torch.int32, device=self.device
            )
        return inputs

    def _params(
        self,
        sequence_lengths: torch.Tensor,
        kv_cache_offset: torch.Tensor | None,
    ) -> FusedRopeAttnParams:
        empty = torch.empty(0, dtype=torch.int32, device=self.device)
        return FusedRopeAttnParams(
            kv_cache_offset=kv_cache_offset,
            kv_cache_offset_h=None,
            padding_offset=None,
            position_ids=None,
            cu_seqlens=empty,
            cu_kv_seqlens=empty,
            input_lengths=empty,
            prefix_lengths=empty,
            sequence_lengths=sequence_lengths,
            max_seq_len=0,
            max_prefix_length=0,
            context_total_kv_length=0,
            decode_plan=True,
            attn_type=torch.float16,
        )

    def test_replay_reuses_buffer_and_mirrors_source_values(self):
        config = self._create_config()
        op = FusedRopeKVCacheDecodeOp(config.attn_configs)

        capture_inputs = self._inputs([10, 20, 30, 40])
        capture_buffer = op.refresh_sequence_lengths(capture_inputs)
        capture_ptr = capture_buffer.data_ptr()

        replay_inputs = self._inputs([100, 200, 0, 0])
        replay_buffer = op.refresh_sequence_lengths(replay_inputs)
        torch.cuda.synchronize()

        self.assertIs(replay_buffer, capture_buffer)
        self.assertEqual(replay_buffer.data_ptr(), capture_ptr)
        self.assertTrue(replay_buffer.is_cuda)
        self.assertEqual(replay_buffer.dtype, torch.int32)
        torch.testing.assert_close(
            replay_buffer.cpu(), torch.tensor([100, 200, 0, 0], dtype=torch.int32)
        )

    def test_cuda_source_is_copied_into_owned_buffer(self):
        config = self._create_config()
        op = FusedRopeKVCacheDecodeOp(config.attn_configs)
        inputs = PyAttentionInputs()
        inputs.is_cuda_graph = False
        inputs.sequence_lengths = torch.tensor(
            [7], dtype=torch.int32, device=self.device
        )

        sequence_lengths = op.refresh_sequence_lengths(inputs)

        self.assertTrue(sequence_lengths.is_cuda)
        self.assertNotEqual(
            sequence_lengths.data_ptr(), inputs.sequence_lengths.data_ptr()
        )
        torch.testing.assert_close(sequence_lengths, inputs.sequence_lengths)

    def test_pageable_cpu_source_uses_sync_copy_outside_graph(self):
        config = self._create_config()
        op = FusedRopeKVCacheDecodeOp(config.attn_configs)
        inputs = self._inputs([1], is_cuda_graph=False, pinned=False)

        sequence_lengths = op.refresh_sequence_lengths(inputs, self.device)

        torch.testing.assert_close(sequence_lengths.cpu(), inputs.sequence_lengths)

    def test_contract_errors(self):
        config = self._create_config()

        missing = PyAttentionInputs()
        missing.is_cuda_graph = False

        cases = [
            ("missing", missing, None, "non-empty"),
            (
                "empty",
                self._inputs([], is_cuda_graph=False),
                None,
                "non-empty",
            ),
            (
                "dtype",
                self._inputs([1], is_cuda_graph=False, dtype=torch.int64),
                None,
                "torch.int32",
            ),
            (
                "pageable_graph_source",
                self._inputs([1], pinned=False),
                self.device,
                "CUDA or pinned",
            ),
            (
                "missing_target",
                self._inputs([1], is_cuda_graph=False, with_target=False),
                None,
                "CUDA target device",
            ),
        ]
        for name, inputs, device, message in cases:
            with self.subTest(name=name):
                op = FusedRopeKVCacheDecodeOp(config.attn_configs)
                with self.assertRaisesRegex(DecodeRopeContractError, message):
                    op.refresh_sequence_lengths(inputs, device)

        pageable_replay = self._inputs([1], is_cuda_graph=False, pinned=False)
        with self.assertRaisesRegex(DecodeRopeContractError, "CUDA or pinned"):
            FusedRopeKVCacheDecodeOp(config.attn_configs).refresh_sequence_lengths(
                pageable_replay,
                self.device,
                forbid_reallocation=True,
            )

    def test_prepare_reuses_owned_sequence_lengths_buffer(self):
        config = self._create_config()
        config.attn_configs.need_rope_kv_cache = True
        op = FusedRopeKVCacheDecodeOp(config.attn_configs)
        inputs = self._create_attention_inputs_base(1, [2], 64)
        inputs.is_cuda_graph = True

        first = op.prepare(inputs)
        first_ptr = first.sequence_lengths.data_ptr()
        inputs.sequence_lengths.fill_(7)
        second = op.prepare(inputs)
        torch.cuda.synchronize()

        self.assertIs(first.sequence_lengths, second.sequence_lengths)
        self.assertEqual(second.sequence_lengths.data_ptr(), first_ptr)
        torch.testing.assert_close(
            second.sequence_lengths.cpu(), torch.tensor([7], dtype=torch.int32)
        )

    def test_real_kernel_uses_owned_device_sequence_lengths(self):
        cases = ([5], [5, 9, 1, 1])
        for sequence_lengths in cases:
            with self.subTest(sequence_lengths=sequence_lengths):
                config, inputs, op, params, qkv, kv_cache = self._real_kernel_case(
                    list(sequence_lengths)
                )

                actual = op.forward(qkv, kv_cache, params)
                expected = self._base_rope_reference(
                    qkv,
                    inputs.sequence_lengths,
                    config.head_num,
                    config.size_per_head,
                    config.attn_configs.rope_config.base,
                )

                self.assertTrue(params.sequence_lengths.is_cuda)
                self.assertIs(params.sequence_lengths, op._sequence_lengths_device)
                torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)

    def test_real_kernel_graph_replay_observes_refreshed_lengths(self):
        capture_lengths = [3, 6, 1, 1]
        config, inputs, op, params, qkv, kv_cache = self._real_kernel_case(
            capture_lengths
        )
        sequence_lengths_ptr = params.sequence_lengths.data_ptr()

        # Materialize lazy kernel/rope-cache state before graph capture.
        op.forward(qkv, kv_cache, params)
        torch.cuda.synchronize()
        with record_cuda_graph() as graph:
            graph_output = op.forward(qkv, kv_cache, params)
        graph.replay()
        torch.cuda.synchronize()
        capture_output = graph_output.clone()
        capture_expected = self._base_rope_reference(
            qkv,
            inputs.sequence_lengths,
            config.head_num,
            config.size_per_head,
            config.attn_configs.rope_config.base,
        )
        torch.testing.assert_close(
            capture_output, capture_expected, rtol=1e-2, atol=1e-2
        )

        # Keep the graph-batch tail at zero, matching CudaGraphRunner padding.
        replay_positions = torch.tensor([11, 19, 0, 0], dtype=torch.int32)
        inputs.sequence_lengths.copy_(replay_positions)
        replay_buffer = op.refresh_sequence_lengths(inputs, forbid_reallocation=True)
        self.assertIs(replay_buffer, params.sequence_lengths)
        self.assertEqual(replay_buffer.data_ptr(), sequence_lengths_ptr)

        graph.replay()
        torch.cuda.synchronize()
        replay_output = graph_output.clone()
        replay_expected = self._base_rope_reference(
            qkv,
            replay_positions,
            config.head_num,
            config.size_per_head,
            config.attn_configs.rope_config.base,
        )
        torch.testing.assert_close(replay_output, replay_expected, rtol=1e-2, atol=1e-2)
        self.assertFalse(torch.allclose(capture_output[:2], replay_output[:2]))
        torch.testing.assert_close(capture_output[2:], replay_output[2:])

    def test_prepare_skips_unused_rope_buffer_for_pageable_input(self):
        config = self._create_config()
        config.attn_configs.need_rope_kv_cache = False
        op = FusedRopeKVCacheDecodeOp(config.attn_configs)
        inputs = self._create_attention_inputs_base(1, [2], 64)
        inputs.sequence_lengths = torch.tensor([1], dtype=torch.int32)
        inputs.is_cuda_graph = True

        params = op.prepare(inputs)

        self.assertIs(params.sequence_lengths, inputs.sequence_lengths)
        self.assertIsNone(op._sequence_lengths_device)

    def test_graph_shape_change_is_rejected(self):
        config = self._create_config()
        op = FusedRopeKVCacheDecodeOp(config.attn_configs)
        op.refresh_sequence_lengths(self._inputs([1, 2], is_cuda_graph=False))

        with self.assertRaisesRegex(DecodeRopeContractError, "cannot resize or move"):
            op.refresh_sequence_lengths(
                self._inputs([3], is_cuda_graph=False), forbid_reallocation=True
            )

    def test_indexless_device_reuses_buffer_and_eager_resize_rebinds(self):
        config = self._create_config()
        op = FusedRopeKVCacheDecodeOp(config.attn_configs)
        first_inputs = self._inputs([1, 2], is_cuda_graph=False)
        first = op.refresh_sequence_lengths(first_inputs, torch.device("cuda"))
        first_ptr = first.data_ptr()

        second = op.refresh_sequence_lengths(first_inputs, self.device)
        self.assertIs(second, first)
        self.assertEqual(second.data_ptr(), first_ptr)

        resized_inputs = self._inputs([3], is_cuda_graph=False)
        resized = op.refresh_sequence_lengths(resized_inputs, self.device)
        self.assertIsNot(resized, first)
        self.assertEqual(tuple(resized.shape), (1,))
        torch.testing.assert_close(resized.cpu(), torch.tensor([3], dtype=torch.int32))

    def test_forward_rejects_missing_offset_and_non_owned_lengths(self):
        config = self._create_config()
        op = FusedRopeKVCacheDecodeOp(config.attn_configs)
        qkv = torch.empty(0, device=self.device)
        host_lengths = torch.tensor([1], dtype=torch.int32).pin_memory()

        with self.assertRaisesRegex(DecodeRopeContractError, "kv_cache_offset"):
            op.forward(qkv, None, self._params(host_lengths, None))

        offset = torch.empty(0, dtype=torch.int32, device=self.device)
        with self.assertRaisesRegex(DecodeRopeContractError, "must be on CUDA"):
            op.forward(qkv, None, self._params(host_lengths, offset))

        foreign_device_lengths = torch.tensor(
            [1], dtype=torch.int32, device=self.device
        )
        with self.assertRaisesRegex(DecodeRopeContractError, "op-owned"):
            op.forward(qkv, None, self._params(foreign_device_lengths, offset))

    def test_xqa_graph_rejects_replaced_fmha_host_buffer(self):
        impl = XQAImpl.__new__(XQAImpl)
        captured = torch.tensor([1], dtype=torch.int32).pin_memory()
        impl._fmha_sequence_lengths_ptr = captured.data_ptr()
        # prepare_cuda_graph itself defines the graph boundary; the pybind input
        # flag is not guaranteed to remain set during replay preparation.
        replay_inputs = self._inputs([2], is_cuda_graph=False)

        with self.assertRaisesRegex(DecodeRopeContractError, "captured host"):
            impl.prepare_cuda_graph(replay_inputs)

    def test_xqa_fast_path_refreshes_rope_buffer(self):
        host_lengths = torch.tensor([4], dtype=torch.int32).pin_memory()
        replay_inputs = PyAttentionInputs()
        replay_inputs.is_cuda_graph = True
        replay_inputs.sequence_lengths = host_lengths
        replay_inputs.input_lengths = torch.ones(
            1, dtype=torch.int32, device=self.device
        )
        replay_inputs.kv_cache_kernel_block_id_device = torch.zeros(
            (1, 1), dtype=torch.int32, device=self.device
        )

        class FakeFmha:
            def update(self, params, inputs):
                return None

            def update_kv_cache_offset(self, offset, block_ids):
                return None

        class FakeRope:
            def __init__(self, device):
                self.calls = 0
                self.buffer = torch.empty(1, dtype=torch.int32, device=device)

            def refresh_sequence_lengths(self, inputs, *, forbid_reallocation=False):
                self.calls += 1
                assert forbid_reallocation
                self.buffer.copy_(inputs.sequence_lengths)
                return self.buffer

        fake_rope = FakeRope(self.device)
        impl = XQAImpl.__new__(XQAImpl)
        impl.fmha_impl = FakeFmha()
        impl.fmha_params = object()
        impl.rope_kvcache_impl = fake_rope
        impl.rope_params = SimpleNamespace(
            kv_cache_offset=torch.empty(0, dtype=torch.int32, device=self.device),
            sequence_lengths=torch.empty(1, dtype=torch.int32, device=self.device),
        )
        impl.need_rope_kv_cache = True
        impl._fmha_sequence_lengths_ptr = host_lengths.data_ptr()

        impl.prepare_cuda_graph(replay_inputs)

        self.assertEqual(fake_rope.calls, 1)
        self.assertIs(impl.rope_params.sequence_lengths, fake_rope.buffer)
        torch.testing.assert_close(fake_rope.buffer.cpu(), host_lengths)

    def test_xqa_host_fallback_refreshes_rope_buffer(self):
        host_lengths = torch.tensor([5], dtype=torch.int32).pin_memory()
        replay_inputs = PyAttentionInputs()
        replay_inputs.is_cuda_graph = True
        replay_inputs.sequence_lengths = host_lengths
        replay_inputs.input_lengths = torch.ones(1, dtype=torch.int32)
        replay_inputs.kv_cache_kernel_block_id_device = torch.zeros(
            (1, 1), dtype=torch.int32, device=self.device
        )

        class FakeFmha:
            def __init__(self, device):
                self.prepare_calls = 0
                self.offset = torch.ones((1, 1), dtype=torch.int32, device=device)

            def prepare(self, inputs):
                self.prepare_calls += 1
                return SimpleNamespace(kv_cache_offset=self.offset)

        class FakeRope:
            def __init__(self, device):
                self.prepare_calls = 0
                self.refresh_calls = 0
                self.buffer = torch.empty(1, dtype=torch.int32, device=device)
                self.offset = torch.ones((1, 1), dtype=torch.int32, device=device)

            def prepare(self, inputs, *, forbid_reallocation=False):
                self.prepare_calls += 1
                assert forbid_reallocation
                return SimpleNamespace(
                    kv_cache_offset=self.offset,
                    sequence_lengths=self.buffer,
                )

            def refresh_sequence_lengths(self, inputs, *, forbid_reallocation=False):
                self.refresh_calls += 1
                assert forbid_reallocation
                self.buffer.copy_(inputs.sequence_lengths)
                return self.buffer

        fake_fmha = FakeFmha(self.device)
        fake_rope = FakeRope(self.device)
        impl = XQAImpl.__new__(XQAImpl)
        impl.fmha_impl = fake_fmha
        impl.fmha_params = SimpleNamespace(
            kv_cache_offset=torch.zeros((1, 1), dtype=torch.int32, device=self.device)
        )
        impl.rope_kvcache_impl = fake_rope
        impl.rope_params = SimpleNamespace(
            kv_cache_offset=torch.zeros((1, 1), dtype=torch.int32, device=self.device),
            sequence_lengths=torch.empty(1, dtype=torch.int32, device=self.device),
        )
        impl.need_rope_kv_cache = True
        impl._fmha_sequence_lengths_ptr = host_lengths.data_ptr()

        impl.prepare_cuda_graph(replay_inputs)

        self.assertEqual(fake_fmha.prepare_calls, 1)
        self.assertEqual(fake_rope.prepare_calls, 1)
        self.assertEqual(fake_rope.refresh_calls, 1)
        self.assertIs(impl.rope_params.sequence_lengths, fake_rope.buffer)
        torch.testing.assert_close(fake_rope.buffer.cpu(), host_lengths)

    def test_xqa_fast_path_without_offset_update_refreshes_rope_buffer(self):
        host_lengths = torch.tensor([6], dtype=torch.int32).pin_memory()
        replay_inputs = PyAttentionInputs()
        replay_inputs.is_cuda_graph = True
        replay_inputs.sequence_lengths = host_lengths
        replay_inputs.input_lengths = torch.ones(
            1, dtype=torch.int32, device=self.device
        )
        replay_inputs.kv_cache_kernel_block_id_device = torch.zeros(
            (1, 1), dtype=torch.int32, device=self.device
        )

        class FakeFmha:
            def __init__(self):
                self.update_calls = 0

            def update(self, params, inputs):
                self.update_calls += 1

        class FakeRope:
            def __init__(self, device):
                self.prepare_calls = 0
                self.refresh_calls = 0
                self.buffer = torch.empty(1, dtype=torch.int32, device=device)
                self.offset = torch.ones((1, 1), dtype=torch.int32, device=device)

            def prepare(self, inputs, *, forbid_reallocation=False):
                self.prepare_calls += 1
                assert forbid_reallocation
                return SimpleNamespace(
                    kv_cache_offset=self.offset,
                    sequence_lengths=self.buffer,
                )

            def refresh_sequence_lengths(self, inputs, *, forbid_reallocation=False):
                self.refresh_calls += 1
                assert forbid_reallocation
                self.buffer.copy_(inputs.sequence_lengths)
                return self.buffer

        fake_fmha = FakeFmha()
        fake_rope = FakeRope(self.device)
        impl = XQAImpl.__new__(XQAImpl)
        impl.fmha_impl = fake_fmha
        impl.fmha_params = object()
        impl.rope_kvcache_impl = fake_rope
        impl.rope_params = SimpleNamespace(
            kv_cache_offset=torch.zeros((1, 1), dtype=torch.int32, device=self.device),
            sequence_lengths=torch.empty(1, dtype=torch.int32, device=self.device),
        )
        impl.need_rope_kv_cache = True
        impl._fmha_sequence_lengths_ptr = host_lengths.data_ptr()

        impl.prepare_cuda_graph(replay_inputs)

        self.assertEqual(fake_fmha.update_calls, 1)
        self.assertEqual(fake_rope.prepare_calls, 1)
        self.assertEqual(fake_rope.refresh_calls, 1)
        self.assertIs(impl.rope_params.sequence_lengths, fake_rope.buffer)
        torch.testing.assert_close(fake_rope.buffer.cpu(), host_lengths)

    def test_factory_reraises_decode_rope_contract_errors(self):
        class ContractFailImpl:
            accepts_fmha_config = False

            @classmethod
            def support(cls, attn_configs, attn_inputs):
                return True

            @classmethod
            def support_parallelism_config(cls, parallelism_config):
                return True

            def __init__(self, *args, **kwargs):
                raise DecodeRopeContractError("invalid decode RoPE input")

        class FallbackImpl:
            accepts_fmha_config = False
            construction_calls = 0

            @classmethod
            def support(cls, attn_configs, attn_inputs):
                return True

            @classmethod
            def support_parallelism_config(cls, parallelism_config):
                return True

            def __init__(self, *args, **kwargs):
                type(self).construction_calls += 1

        inputs = PyAttentionInputs()
        inputs.is_prefill = False
        config = self._create_config().attn_configs

        with patch.object(
            attn_factory,
            "DECODE_MHA_IMPS",
            [ContractFailImpl, FallbackImpl],
        ), patch.object(attn_factory, "VALIDATE_FMHA_CONFIG", None):
            with self.assertRaisesRegex(
                DecodeRopeContractError, "invalid decode RoPE input"
            ):
                attn_factory.get_fmha_impl(config, None, inputs)

        self.assertEqual(FallbackImpl.construction_calls, 0)


if __name__ == "__main__":
    unittest.main()
