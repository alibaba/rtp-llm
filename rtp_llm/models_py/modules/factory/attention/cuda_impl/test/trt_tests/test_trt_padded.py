"""Tests for TRTAttnOp (padded mode)

Tests standard prefill without prefix cache but with padding.
This mode is used for CUDA graph scenarios where fixed input shapes are required.
"""

import unittest

import pytest

from rtp_llm.models_py.modules.factory.attention.cuda_impl.test.trt_tests.test_trt_base import (
    TRTAttnTestBase,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.test.trt_tests.trt_test_utils import (
    print_attn_inputs_detail,
)
try:
    from rtp_llm.ops.compute_ops import TRTAttnOp
except ImportError as exc:
    pytest.skip(f"TRTAttnOp import failed: {exc}", allow_module_level=True)

@pytest.mark.H20
@pytest.mark.cuda
@pytest.mark.gpu
class TestTRTAttnOpPadded(TRTAttnTestBase):
    """Test suite for TRTAttnOp in padded mode

    TRTAttnOp (TRTNormalPrefillOp):
    - Standard prefill without prefix cache
    - prefix_lengths must be 0 or None
    - Only processes new input_lengths tokens
    - Padded mode: sequences padded to max_seq_len for CUDA graph
    """

    def test_basic(self):
        """Test TRTAttnOp with padded mode - basic"""
        print("\n=== Test TRTAttnOp Padded: Basic ===", flush=True)

        batch_size = 2
        input_lengths = [64, 128]
        head_num = 32
        head_num_kv = 8
        size_per_head = 128
        seq_size_per_block = 64
        max_seq_len = max(input_lengths)

        attn_configs = self._create_config(
            head_num=head_num,
            head_num_kv=head_num_kv,
            size_per_head=size_per_head,
            seq_size_per_block=seq_size_per_block,
        )

        attn_inputs = self._create_prefill_attention_inputs_padded(
            batch_size,
            input_lengths,
            max_seq_len,
            seq_size_per_block,
            prefix_lengths=None,
        )

        attn_op = TRTAttnOp(attn_configs)

        self.run_correctness_test(
            attn_op=attn_op,
            op_name="TRTAttnOp",
            batch_size=batch_size,
            input_lengths=input_lengths,
            head_num=head_num,
            head_num_kv=head_num_kv,
            size_per_head=size_per_head,
            seq_size_per_block=seq_size_per_block,
            attn_configs=attn_configs,
            attn_inputs=attn_inputs,
            prefix_lengths=None,
            use_padded=True,
        )

    def test_batch(self):
        """Test TRTAttnOp with padded mode and variable lengths"""
        print("\n=== Test TRTAttnOp Padded: Batch ===", flush=True)

        batch_size = 4
        input_lengths = [32, 64, 96, 128]
        head_num = 32
        head_num_kv = 8
        size_per_head = 128
        seq_size_per_block = 64
        max_seq_len = max(input_lengths)

        attn_configs = self._create_config(
            head_num=head_num,
            head_num_kv=head_num_kv,
            size_per_head=size_per_head,
            seq_size_per_block=seq_size_per_block,
        )

        attn_inputs = self._create_prefill_attention_inputs_padded(
            batch_size,
            input_lengths,
            max_seq_len,
            seq_size_per_block,
            prefix_lengths=None,
        )

        attn_op = TRTAttnOp(attn_configs)

        self.run_correctness_test(
            attn_op=attn_op,
            op_name="TRTAttnOp",
            batch_size=batch_size,
            input_lengths=input_lengths,
            head_num=head_num,
            head_num_kv=head_num_kv,
            size_per_head=size_per_head,
            seq_size_per_block=seq_size_per_block,
            attn_configs=attn_configs,
            attn_inputs=attn_inputs,
            prefix_lengths=None,
            use_padded=True,
        )

    def test_long_sequence(self):
        """Test TRTAttnOp with padded mode and long sequences"""
        print("\n=== Test TRTAttnOp Padded: Long Sequence ===", flush=True)

        batch_size = 2
        input_lengths = [512, 1024]
        head_num = 32
        head_num_kv = 8
        size_per_head = 128
        seq_size_per_block = 64
        max_seq_len = max(input_lengths)

        attn_configs = self._create_config(
            head_num=head_num,
            head_num_kv=head_num_kv,
            size_per_head=size_per_head,
            seq_size_per_block=seq_size_per_block,
        )

        attn_inputs = self._create_prefill_attention_inputs_padded(
            batch_size,
            input_lengths,
            max_seq_len,
            seq_size_per_block,
            prefix_lengths=None,
        )

        attn_op = TRTAttnOp(attn_configs)

        self.run_correctness_test(
            attn_op=attn_op,
            op_name="TRTAttnOp",
            batch_size=batch_size,
            input_lengths=input_lengths,
            head_num=head_num,
            head_num_kv=head_num_kv,
            size_per_head=size_per_head,
            seq_size_per_block=seq_size_per_block,
            attn_configs=attn_configs,
            attn_inputs=attn_inputs,
            prefix_lengths=None,
            use_padded=True,
        )

    def test_consistency_with_nonpadded(self):
        """Test consistency between padded and non-padded modes

        This test verifies that padded and non-padded modes produce the same results
        when given the same actual data (in different layouts).
        """
        print("\n=== Test TRTAttnOp Padded: Consistency ===", flush=True)

        import torch

        from rtp_llm.models_py.modules.factory.attention.cuda_impl.test.base_attention_test import (
            compare_tensors,
            set_seed,
        )
        from rtp_llm.ops.compute_ops import get_typemeta

        batch_size = 2
        input_lengths = [64, 128]
        head_num = 32
        head_num_kv = 8
        size_per_head = 128
        seq_size_per_block = 64
        max_seq_len = max(input_lengths)
        total_tokens = sum(input_lengths)

        set_seed(42)

        attn_configs = self._create_config(
            head_num=head_num,
            head_num_kv=head_num_kv,
            size_per_head=size_per_head,
            seq_size_per_block=seq_size_per_block,
        )

        # Create inputs for both modes
        attn_inputs_nonpadded = self._create_prefill_attention_inputs(
            batch_size, input_lengths, seq_size_per_block, prefix_lengths=None
        )
        attn_inputs_padded = self._create_prefill_attention_inputs_padded(
            batch_size,
            input_lengths,
            max_seq_len,
            seq_size_per_block,
            prefix_lengths=None,
        )

        # Create QKV with same seed for both modes
        qkv_dim = (head_num + 2 * head_num_kv) * size_per_head

        # Non-padded QKV: [total_tokens, qkv_dim] - compact layout
        set_seed(123)
        qkv_nonpadded = torch.randn(
            total_tokens, qkv_dim, dtype=attn_configs.dtype, device=self.device
        )

        # Padded QKV: [batch_size * max_seq_len, qkv_dim] - padded layout with same data
        qkv_padded = torch.zeros(
            batch_size * max_seq_len,
            qkv_dim,
            dtype=attn_configs.dtype,
            device=self.device,
        )
        offset = 0
        for i, seq_len in enumerate(input_lengths):
            seq_start = i * max_seq_len
            qkv_padded[seq_start : seq_start + seq_len] = qkv_nonpadded[
                offset : offset + seq_len
            ]
            offset += seq_len

        attn_inputs_nonpadded.dtype = get_typemeta(qkv_nonpadded)
        attn_inputs_padded.dtype = get_typemeta(qkv_padded)

        # IMPORTANT: Create separate attn_op instances for each mode
        # because trt_v2_runner_ is created based on is_s_padded in the first prepare() call
        attn_op_nonpadded = TRTAttnOp(attn_configs)
        attn_op_padded = TRTAttnOp(attn_configs)

        # Check support and prepare both modes with their respective attn_op instances
        self.assertTrue(
            attn_op_nonpadded.support(attn_inputs_nonpadded),
            "TRTAttnOp does not support non-padded mode inputs",
        )
        self.assertTrue(
            attn_op_padded.support(attn_inputs_padded),
            "TRTAttnOp does not support padded mode inputs",
        )

        params_nonpadded = attn_op_nonpadded.prepare(attn_inputs_nonpadded)
        params_padded = attn_op_padded.prepare(attn_inputs_padded)
        # print_attn_inputs_detail(attn_inputs_nonpadded)
        # print_attn_inputs_detail(attn_inputs_padded)

        # Run forward pass for both modes (no KV cache needed for this test)
        output_nonpadded = attn_op_nonpadded.forward(
            qkv_nonpadded, None, params_nonpadded
        ).clone()
        output_padded = attn_op_padded.forward(qkv_padded, None, params_padded).clone()

        print(f"Non-padded output shape: {output_nonpadded.shape}", flush=True)
        print(f"Padded output shape: {output_padded.shape}", flush=True)

        # Extract valid outputs from padded results
        output_padded_extracted = []
        for i, seq_len in enumerate(input_lengths):
            seq_start = i * max_seq_len
            output_padded_extracted.append(
                output_padded[seq_start : seq_start + seq_len]
            )
        output_padded_extracted = torch.cat(output_padded_extracted, dim=0)
        print(f"output_nonpadded1: {output_nonpadded[:64]}")
        print(f"output_padded_extracted1: {output_padded_extracted[:64]}")
        print(f"output_nonpadded2: {output_nonpadded[128:]}")
        print(f"output_padded_extracted2: {output_padded_extracted[128:]}")
        # Compare shapes
        self.assertEqual(
            output_nonpadded.shape,
            output_padded_extracted.shape,
            "Output shapes should match after extracting valid data from padded output",
        )

        # Compare values
        max_diff = torch.max(
            torch.abs(output_nonpadded - output_padded_extracted)
        ).item()
        mean_diff = torch.mean(
            torch.abs(output_nonpadded - output_padded_extracted)
        ).item()

        print(
            f"Consistency check:\n"
            f"  Max diff: {max_diff:.6e}\n"
            f"  Mean diff: {mean_diff:.6e}",
            flush=True,
        )

        compare_tensors(
            output_nonpadded,
            output_padded_extracted,
            rtol=5e-3,
            atol=5e-3,
            name="TRTAttnOp padded vs non-padded",
        )

        print("✓ Consistency check passed!", flush=True)


if __name__ == "__main__":
    unittest.main()
