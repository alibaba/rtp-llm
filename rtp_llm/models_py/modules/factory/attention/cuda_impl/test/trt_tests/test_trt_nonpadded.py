"""Tests for TRTAttnOp (non-padded mode)

Tests standard prefill without prefix cache and without padding.
This mode is used for dynamic batch processing.
"""

import unittest

import pytest

from rtp_llm.models_py.modules.factory.attention.cuda_impl.test.trt_tests.test_trt_base import (
    TRTAttnTestBase,
)
try:
    from rtp_llm.ops.compute_ops import TRTAttnOp
except ImportError as exc:
    pytest.skip(f"TRTAttnOp import failed: {exc}", allow_module_level=True)


@pytest.mark.H20
@pytest.mark.cuda
@pytest.mark.gpu
class TestTRTAttnOpNonPadded(TRTAttnTestBase):
    """Test suite for TRTAttnOp in non-padded mode

    TRTAttnOp (TRTNormalPrefillOp):
    - Standard prefill without prefix cache
    - prefix_lengths must be 0 or None
    - Only processes new input_lengths tokens
    - Non-padded mode: variable sequence lengths (no padding)
    """

    def test_basic(self):
        """Test basic TRTAttnOp with single sequence"""
        print("\n=== Test TRTAttnOp Non-Padded: Basic ===", flush=True)

        batch_size = 1
        input_lengths = [128]
        head_num = 32
        head_num_kv = 8
        size_per_head = 128
        seq_size_per_block = 64

        attn_configs = self._create_config(
            head_num=head_num,
            head_num_kv=head_num_kv,
            size_per_head=size_per_head,
            seq_size_per_block=seq_size_per_block,
        )

        attn_inputs = self._create_prefill_attention_inputs(
            batch_size, input_lengths, seq_size_per_block, prefix_lengths=None
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
            use_padded=False,
        )

    def test_batch(self):
        """Test TRTAttnOp with multiple sequences of variable lengths"""
        print("\n=== Test TRTAttnOp Non-Padded: Batch ===", flush=True)

        batch_size = 4
        input_lengths = [64, 128, 256, 512]
        head_num = 32
        head_num_kv = 8
        size_per_head = 128
        seq_size_per_block = 64

        attn_configs = self._create_config(
            head_num=head_num,
            head_num_kv=head_num_kv,
            size_per_head=size_per_head,
            seq_size_per_block=seq_size_per_block,
        )

        attn_inputs = self._create_prefill_attention_inputs(
            batch_size, input_lengths, seq_size_per_block, prefix_lengths=None
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
            use_padded=False,
        )

    def test_gqa(self):
        """Test TRTAttnOp with grouped query attention"""
        print("\n=== Test TRTAttnOp Non-Padded: GQA ===", flush=True)

        batch_size = 2
        input_lengths = [256, 512]
        head_num = 32
        head_num_kv = 4  # More aggressive GQA
        size_per_head = 128
        seq_size_per_block = 64

        attn_configs = self._create_config(
            head_num=head_num,
            head_num_kv=head_num_kv,
            size_per_head=size_per_head,
            seq_size_per_block=seq_size_per_block,
        )

        attn_inputs = self._create_prefill_attention_inputs(
            batch_size, input_lengths, seq_size_per_block, prefix_lengths=None
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
            use_padded=False,
        )

    def test_long_sequence(self):
        """Test TRTAttnOp with long sequences"""
        print("\n=== Test TRTAttnOp Non-Padded: Long Sequence ===", flush=True)

        batch_size = 2
        input_lengths = [1024, 2048]
        head_num = 32
        head_num_kv = 8
        size_per_head = 128
        seq_size_per_block = 64

        attn_configs = self._create_config(
            head_num=head_num,
            head_num_kv=head_num_kv,
            size_per_head=size_per_head,
            seq_size_per_block=seq_size_per_block,
        )

        attn_inputs = self._create_prefill_attention_inputs(
            batch_size, input_lengths, seq_size_per_block, prefix_lengths=None
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
            use_padded=False,
        )


if __name__ == "__main__":
    unittest.main()
