"""Tests for TRTPagedAttnOp (non-padded mode)

Tests paged prefill with prefix cache but without padding.
This mode is used when there's existing KV cache (prefix/prompt caching).
"""

import unittest

from rtp_llm.models_py.modules.factory.attention.cuda_impl.test.trt_tests.test_trt_base import (
    TRTAttnTestBase,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.test.trt_tests.trt_test_utils import (
    print_attn_inputs_detail,
)
from rtp_llm.ops.compute_ops import TRTPagedAttnOp


class TestTRTPagedAttnOpNonPadded(TRTAttnTestBase):
    """Test suite for TRTPagedAttnOp in non-padded mode

    TRTPagedAttnOp (TRTPagedPrefillOp):
    - Paged prefill with prefix cache (prompt caching)
    - prefix_lengths must be > 0 (already cached KV)
    - Processes new input_lengths tokens with existing prefix_lengths cache
    - Total KV length = prefix_lengths + input_lengths
    - Non-padded mode: variable sequence lengths (no padding)
    """

    def test_basic(self):
        """Test basic TRTPagedAttnOp with single sequence and prefix cache"""
        print(
            "\n=== Test TRTPagedAttnOp Non-Padded: Basic (With Prefix) ===", flush=True
        )

        batch_size = 1
        input_lengths = [4]  # New tokens to process
        prefix_lengths = [2]  # Already cached KV length
        head_num = 8
        head_num_kv = 2
        size_per_head = 32
        seq_size_per_block = 8

        attn_configs = self._create_config(
            head_num=head_num,
            head_num_kv=head_num_kv,
            size_per_head=size_per_head,
            seq_size_per_block=seq_size_per_block,
        )

        attn_inputs = self._create_prefill_attention_inputs(
            batch_size, input_lengths, seq_size_per_block, prefix_lengths=prefix_lengths
        )

        attn_op = TRTPagedAttnOp(attn_configs)

        self.run_correctness_test(
            attn_op=attn_op,
            op_name="TRTPagedAttnOp",
            batch_size=batch_size,
            input_lengths=input_lengths,
            head_num=head_num,
            head_num_kv=head_num_kv,
            size_per_head=size_per_head,
            seq_size_per_block=seq_size_per_block,
            attn_configs=attn_configs,
            attn_inputs=attn_inputs,
            prefix_lengths=prefix_lengths,
            use_padded=False,
        )

    # def test_batch(self):
    #     """Test TRTPagedAttnOp with multiple sequences and different prefix lengths"""
    #     print("\n=== Test TRTPagedAttnOp Non-Padded: Batch ===", flush=True)

    #     batch_size = 4
    #     input_lengths = [32, 64, 128, 256]  # New tokens to process
    #     prefix_lengths = [32, 64, 128, 256]  # Variable prefix cache lengths
    #     head_num = 32
    #     head_num_kv = 8
    #     size_per_head = 128
    #     seq_size_per_block = 64

    #     attn_configs = self._create_config(
    #         head_num=head_num,
    #         head_num_kv=head_num_kv,
    #         size_per_head=size_per_head,
    #         seq_size_per_block=seq_size_per_block,
    #     )

    #     attn_inputs = self._create_prefill_attention_inputs(
    #         batch_size, input_lengths, seq_size_per_block, prefix_lengths=prefix_lengths
    #     )

    #     attn_op = TRTPagedAttnOp(attn_configs)

    #     self.run_correctness_test(
    #         attn_op=attn_op,
    #         op_name="TRTPagedAttnOp",
    #         batch_size=batch_size,
    #         input_lengths=input_lengths,
    #         head_num=head_num,
    #         head_num_kv=head_num_kv,
    #         size_per_head=size_per_head,
    #         seq_size_per_block=seq_size_per_block,
    #         attn_configs=attn_configs,
    #         attn_inputs=attn_inputs,
    #         prefix_lengths=prefix_lengths,
    #         use_padded=False,
    #     )

    # def test_gqa(self):
    #     """Test TRTPagedAttnOp with grouped query attention and prefix cache"""
    #     print("\n=== Test TRTPagedAttnOp Non-Padded: GQA ===", flush=True)

    #     batch_size = 2
    #     input_lengths = [128, 256]  # New tokens
    #     prefix_lengths = [128, 256]  # Prefix cache
    #     head_num = 32
    #     head_num_kv = 4
    #     size_per_head = 128
    #     seq_size_per_block = 64

    #     attn_configs = self._create_config(
    #         head_num=head_num,
    #         head_num_kv=head_num_kv,
    #         size_per_head=size_per_head,
    #         seq_size_per_block=seq_size_per_block,
    #     )

    #     attn_inputs = self._create_prefill_attention_inputs(
    #         batch_size, input_lengths, seq_size_per_block, prefix_lengths=prefix_lengths
    #     )

    #     attn_op = TRTPagedAttnOp(attn_configs)

    #     self.run_correctness_test(
    #         attn_op=attn_op,
    #         op_name="TRTPagedAttnOp",
    #         batch_size=batch_size,
    #         input_lengths=input_lengths,
    #         head_num=head_num,
    #         head_num_kv=head_num_kv,
    #         size_per_head=size_per_head,
    #         seq_size_per_block=seq_size_per_block,
    #         attn_configs=attn_configs,
    #         attn_inputs=attn_inputs,
    #         prefix_lengths=prefix_lengths,
    #         use_padded=False,
    #     )

    # def test_long_sequence(self):
    #     """Test TRTPagedAttnOp with long sequences and long prefix cache"""
    #     print("\n=== Test TRTPagedAttnOp Non-Padded: Long Sequence ===", flush=True)

    #     batch_size = 2
    #     input_lengths = [512, 1024]  # New tokens
    #     prefix_lengths = [512, 1024]  # Long prefix cache
    #     head_num = 32
    #     head_num_kv = 8
    #     size_per_head = 128
    #     seq_size_per_block = 64

    #     attn_configs = self._create_config(
    #         head_num=head_num,
    #         head_num_kv=head_num_kv,
    #         size_per_head=size_per_head,
    #         seq_size_per_block=seq_size_per_block,
    #     )

    #     attn_inputs = self._create_prefill_attention_inputs(
    #         batch_size, input_lengths, seq_size_per_block, prefix_lengths=prefix_lengths
    #     )

    #     attn_op = TRTPagedAttnOp(attn_configs)

    #     self.run_correctness_test(
    #         attn_op=attn_op,
    #         op_name="TRTPagedAttnOp",
    #         batch_size=batch_size,
    #         input_lengths=input_lengths,
    #         head_num=head_num,
    #         head_num_kv=head_num_kv,
    #         size_per_head=size_per_head,
    #         seq_size_per_block=seq_size_per_block,
    #         attn_configs=attn_configs,
    #         attn_inputs=attn_inputs,
    #         prefix_lengths=prefix_lengths,
    #         use_padded=False,
    #     )


if __name__ == "__main__":
    unittest.main()
