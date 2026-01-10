"""Tests for TRTAttnOp (padded mode)

Tests standard prefill without prefix cache but with padding.
This mode is used for CUDA graph scenarios where fixed input shapes are required.
"""

import unittest

from rtp_llm.models_py.modules.factory.attention.cuda_impl.test.trt_tests.test_trt_base import (
    TRTAttnTestBase,
)
from rtp_llm.ops.compute_ops import TRTAttnOp


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

    # def test_batch(self):
    #     """Test TRTAttnOp with padded mode and variable lengths"""
    #     print("\n=== Test TRTAttnOp Padded: Batch ===", flush=True)

    #     batch_size = 4
    #     input_lengths = [32, 64, 96, 128]
    #     head_num = 32
    #     head_num_kv = 8
    #     size_per_head = 128
    #     seq_size_per_block = 64
    #     max_seq_len = max(input_lengths)

    #     attn_configs = self._create_config(
    #         head_num=head_num,
    #         head_num_kv=head_num_kv,
    #         size_per_head=size_per_head,
    #         seq_size_per_block=seq_size_per_block,
    #     )

    #     attn_inputs = self._create_prefill_attention_inputs_padded(
    #         batch_size, input_lengths, max_seq_len, seq_size_per_block, prefix_lengths=None
    #     )

    #     attn_op = TRTAttnOp(attn_configs)

    #     self.run_correctness_test(
    #         attn_op=attn_op,
    #         op_name="TRTAttnOp",
    #         batch_size=batch_size,
    #         input_lengths=input_lengths,
    #         head_num=head_num,
    #         head_num_kv=head_num_kv,
    #         size_per_head=size_per_head,
    #         seq_size_per_block=seq_size_per_block,
    #         attn_configs=attn_configs,
    #         attn_inputs=attn_inputs,
    #         prefix_lengths=None,
    #         use_padded=True,
    #     )

    # def test_long_sequence(self):
    #     """Test TRTAttnOp with padded mode and long sequences"""
    #     print("\n=== Test TRTAttnOp Padded: Long Sequence ===", flush=True)

    #     batch_size = 2
    #     input_lengths = [512, 1024]
    #     head_num = 32
    #     head_num_kv = 8
    #     size_per_head = 128
    #     seq_size_per_block = 64
    #     max_seq_len = max(input_lengths)

    #     attn_configs = self._create_config(
    #         head_num=head_num,
    #         head_num_kv=head_num_kv,
    #         size_per_head=size_per_head,
    #         seq_size_per_block=seq_size_per_block,
    #     )

    #     attn_inputs = self._create_prefill_attention_inputs_padded(
    #         batch_size, input_lengths, max_seq_len, seq_size_per_block, prefix_lengths=None
    #     )

    #     attn_op = TRTAttnOp(attn_configs)

    #     self.run_correctness_test(
    #         attn_op=attn_op,
    #         op_name="TRTAttnOp",
    #         batch_size=batch_size,
    #         input_lengths=input_lengths,
    #         head_num=head_num,
    #         head_num_kv=head_num_kv,
    #         size_per_head=size_per_head,
    #         seq_size_per_block=seq_size_per_block,
    #         attn_configs=attn_configs,
    #         attn_inputs=attn_inputs,
    #         prefix_lengths=None,
    #         use_padded=True,
    #     )

    # def test_consistency_with_nonpadded(self):
    #     """Test consistency between padded and non-padded modes"""
    #     print("\n=== Test TRTAttnOp Padded: Consistency ===", flush=True)

    #     from rtp_llm.models_py.modules.factory.attention.cuda_impl.test.base_attention_test import set_seed, compare_tensors
    #     import torch
    #     import math
    #     from rtp_llm.ops.compute_ops import get_typemeta

    #     batch_size = 2
    #     input_lengths = [64, 128]
    #     head_num = 32
    #     head_num_kv = 8
    #     size_per_head = 128
    #     seq_size_per_block = 64
    #     max_seq_len = max(input_lengths)

    #     set_seed(42)

    #     attn_configs = self._create_config(
    #         head_num=head_num,
    #         head_num_kv=head_num_kv,
    #         size_per_head=size_per_head,
    #         seq_size_per_block=seq_size_per_block,
    #     )

    #     # Create inputs for both modes
    #     attn_inputs_nonpadded = self._create_prefill_attention_inputs(
    #         batch_size, input_lengths, seq_size_per_block, prefix_lengths=None
    #     )
    #     attn_inputs_padded = self._create_prefill_attention_inputs_padded(
    #         batch_size, input_lengths, max_seq_len, seq_size_per_block, prefix_lengths=None
    #     )

    #     # Create identical QKV
    #     total_tokens = sum(input_lengths)

    #     set_seed(123)
    #     qkv = self._create_qkv_tensor(
    #         total_tokens, head_num, head_num_kv, size_per_head, dtype=attn_configs.dtype
    #     )

    #     attn_inputs_nonpadded.dtype = get_typemeta(qkv)
    #     attn_inputs_padded.dtype = get_typemeta(qkv)

    #     attn_op = TRTAttnOp(attn_configs)

    #     # Prepare and run both modes
    #     params_nonpadded = attn_op.prepare(attn_inputs_nonpadded)
    #     params_padded = attn_op.prepare(attn_inputs_padded)

    #     # Non-padded KV cache
    #     total_blocks_nonpadded = sum(
    #         [math.ceil(seq_len / seq_size_per_block) for seq_len in input_lengths]
    #     )
    #     kv_cache_nonpadded, _, _ = self._create_kv_cache(
    #         total_blocks_nonpadded, seq_size_per_block, head_num_kv, size_per_head, dtype=torch.float16
    #     )

    #     # Padded KV cache
    #     max_blocks_per_seq = math.ceil(max_seq_len / seq_size_per_block)
    #     total_blocks_padded = max_blocks_per_seq * batch_size
    #     kv_cache_padded, _, _ = self._create_kv_cache(
    #         total_blocks_padded, seq_size_per_block, head_num_kv, size_per_head, dtype=torch.float16
    #     )

    #     # Run forward pass
    #     output_nonpadded = attn_op.forward(qkv.clone(), kv_cache_nonpadded, params_nonpadded)
    #     output_padded = attn_op.forward(qkv.clone(), kv_cache_padded, params_padded)

    #     # Compare outputs
    #     self.assertEqual(output_nonpadded.shape, output_padded.shape)

    #     max_diff = torch.max(torch.abs(output_nonpadded - output_padded)).item()
    #     mean_diff = torch.mean(torch.abs(output_nonpadded - output_padded)).item()

    #     print(
    #         f"Consistency check:\n"
    #         f"  Max diff: {max_diff:.6e}\n"
    #         f"  Mean diff: {mean_diff:.6e}", flush=True
    #     )

    #     compare_tensors(
    #         output_nonpadded,
    #         output_padded,
    #         rtol=1e-3,
    #         atol=1e-5,
    #         name="TRTAttnOp padded vs non-padded",
    #     )

    #     print("✓ Consistency check passed!", flush=True)


if __name__ == "__main__":
    unittest.main()
