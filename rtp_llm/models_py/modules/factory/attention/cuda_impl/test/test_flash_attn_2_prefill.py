"""Tests for FlashAttn2PrefillImpl (sm120 eager varlen prefill, no prefix).

Correctness follows the same reference path as the fmha_v2 impl tests:
apply_base_rope_to_qkv_reference + compute_pytorch_prefill_reference. The
impl is constructed directly — support() is sm12x-gated for factory
routing, but the FA2 kernels are arch-portable (Ampere and later), so the
math is validated on whichever GPU runs the test.
"""

import unittest

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_impl.flash_attn_2 import (
    FlashAttn2PrefillImpl,
    _HAS_FLASH_ATTN_2,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.test.trt_tests.test_trt_base import (
    TRTLLMFMHAv2TestBase,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.test.trt_tests.trt_test_utils import (
    apply_base_rope_to_qkv_reference,
    compute_pytorch_prefill_reference,
)
from rtp_llm.ops import RopeStyle
from rtp_llm.ops.compute_ops import get_typemeta


@unittest.skipUnless(_HAS_FLASH_ATTN_2, "flash_attn 2 not available")
class TestFlashAttn2PrefillImpl(TRTLLMFMHAv2TestBase):
    def _run_impl_rope_correctness(
        self,
        rope_style,
        input_lengths,
        head_num,
        head_num_kv,
        head_dim,
    ):
        tokens_per_block = 64
        rope_base = 10000
        batch_size = len(input_lengths)
        total_tokens = sum(input_lengths)

        attn_configs = self._create_config(
            head_num=head_num,
            head_num_kv=head_num_kv,
            size_per_head=head_dim,
            seq_size_per_block=tokens_per_block,
        )
        attn_configs.rope_config.style = rope_style
        attn_configs.rope_config.dim = head_dim
        attn_configs.rope_config.base = rope_base
        attn_configs.rope_config.max_pos = max(input_lengths)
        attn_configs.max_seq_len = max(input_lengths)
        attn_inputs = self._create_prefill_attention_inputs(
            batch_size,
            input_lengths,
            tokens_per_block,
            prefix_lengths=None,
        )

        qkv = self._create_qkv_tensor(
            total_tokens,
            head_num,
            head_num_kv,
            head_dim,
            dtype=attn_configs.dtype,
        )
        attn_inputs.dtype = get_typemeta(qkv)

        expected_qkv = qkv
        if rope_style != RopeStyle.No:
            expected_qkv = apply_base_rope_to_qkv_reference(
                qkv,
                input_lengths,
                head_num,
                head_num_kv,
                head_dim,
                rope_base,
            )

        total_blocks = self._calculate_total_blocks(
            input_lengths, tokens_per_block
        )
        kv_cache, _, _ = self._create_kv_cache(
            total_blocks,
            tokens_per_block,
            head_num_kv,
            head_dim,
            dtype=attn_configs.dtype,
        )
        kv_cache.kv_cache_base = kv_cache.kv_cache_base.flatten(1)

        impl = FlashAttn2PrefillImpl(attn_configs, attn_inputs)
        actual = impl.forward(qkv.clone(), kv_cache, layer_idx=0)

        expected = compute_pytorch_prefill_reference(
            expected_qkv,
            input_lengths,
            head_num,
            head_num_kv,
            head_dim,
        )
        torch.testing.assert_close(actual, expected, rtol=5e-3, atol=5e-3)

    def test_impl_without_rope_matches_torch(self):
        self._run_impl_rope_correctness(
            RopeStyle.No, [128, 64], head_num=32, head_num_kv=8, head_dim=64
        )

    def test_impl_with_rope_matches_torch(self):
        self._run_impl_rope_correctness(
            RopeStyle.Base, [128, 64], head_num=32, head_num_kv=8, head_dim=64
        )

    def test_impl_gqa_head128_matches_torch(self):
        self._run_impl_rope_correctness(
            RopeStyle.Base, [96, 33], head_num=8, head_num_kv=2, head_dim=128
        )

    def test_impl_mha_matches_torch(self):
        self._run_impl_rope_correctness(
            RopeStyle.Base, [64], head_num=8, head_num_kv=8, head_dim=64
        )


if __name__ == "__main__":
    unittest.main()
