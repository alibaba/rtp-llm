import math
import os
import random
import sys
from typing import List, Optional
from unittest import SkipTest, TestCase, main

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_impl.test.atten_test_util import (
    attention_prefill_ref,
    gen_attention_inputs,
    write_kv_cache,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.trtllm_gen import (
    FlashInferTRTLLMDecodeOp,
    FlashInferTRTLLMPrefillOp,
)
from rtp_llm.test.utils.numeric_util import assert_close_with_mismatch_tolerance

device = torch.device("cuda")

from rtp_llm.ops import AttentionConfigs, KvCacheDataType
from rtp_llm.ops.compute_ops import KVCache, PyAttentionInputs


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class FlashInferPythonMHATest(TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        self.device = torch.device("cuda")
        set_seed(25536)
        self.num_pages = 1024
        self.page_size = 64
        self.head_dim = 128
        self.num_kv_heads = 8
        self.num_heads = 64

        self.k_cache = (
            torch.rand(
                self.num_pages,
                self.num_kv_heads,
                self.page_size,
                self.head_dim,
                dtype=torch.bfloat16,
                device="cuda:0",
            )
            * 2
            - 1
        ).to(torch.float8_e4m3fn)
        self.v_cache = (
            torch.rand(
                self.num_pages,
                self.num_kv_heads,
                self.page_size,
                self.head_dim,
                dtype=torch.bfloat16,
                device="cuda:0",
            )
            * 2
            - 1
        ).to(torch.float8_e4m3fn)
        self.kv_cache: KVCache = KVCache()
        self.kv_cache.k_cache_base = torch.stack([self.k_cache, self.v_cache], dim=1)
        self.kv_cache.v_cache_base = self.v_cache

    def _create_config(self) -> AttentionConfigs:
        """Create a standard AttentionConfigs config for testing."""
        config = AttentionConfigs()
        config.head_num = self.num_heads
        config.kv_head_num = self.num_kv_heads
        config.size_per_head = self.head_dim
        config.tokens_per_block = self.page_size
        config.kv_cache_dtype = KvCacheDataType.FP8
        config.use_mla = False
        config.is_causal = True
        config.fuse_qkv_add_bias = True
        config.q_scaling = 1.0
        return config

    def test_flashinfer_trtllm_prefill(self):
        """Test FlashInferTRTLLM prefill attention with reference comparison."""
        # Check if SM_100 is available
        is_sm_100 = torch.cuda.get_device_capability()[0] in [10]
        if not is_sm_100:
            raise SkipTest("FlashInferTRTLLM requires SM_100 (compute capability 10.0)")
        input_lengths = [2, 129, 255, 63]
        num_tokens = sum(input_lengths)
        config = self._create_config()
        attn_inputs = gen_attention_inputs(
            self.page_size, self.num_pages, input_lengths=input_lengths
        )
        hidden_size = self.head_dim * self.num_heads
        qkv = (
            torch.rand(
                [
                    num_tokens,
                    hidden_size + 2 * self.num_kv_heads * self.head_dim,
                ],
                dtype=torch.bfloat16,
                device=self.device,
            )
            * 2
            - 1
        )

        # Split qkv for reference implementation
        q_size = self.head_dim * self.num_heads
        k_size = self.head_dim * self.num_kv_heads
        v_size = self.head_dim * self.num_kv_heads
        q_ref = qkv[:, :q_size].reshape(num_tokens, self.num_heads, self.head_dim)
        k_ref = qkv[:, q_size : q_size + k_size].reshape(
            num_tokens, self.num_kv_heads, self.head_dim
        )
        v_ref = qkv[:, q_size + k_size : q_size + k_size + v_size].reshape(
            num_tokens, self.num_kv_heads, self.head_dim
        )

        write_kv_cache(
            k_ref,
            v_ref,
            self.kv_cache,
            attn_inputs.input_lengths,
            attn_inputs.kv_cache_block_id_host,
        )
        # Run FlashInferTRTLLM implementation
        op = FlashInferTRTLLMPrefillOp(config)
        input_params = op.prepare(attn_inputs)
        out_trtllm = op.forward(q_ref, self.kv_cache, input_params)
        # Run reference implementation
        out_ref = attention_prefill_ref(
            q_ref,
            k_ref,
            v_ref,
            attn_inputs.sequence_lengths,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            causal=True,
        )

        # Reshape output to match reference
        out_trtllm_reshaped = out_trtllm.reshape(
            num_tokens, self.num_heads, self.head_dim
        )
        # Convert to float32 for comparison
        out_trtllm_f32 = out_trtllm_reshaped.float()
        out_ref_f32 = out_ref.float()
        atol = 0.04
        rtol = 0.04
        allowed_mismatch_rate = 1e-5
        assert_close_with_mismatch_tolerance(
            out_trtllm_f32,
            out_ref_f32,
            atol=atol,
            rtol=rtol,
            max_mismatched_elements=int(allowed_mismatch_rate * out_ref_f32.numel()),
        )

    def test_flashinfer_trtllm_decode(self):
        """Test FlashInferTRTLLM decode attention with reference comparison."""
        # Check if SM_100 is available
        is_sm_100 = torch.cuda.get_device_capability()[0] in [10]
        if not is_sm_100:
            raise SkipTest("FlashInferTRTLLM requires SM_100 (compute capability 10.0)")
        sequence_lengths = [2, 129, 255, 63]
        batch_size = len(sequence_lengths)
        num_tokens = sum(sequence_lengths)
        config = self._create_config()
        attn_inputs = gen_attention_inputs(
            self.page_size, self.num_pages, sequence_lengths=sequence_lengths
        )
        hidden_size = self.head_dim * self.num_heads
        qkv = (
            torch.rand(
                [
                    num_tokens,
                    hidden_size + 2 * self.num_kv_heads * self.head_dim,
                ],
                dtype=torch.bfloat16,
                device=self.device,
            )
            * 2
            - 1
        )
        # Split qkv for reference implementation
        q_size = self.head_dim * self.num_heads
        k_size = self.head_dim * self.num_kv_heads
        v_size = self.head_dim * self.num_kv_heads
        q_ref = qkv[:, :q_size].reshape(num_tokens, self.num_heads, self.head_dim)
        k_ref = qkv[:, q_size : q_size + k_size].reshape(
            num_tokens, self.num_kv_heads, self.head_dim
        )
        v_ref = qkv[:, q_size + k_size : q_size + k_size + v_size].reshape(
            num_tokens, self.num_kv_heads, self.head_dim
        )
        last_token_idx = attn_inputs.cu_seqlens[1:] - 1
        # Run reference implementation
        out_ref = attention_prefill_ref(
            q_ref,
            k_ref,
            v_ref,
            attn_inputs.sequence_lengths,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            causal=True,
        )

        out_ref = out_ref[last_token_idx]
        write_kv_cache(
            k_ref,
            v_ref,
            self.kv_cache,
            attn_inputs.sequence_lengths,
            attn_inputs.kv_cache_block_id_host,
        )

        op = FlashInferTRTLLMDecodeOp(config)
        q = q_ref[last_token_idx]
        attn_inputs.sequence_lengths -= 1
        input_params = op.prepare(attn_inputs)
        out_trtllm = op.forward(q, self.kv_cache, input_params)
        # Reshape output to match reference
        out_trtllm_reshaped = out_trtllm.reshape(-1, self.num_heads, self.head_dim)
        out_trtllm_f32 = out_trtllm_reshaped.float()
        out_ref_f32 = out_ref.float()
        atol = 0.04  # More relaxed for float8
        rtol = 0.04
        allowed_mismatch_rate = 1e-3
        assert_close_with_mismatch_tolerance(
            out_trtllm_f32,
            out_ref_f32,
            atol=atol,
            rtol=rtol,
            max_mismatched_elements=int(allowed_mismatch_rate * out_ref_f32.numel()),
        )


if __name__ == "__main__":
    main()
