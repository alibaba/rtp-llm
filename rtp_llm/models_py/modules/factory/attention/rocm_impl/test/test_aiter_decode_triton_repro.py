"""Minimal reproducer for AITER Triton paged-attention decode corruption.

The tensor geometry matches a Qwen3.5 decode request.  Inputs are generated from
a fixed seed so the test is self-contained and can be shared outside Alibaba.
"""

import math
import unittest

import torch

from rtp_llm.models_py.modules.factory.attention.rocm_impl.aiter import (
    AiterDecodeAttnOpNonAsm,
    AiterDecodeAttnOpTriton,
    AiterDecodeImplNonAsm,
    AiterDecodeImplTriton,
)
from rtp_llm.ops import AttentionConfigs, KvCacheDataType
from rtp_llm.ops.compute_ops import LayerKVCache, PyAttentionInputs, get_typemeta

HEAD_NUM = 24
KV_HEAD_NUM = 4
HEAD_DIM = 256
BLOCK_SIZE = 16
CONTEXT_LENGTH = 6359
NUM_BLOCKS = math.ceil(CONTEXT_LENGTH / BLOCK_SIZE)


def make_config() -> AttentionConfigs:
    config = AttentionConfigs()
    config.head_num = HEAD_NUM
    config.kv_head_num = KV_HEAD_NUM
    config.size_per_head = HEAD_DIM
    config.tokens_per_block = BLOCK_SIZE
    config.kernel_tokens_per_block = BLOCK_SIZE
    config.max_seq_len = 40960
    config.kv_cache_dtype = KvCacheDataType.BASE
    config.dtype = torch.bfloat16
    config.need_rope_kv_cache = False
    return config


def make_inputs(device: torch.device) -> PyAttentionInputs:
    inputs = PyAttentionInputs()
    inputs.is_prefill = False
    inputs.is_cuda_graph = False
    # The current token has already been inserted into the cache.  The attention
    # kernel sees the complete 6359-token context, while RTP-LLM's input length
    # before cache insertion is 6358.
    inputs.sequence_lengths = torch.tensor([CONTEXT_LENGTH - 1], dtype=torch.int32)
    inputs.input_lengths = torch.tensor([1], dtype=torch.int32)
    block_table = torch.arange(NUM_BLOCKS, dtype=torch.int32, device=device).view(1, -1)
    inputs.kv_cache_kernel_block_id_device = block_table
    inputs.kv_cache_block_id_device = block_table
    inputs.dtype = get_typemeta(torch.empty((), dtype=torch.bfloat16))
    inputs.cache_store_inputs = None
    return inputs


def make_impl(impl_class, op_class, config, inputs):
    # RoPE/cache insertion is intentionally bypassed: inputs represent tensors at
    # the paged-attention boundary, isolating only the decode kernel.
    impl = impl_class.__new__(impl_class)
    impl.need_rope_kv_cache = False
    impl.fmha_impl = op_class(config)
    impl.attn_inputs = inputs
    impl.fmha_params = impl.fmha_impl.prepare(inputs)
    impl.write_cache_store_impl = None
    return impl


def run_impl(impl_class, op_class, config, inputs, query, kv_cache):
    cache = LayerKVCache()
    cache.kv_cache_base = kv_cache.clone()
    cache.kv_scale_base = torch.empty(0, device=query.device)
    impl = make_impl(impl_class, op_class, config, inputs)
    return impl.forward(query.clone(), cache, layer_idx=3)


class AiterDecodeTritonReproTest(unittest.TestCase):
    def test_triton_matches_nonasm_for_qwen35_shape(self):
        if not torch.cuda.is_available() or torch.version.hip is None:
            self.skipTest("requires a ROCm GPU")

        generator = torch.Generator().manual_seed(0)
        query = torch.randn(
            (1, HEAD_NUM, HEAD_DIM), generator=generator, dtype=torch.bfloat16
        ).cuda()
        kv_cache = torch.randn(
            (NUM_BLOCKS, 2, KV_HEAD_NUM, BLOCK_SIZE, HEAD_DIM),
            generator=generator,
            dtype=torch.bfloat16,
        ).cuda()
        config = make_config()
        inputs = make_inputs(query.device)

        triton_output = run_impl(
            AiterDecodeImplTriton,
            AiterDecodeAttnOpTriton,
            config,
            inputs,
            query,
            kv_cache,
        )
        nonasm_output = run_impl(
            AiterDecodeImplNonAsm,
            AiterDecodeAttnOpNonAsm,
            config,
            inputs,
            query,
            kv_cache,
        )

        diff = (triton_output.float() - nonasm_output.float()).flatten()
        relative_l2 = diff.norm() / nonasm_output.float().flatten().norm()
        self.assertLess(
            relative_l2.item(),
            0.01,
            "AITER Triton decode disagrees with paged_attention_rocm: "
            f"relative_l2={relative_l2.item():.6f}",
        )


if __name__ == "__main__":
    unittest.main()
