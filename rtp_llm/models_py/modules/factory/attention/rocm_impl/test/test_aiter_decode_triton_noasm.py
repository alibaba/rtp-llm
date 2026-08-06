"""Layout-aware parity test for AITER Triton vs Non-ASM decode.

This test isolates decode FMHA kernels at the paged-attention boundary and
feeds each kernel the physical KV layout it expects, while preserving the same
semantic K/V values.
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
    # Decode sees full context after current token is inserted into KV cache.
    inputs.sequence_lengths = torch.tensor([CONTEXT_LENGTH - 1], dtype=torch.int32)
    inputs.input_lengths = torch.tensor([1], dtype=torch.int32)
    block_table = torch.arange(NUM_BLOCKS, dtype=torch.int32, device=device).view(1, -1)
    inputs.kv_cache_kernel_block_id_device = block_table
    inputs.kv_cache_block_id_device = block_table
    inputs.dtype = get_typemeta(torch.empty((), dtype=torch.bfloat16))
    inputs.cache_store_inputs = None
    return inputs


def make_impl(impl_class, op_class, config, inputs):
    # Bypass RoPE/cache insertion and test decode FMHA kernels directly.
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


def pack_cache(key_phys: torch.Tensor, value_phys: torch.Tensor) -> torch.Tensor:
    return torch.stack([key_phys, value_phys], dim=1)


def physical_key_for_decode(semantic_key: torch.Tensor) -> torch.Tensor:
    # Both decode paths reinterpret K as vectorized [hd//x, ps, x] via a view.
    # Keeping K in canonical [ps, hd] memory order is enough.
    return semantic_key.contiguous()


def physical_value_for_nonasm(semantic_value: torch.Tensor) -> torch.Tensor:
    # Non-ASM paged_attention_rocm reads BASE V as linear [hd, ps].
    return semantic_value.permute(0, 1, 3, 2).contiguous().view(
        NUM_BLOCKS, KV_HEAD_NUM, BLOCK_SIZE, HEAD_DIM
    )


def physical_value_for_triton(semantic_value: torch.Tensor) -> torch.Tensor:
    # Triton pa_decode_gluon (VALUE_TRANSPOSED=True) reads V as [ps//x, hd, x].
    x_vec = 16 // semantic_value.element_size()
    assert BLOCK_SIZE % x_vec == 0
    return (
        semantic_value.view(
            NUM_BLOCKS, KV_HEAD_NUM, BLOCK_SIZE // x_vec, x_vec, HEAD_DIM
        )
        .permute(0, 1, 2, 4, 3)
        .contiguous()
        .view(NUM_BLOCKS, KV_HEAD_NUM, BLOCK_SIZE, HEAD_DIM)
    )


class AiterDecodeLayoutParityTest(unittest.TestCase):
    def test_layout_mismatch_reproduces_large_error(self):
        if not torch.cuda.is_available() or torch.version.hip is None:
            self.skipTest("requires a ROCm GPU")

        generator = torch.Generator().manual_seed(0)
        query = torch.randn(
            (1, HEAD_NUM, HEAD_DIM), generator=generator, dtype=torch.bfloat16
        ).cuda()

        # A single shared physical V layout is not comparable across kernels.
        shared_cache = torch.randn(
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
            shared_cache,
        )
        nonasm_output = run_impl(
            AiterDecodeImplNonAsm,
            AiterDecodeAttnOpNonAsm,
            config,
            inputs,
            query,
            shared_cache,
        )

        diff = (triton_output.float() - nonasm_output.float()).flatten()
        relative_l2 = diff.norm() / nonasm_output.float().flatten().norm()
        self.assertGreater(
            relative_l2.item(),
            0.5,
            f"unexpectedly small mismatch: relative_l2={relative_l2.item():.6f}",
        )

    def test_triton_matches_nonasm_with_layout_aware_cache(self):
        if not torch.cuda.is_available() or torch.version.hip is None:
            self.skipTest("requires a ROCm GPU")

        generator = torch.Generator().manual_seed(0)
        query = torch.randn(
            (1, HEAD_NUM, HEAD_DIM), generator=generator, dtype=torch.bfloat16
        ).cuda()

        # Generate shared semantic KV, then materialize path-specific physical layout.
        semantic_key = torch.randn(
            (NUM_BLOCKS, KV_HEAD_NUM, BLOCK_SIZE, HEAD_DIM),
            generator=generator,
            dtype=torch.bfloat16,
        ).cuda()
        semantic_value = torch.randn(
            (NUM_BLOCKS, KV_HEAD_NUM, BLOCK_SIZE, HEAD_DIM),
            generator=generator,
            dtype=torch.bfloat16,
        ).cuda()

        key_phys = physical_key_for_decode(semantic_key)
        nonasm_cache = pack_cache(key_phys, physical_value_for_nonasm(semantic_value))
        triton_cache = pack_cache(key_phys, physical_value_for_triton(semantic_value))

        config = make_config()
        inputs = make_inputs(query.device)
        triton_output = run_impl(
            AiterDecodeImplTriton,
            AiterDecodeAttnOpTriton,
            config,
            inputs,
            query,
            triton_cache,
        )
        nonasm_output = run_impl(
            AiterDecodeImplNonAsm,
            AiterDecodeAttnOpNonAsm,
            config,
            inputs,
            query,
            nonasm_cache,
        )

        diff = (triton_output.float() - nonasm_output.float()).flatten()
        relative_l2 = diff.norm() / nonasm_output.float().flatten().norm()
        self.assertLess(
            relative_l2.item(),
            0.01,
            "layout-aware comparison still mismatches: "
            f"relative_l2={relative_l2.item():.6f}",
        )


if __name__ == "__main__":
    unittest.main()
