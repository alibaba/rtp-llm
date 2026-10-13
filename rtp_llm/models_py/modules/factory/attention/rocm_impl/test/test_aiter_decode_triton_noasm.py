"""Layout-aware parity test for AITER Triton vs Non-ASM decode.

This test isolates decode FMHA kernels at the paged-attention boundary and
feeds each kernel the physical KV layout it expects, while preserving the same
semantic K/V values.
"""

from __future__ import annotations

import math
import unittest

import torch

_IS_ROCM_BUILD = torch.version.hip is not None
try:
    from rtp_llm.models_py.modules.factory.attention import attn_factory
    from rtp_llm.models_py.modules.factory.attention.rocm_impl.aiter import (
        AiterDecodeAttnOpNonAsm,
        AiterDecodeAttnOpTriton,
        AiterDecodeImplNonAsm,
        AiterDecodeImplTriton,
    )
    from rtp_llm.ops import AttentionConfigs, FMHAConfig, KvCacheDataType
    from rtp_llm.ops.compute_ops import (
        FusedRopeKVCacheDecodeOpAsm,
        FusedRopeKVCacheDecodeOpNonAsm,
        LayerKVCache,
        PyAttentionInputs,
        get_typemeta,
    )

except ImportError:
    if _IS_ROCM_BUILD:
        raise
    _ROCM_IMPORTS_AVAILABLE = False
else:
    _ROCM_IMPORTS_AVAILABLE = True

HEAD_NUM = 24
KV_HEAD_NUM = 4
HEAD_DIM = 256
BLOCK_SIZE = 16
CONTEXT_LENGTH = 6359
NUM_BLOCKS = math.ceil(CONTEXT_LENGTH / BLOCK_SIZE)


def make_config(block_size: int = BLOCK_SIZE) -> AttentionConfigs:
    config = AttentionConfigs()
    config.head_num = HEAD_NUM
    config.kv_head_num = KV_HEAD_NUM
    config.size_per_head = HEAD_DIM
    config.tokens_per_block = block_size
    config.kernel_tokens_per_block = block_size
    config.max_seq_len = 40960
    config.kv_cache_dtype = KvCacheDataType.BASE
    config.dtype = torch.bfloat16
    config.need_rope_kv_cache = True
    return config


def make_inputs(device: torch.device, block_size: int = BLOCK_SIZE):
    inputs = PyAttentionInputs()
    inputs.is_prefill = False
    inputs.is_cuda_graph = False
    # Decode sees full context after current token is inserted into KV cache.
    inputs.sequence_lengths = torch.tensor([CONTEXT_LENGTH - 1], dtype=torch.int32)
    inputs.input_lengths = torch.tensor([1], dtype=torch.int32)
    num_blocks = math.ceil(CONTEXT_LENGTH / block_size)
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).view(1, -1)
    inputs.kv_cache_kernel_block_id_device = block_table
    inputs.kv_cache_kernel_block_id = block_table.cpu()
    inputs.kv_cache_block_id_device = block_table
    inputs.dtype = get_typemeta(torch.empty((), dtype=torch.bfloat16))
    return inputs


def make_impl(impl_class, op, inputs):
    # Bypass RoPE/cache insertion and test decode FMHA kernels directly.
    impl = impl_class.__new__(impl_class)
    impl.need_rope_kv_cache = False
    impl.fmha_impl = op
    impl.attn_inputs = inputs
    impl.fmha_params = impl.fmha_impl.prepare(inputs)
    impl.write_cache_store_impl = None
    return impl


def run_impl(impl_class, op, inputs, query, kv_cache):
    cache = LayerKVCache()
    cache.kv_cache_base = kv_cache.clone()
    cache.kv_scale_base = torch.empty(0, device=query.device)
    impl = make_impl(impl_class, op, inputs)
    return impl.forward(query.clone(), cache, layer_idx=3)


def pack_cache(key_phys: torch.Tensor, value_phys: torch.Tensor) -> torch.Tensor:
    return torch.stack([key_phys, value_phys], dim=1)


def physical_key_for_decode(semantic_key: torch.Tensor) -> torch.Tensor:
    # Both decode paths reinterpret K as vectorized [hd//x, ps, x] via a view.
    # Keeping K in canonical [ps, hd] memory order is enough.
    return semantic_key.contiguous()


def physical_value_for_nonasm(semantic_value: torch.Tensor) -> torch.Tensor:
    # Non-ASM paged_attention_rocm reads BASE V as linear [hd, ps].
    return (
        semantic_value.permute(0, 1, 3, 2)
        .contiguous()
        .view(NUM_BLOCKS, KV_HEAD_NUM, BLOCK_SIZE, HEAD_DIM)
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


@unittest.skipUnless(torch.cuda.is_available() and _IS_ROCM_BUILD, "Requires ROCm GPU")
@unittest.skipUnless(_ROCM_IMPORTS_AVAILABLE, "Requires ROCm attention modules")
class AiterDecodeLayoutParityTest(unittest.TestCase):
    @staticmethod
    def _relative_l2(actual: torch.Tensor, reference: torch.Tensor) -> float:
        reference = reference.float().flatten()
        diff = actual.float().flatten() - reference
        return (diff.norm() / reference.norm()).item()

    def test_triton_matches_nonasm_with_layout_aware_cache(self):
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
        nonasm_cache = pack_cache(
            key_phys, physical_value_for_nonasm(semantic_value)
        ).flatten(1)
        triton_cache = pack_cache(
            key_phys, physical_value_for_triton(semantic_value)
        ).flatten(1)

        config = make_config()
        inputs = make_inputs(query.device)
        nonasm_output = run_impl(
            AiterDecodeImplNonAsm,
            AiterDecodeAttnOpNonAsm(config),
            inputs,
            query,
            nonasm_cache,
        )

        def run_triton(cache, linear_v):
            return run_impl(
                AiterDecodeImplTriton,
                AiterDecodeAttnOpTriton(config, linear_v=linear_v),
                inputs,
                query,
                cache,
            )

        # Check both Triton V-reader contracts against the same Non-ASM reference.
        for cache, linear_v in ((triton_cache, False), (nonasm_cache, True)):
            with self.subTest(linear_v=linear_v):
                relative_l2 = self._relative_l2(
                    run_triton(cache, linear_v), nonasm_output
                )
                self.assertLess(
                    relative_l2,
                    0.01,
                    "layout-aware comparison still mismatches: "
                    f"linear_v={linear_v}, relative_l2={relative_l2:.6f}",
                )

        # Pairing the vectorized reader with the linear cache must diverge,
        # otherwise the two assertions above would hold for any reader.
        mismatched_l2 = self._relative_l2(
            run_triton(nonasm_cache, False), nonasm_output
        )
        self.assertGreater(mismatched_l2, 0.1, f"{mismatched_l2=:.6f}")

    def test_factory_pairs_reader_and_writer(self):
        cases = (
            (KvCacheDataType.BASE, False, 16, True, FusedRopeKVCacheDecodeOpNonAsm),
            (KvCacheDataType.BASE, True, 32, False, FusedRopeKVCacheDecodeOpAsm),
            (KvCacheDataType.FP8, False, 16, False, FusedRopeKVCacheDecodeOpAsm),
            (KvCacheDataType.FP8, False, 32, False, FusedRopeKVCacheDecodeOpAsm),
        )
        for kv_dtype, use_asm_pa, page, linear_v, writer in cases:
            with self.subTest(kv_dtype=kv_dtype, page=page):
                config = make_config(page)
                config.kv_cache_dtype = kv_dtype
                fmha_config = FMHAConfig()
                fmha_config.use_aiter_pa = True
                fmha_config.use_asm_pa = use_asm_pa
                fmha_config.use_triton_pa = True
                inputs = make_inputs(torch.device("cuda"), page)
                impl = attn_factory.get_fmha_impl(
                    config, None, inputs, fmha_config=fmha_config
                )
                self.assertIsInstance(impl, AiterDecodeImplTriton)
                self.assertIs(impl.fmha_impl.linear_v, linear_v)
                self.assertIs(type(impl.rope_kvcache_impl), writer)


if __name__ == "__main__":
    unittest.main()
