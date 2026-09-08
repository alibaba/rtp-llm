"""Layout-aware parity test for AITER Triton vs Non-ASM decode.

This test isolates decode FMHA kernels at the paged-attention boundary and
feeds each kernel the physical KV layout it expects, while preserving the same
semantic K/V values.
"""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import torch

_IS_ROCM_BUILD = torch.version.hip is not None
try:
    from rtp_llm.models_py.modules.factory.attention import attn_factory
    from rtp_llm.models_py.modules.factory.attention.rocm_impl.aiter import (
        AiterDecodeAttnOpNonAsm,
        AiterDecodeAttnOpTriton,
        AiterDecodeImplNonAsm,
        AiterDecodeImplTriton,
        FMHAParams,
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
    def test_graph_replay_reads_device_lengths_with_stale_host_mirrors(self):
        inputs = make_inputs(torch.device("cuda"))
        inputs.sequence_lengths = torch.tensor([100, 100], dtype=torch.int32)
        inputs.input_lengths = torch.ones(2, dtype=torch.int32)
        inputs.sequence_lengths_plus_1_device = torch.tensor(
            [101, 1], dtype=torch.int32, device="cuda"
        )
        impl = AiterDecodeImplNonAsm.__new__(AiterDecodeImplNonAsm)
        impl.fmha_params = FMHAParams(inputs, is_prefill=False, graph_max_seq_len=40960)
        rope_lengths = torch.empty_like(impl.fmha_params.seq_lens)
        impl.rope_params = SimpleNamespace(
            update_kv_cache_offset=lambda table: None,
            update_decode_lengths=lambda lengths: rope_lengths.copy_(lengths - 1),
        )
        address = impl.fmha_params.seq_lens.data_ptr()
        output = torch.empty_like(impl.fmha_params.seq_lens)
        rope_output = torch.empty_like(rope_lengths)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            impl.prepare_cuda_graph(inputs)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output.copy_(impl.fmha_params.seq_lens)
                rope_output.copy_(rope_lengths)
            # Model several queued steps and a rounded-up batch slot. Host
            # mirrors deliberately stay at capture-time values throughout.
            for lengths in ([102, 1], [103, 79], [104, 1]):
                inputs.sequence_lengths_plus_1_device.copy_(
                    torch.tensor(lengths, dtype=torch.int32, device="cuda")
                )
                impl.prepare_cuda_graph(inputs)
                graph.replay()
                self.assertEqual(output.cpu().tolist(), lengths)
                self.assertEqual(
                    rope_output.cpu().tolist(), [length - 1 for length in lengths]
                )
                self.assertEqual(impl.fmha_params.seq_lens.data_ptr(), address)
                self.assertEqual(impl.fmha_params.max_seq_len, 40960)
        torch.cuda.current_stream().wait_stream(stream)

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
        inputs = make_inputs(torch.device("cuda"))
        for kv_dtype, use_asm_pa, expected_linear_v, expected_writer in (
            (KvCacheDataType.BASE, False, True, FusedRopeKVCacheDecodeOpNonAsm),
            (KvCacheDataType.BASE, True, False, FusedRopeKVCacheDecodeOpAsm),
            (KvCacheDataType.FP8, False, False, FusedRopeKVCacheDecodeOpAsm),
        ):
            with self.subTest(kv_dtype=kv_dtype, use_asm_pa=use_asm_pa):
                config = make_config()
                config.kv_cache_dtype = kv_dtype
                fmha_config = FMHAConfig()
                fmha_config.use_aiter_pa = True
                fmha_config.use_asm_pa = use_asm_pa
                fmha_config.use_triton_pa = True
                impl = attn_factory.get_fmha_impl(
                    config, None, inputs, fmha_config=fmha_config
                )
                self.assertIsInstance(impl, AiterDecodeImplTriton)
                self.assertEqual(impl.fmha_impl.linear_v, expected_linear_v)
                self.assertIs(type(impl.rope_kvcache_impl), expected_writer)


if __name__ == "__main__":
    unittest.main()
