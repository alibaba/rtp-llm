"""Check batch-invariant attention routing, decode graph replay and chunked prefill."""

import unittest
from copy import copy
from itertools import accumulate
from types import SimpleNamespace
from unittest.mock import patch, sentinel

import torch

_SM103 = (
    torch.version.cuda is not None
    and torch.cuda.is_available()
    and torch.cuda.get_device_capability() == (10, 3)
)
requires_sm103 = unittest.skipUnless(_SM103, "requires an SM103 CUDA GPU")


def qwen3_attention_config(max_seq_len):
    from rtp_llm.ops import AttentionConfigs, KvCacheDataType

    config = AttentionConfigs()
    config.head_num, config.kv_head_num, config.size_per_head = 32, 8, 128
    config.dtype = torch.bfloat16
    config.tokens_per_block = config.kernel_tokens_per_block = 64
    config.max_seq_len = max_seq_len
    config.is_causal = config.need_rope_kv_cache = True
    config.kv_cache_dtype = KvCacheDataType.BASE
    return config


def select_attention(config, inputs, enabled=True, **kwargs):
    from rtp_llm.models_py.modules.factory.attention import AttnImplFactory

    model = SimpleNamespace(
        getAttentionConfigs=lambda tp: config,
        quant_config=None,
        max_seq_len=config.max_seq_len,
    )
    return AttnImplFactory.get_fmha_impl(
        model,
        SimpleNamespace(get_attn_tp_size=lambda: 1),
        None,
        inputs,
        hw_kernel_config=SimpleNamespace(batch_invariant=enabled),
        **kwargs,
    )


def causal_gqa_reference(q, k, v):
    """Compute unpaged BF16-input attention in FP32 on the CPU."""
    prefix_len = k.size(0) - q.size(0)
    group_size = q.size(1) // k.size(1)
    queries = q.transpose(0, 1).to(device="cpu", dtype=torch.float32)
    keys, values = (
        x.transpose(0, 1)
        .to(device="cpu", dtype=torch.float32)
        .repeat_interleave(group_size, dim=0)
        for x in (k, v)
    )
    scores = (queries @ keys.transpose(-1, -2)) * q.size(-1) ** -0.5
    mask = torch.arange(k.size(0)) > prefix_len + torch.arange(q.size(0))[:, None]
    return (scores.masked_fill(mask, -torch.inf).softmax(-1) @ values).transpose(0, 1)


class BatchInvariantAttentionTest(unittest.TestCase):
    def test_factory_requires_explicit_opt_in_and_propagates_failures(self):
        from rtp_llm.config.cuda_graph import (
            GenerationPrefillCudaGraphUnsupportedBackend,
        )
        from rtp_llm.device.device_type import DeviceType
        from rtp_llm.models_py.modules.factory.attention import AttnImplFactory

        config = qwen3_attention_config(128)
        inputs = SimpleNamespace(is_prefill=False, is_target_verify=False)
        backend = "rtp_llm.models_py.determinism.batch_invariant"
        with (
            patch.dict(
                AttnImplFactory.FMHA_IMPL_REGISTRY, mha=lambda *args: sentinel.mha
            ),
            patch.dict("sys.modules", {backend: None}),
        ):
            for disabled in (False, "false"):
                self.assertIs(select_attention(config, inputs, disabled), sentinel.mha)
        with (
            patch(f"{backend}.get_device_type") as device_type,
            patch(
                f"{backend}.BatchInvariantAttention.__init__",
                side_effect=RuntimeError("failed to prepare KV"),
            ) as init,
        ):
            with patch.object(torch.cuda, "is_available", side_effect=AssertionError):
                for platform in (DeviceType.Cpu, DeviceType.ROCm):
                    device_type.return_value = platform
                    with (
                        self.subTest(platform=platform),
                        self.assertRaisesRegex(ValueError, "requires a CUDA backend"),
                    ):
                        select_attention(config, inputs, decode_cascade_max_requests=2)
            device_type.return_value = DeviceType.Cuda
            for graph, mode in (
                (True, "eager"),
                (False, "decode_graph"),
                (False, "generation_prefill_graph"),
                (False, "invalid"),
            ):
                with (
                    self.subTest(graph=graph, mode=mode),
                    self.assertRaisesRegex(ValueError, "selection mode"),
                ):
                    select_attention(
                        config,
                        inputs,
                        is_cuda_graph=graph,
                        cuda_graph_selection_mode=mode,
                    )
            inputs.is_prefill = True
            with self.assertRaises(GenerationPrefillCudaGraphUnsupportedBackend):
                select_attention(
                    config,
                    inputs,
                    is_cuda_graph=True,
                    cuda_graph_selection_mode="generation_prefill_graph",
                )
            with self.assertRaisesRegex(ValueError, "requires pure decode"):
                select_attention(config, inputs, is_cuda_graph=True)
            inputs.is_target_verify = True
            with self.assertRaisesRegex(ValueError, "requires causal BF16 GQA"):
                select_attention(config, inputs)

            inputs.is_prefill = False
            inputs.is_target_verify = False
            init.assert_not_called()
            for mode in (None, "decode_graph"):
                with self.assertRaisesRegex(RuntimeError, "failed to prepare KV"):
                    select_attention(
                        config,
                        inputs,
                        is_cuda_graph=True,
                        cuda_graph_selection_mode=mode,
                    )
                self.assertTrue(inputs.is_cuda_graph)
            with self.assertRaisesRegex(RuntimeError, "failed to prepare KV"):
                select_attention(config, inputs, decode_cascade_max_requests=2)
            self.assertFalse(inputs.is_cuda_graph)

    @requires_sm103
    @torch.inference_mode()
    def test_decode_graph_refreshes_kv_pages_and_sequence_lengths(self):
        from rtp_llm.models_py.determinism.batch_invariant import (
            BatchInvariantAttention,
            tiled_paged_attention,
        )
        from rtp_llm.ops import RopeStyle
        from rtp_llm.ops.compute_ops import (
            LayerKVCache,
            PyAttentionInputs,
            get_typemeta,
        )

        config = qwen3_attention_config(128)
        config.rope_config.style = RopeStyle.Base
        config.rope_config.dim = config.size_per_head
        device_int = {"device": "cuda", "dtype": torch.int32}
        generator = torch.Generator(device="cuda").manual_seed(604)

        def values(*shape):
            return torch.randn(
                shape, device="cuda", dtype=torch.bfloat16, generator=generator
            )

        qkv = values(
            2, (config.head_num + 2 * config.kv_head_num) * config.size_per_head
        )
        cache = LayerKVCache()
        cache.kv_cache_base = values(5, 2, config.kv_head_num, 64, config.size_per_head)
        lengths = torch.tensor([0, 63], dtype=torch.int32, pin_memory=True)
        inputs = PyAttentionInputs()
        inputs.is_prefill = False
        inputs.dtype = get_typemeta(qkv)
        inputs.sequence_lengths = lengths
        inputs.sequence_lengths_plus_1_device = lengths.cuda() + 1
        inputs.decode_cu_seqlens_device = torch.arange(3, **device_int)
        inputs.kv_cache_kernel_block_id_device = torch.tensor(
            [[1, 2], [3, 4]], **device_int
        )
        impl = select_attention(config, inputs, is_cuda_graph=True)
        self.assertIsInstance(impl, BatchInvariantAttention)
        self.assertIs(impl.paged_attention, tiled_paged_attention)
        self.assertTrue(impl.support_cuda_graph())
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            impl.forward(qkv, cache)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = impl.forward(qkv, cache)

        # Cross a page boundary, then send the unused second row to block zero
        # as the runner does when a two-row capture serves one real request.
        for positions, pages in (
            ([64, 65], [[4, 2], [1, 3]]),
            ([63, 0], [[3, 1], [0, 0]]),
        ):
            lengths.copy_(torch.tensor(positions, dtype=torch.int32))
            inputs.sequence_lengths_plus_1_device.copy_(lengths + 1)
            inputs.kv_cache_kernel_block_id_device.copy_(
                torch.tensor(pages, **device_int)
            )
            qkv.copy_(values(*qkv.shape))
            eager_cache = LayerKVCache()
            eager_cache.kv_cache_base = cache.kv_cache_base.clone()
            expected = select_attention(config, copy(inputs)).forward(
                qkv.clone(), eager_cache
            )
            impl.prepare_cuda_graph(inputs)
            graph.replay()
            torch.testing.assert_close(
                captured.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0
            )
            torch.testing.assert_close(
                cache.kv_cache_base.view(torch.int16),
                eager_cache.kv_cache_base.view(torch.int16),
                rtol=0,
                atol=0,
            )

    @requires_sm103
    @torch.inference_mode()
    def test_prefill_across_reordered_kv_pages_with_prefix(self):
        from rtp_llm.ops import RopeStyle
        from rtp_llm.ops.compute_ops import LayerKVCache, get_typemeta

        page_size, prefix_len, q_len = 64, 126, 7
        total_len = prefix_len + q_len
        config = qwen3_attention_config(total_len)
        config.rope_config.style = RopeStyle.No
        q_heads = config.head_num
        kv_heads = config.kv_head_num
        dim = config.size_per_head
        pages = [4, 1, 6]
        generator = torch.Generator(device="cuda").manual_seed(157)

        def values(*shape):
            return torch.randn(
                shape, device="cuda", dtype=torch.bfloat16, generator=generator
            )

        q = values(q_len, q_heads, dim)
        k = values(total_len, kv_heads, dim)
        v = values(total_len, kv_heads, dim)
        # Prefill begins in the second logical page and appends into the third.
        kv_cache = LayerKVCache()
        cache = values(8, 2, kv_heads, page_size, dim)
        prefix = torch.stack((k[:prefix_len], v[:prefix_len])).transpose(1, 2)
        for start, physical in zip(range(0, prefix_len, page_size), pages):
            block = prefix[:, :, start : start + page_size]
            cache[physical, :, :, : block.size(-2)].copy_(block)
        kv_cache.kv_cache_base = cache

        qkv = torch.cat(
            (q.flatten(1), k[prefix_len:].flatten(1), v[prefix_len:].flatten(1)), dim=1
        )
        lengths = torch.tensor([q_len], dtype=torch.int32)
        prefix_lengths = torch.tensor([prefix_len], dtype=torch.int32)
        device_int = {"device": "cuda", "dtype": torch.int32}
        # The native input object exposes its device length mirrors read-only.
        inputs = SimpleNamespace(
            is_prefill=True,
            is_target_verify=False,
            is_cuda_graph=False,
            dtype=get_typemeta(qkv),
            input_lengths=lengths,
            input_lengths_device=lengths.cuda(),
            prefix_lengths=prefix_lengths,
            prefix_lengths_device=prefix_lengths.cuda(),
            sequence_lengths=torch.empty(0, dtype=torch.int32),
            cu_seqlens_device=torch.tensor([0, q_len], **device_int),
            cu_kv_seqlens_device=torch.tensor([0, total_len], **device_int),
            padding_offset=torch.zeros(q_len, **device_int),
            total_tokens=q_len,
            context_total_kv_length=total_len,
            kv_cache_kernel_block_id_device=torch.tensor([pages], **device_int),
            combo_position_ids=None,
            context_parallel_info=None,
        )

        impl = select_attention(config, inputs)
        self.assertFalse(impl.support_cuda_graph())
        actual = impl.forward(qkv, kv_cache)
        torch.testing.assert_close(
            actual.float().cpu(), causal_gqa_reference(q, k, v), rtol=0.008, atol=0.006
        )

    @requires_sm103
    @torch.inference_mode()
    def test_attention_is_bitwise_invariant_across_batches_chunks_and_decode(self):
        from rtp_llm.models_py.determinism.batch_invariant import tiled_paged_attention

        dim, page_size = 128, 64
        lengths = (193, 321)
        # Logical neighbours live in nonadjacent pages. Page zero is padding.
        pages = ([5, 2, 8, 1], [9, 3, 10, 7, 4, 6])
        device_int = {"device": "cuda", "dtype": torch.int32}
        generator = torch.Generator(device="cuda").manual_seed(824)

        def values(*shape):
            return torch.randn(
                shape, device="cuda", dtype=torch.bfloat16, generator=generator
            )

        # Local Q and KV heads on each TP1, TP2 and TP4 rank of both models.
        for q_heads, kv_heads in ((32, 8), (32, 4), (16, 4), (16, 2), (8, 2), (8, 1)):
            with self.subTest(q_heads=q_heads, kv_heads=kv_heads):
                queries = [values(length, q_heads, dim) for length in lengths]
                keys = [values(length, kv_heads, dim) for length in lengths]
                vals = [values(length, kv_heads, dim) for length in lengths]
                cache = values(11, 2, kv_heads, page_size, dim)
                for row, physical_pages in enumerate(pages):
                    kv = torch.stack((keys[row], vals[row])).transpose(1, 2)
                    for logical, physical in enumerate(physical_pages):
                        start = logical * page_size
                        data = kv[:, :, start : start + page_size]
                        cache[physical, :, :, : data.size(2)].copy_(data)

                def run(*spans, decode=False):
                    # Each span supplies new queries after its cached prefix.
                    q_lengths = [end - start for _, start, end in spans]
                    kv_lengths = [end for _, _, end in spans]
                    max_pages = (max(kv_lengths) + page_size - 1) // page_size
                    table = [
                        (pages[row] + [0] * max_pages)[:max_pages]
                        for row, _, _ in spans
                    ]
                    q = torch.cat(
                        [queries[row][start:end] for row, start, end in spans]
                    )
                    if decode:
                        self.assertTrue(all(n == 1 for n in q_lengths))
                    output = tiled_paged_attention(
                        q,
                        cache,
                        torch.tensor(table, **device_int),
                        torch.tensor([0, *accumulate(q_lengths)], **device_int),
                        torch.tensor(kv_lengths, **device_int),
                        max(q_lengths),
                        decode=decode,
                    )
                    self.assertEqual(output.shape, q.shape)
                    self.assertEqual(output.dtype, q.dtype)
                    self.assertEqual(output.device, q.device)
                    return output.split(q_lengths)

                target = (0, 0, lengths[0])
                neighbour = (1, 0, lengths[1])
                (solo,) = run(target)
                (neighbour_solo,) = run(neighbour)
                # The same continuation token may be scheduled as a decode
                # row or as the tail of a longer prefill chunk.
                for pos in (0, 62, 63, 64, lengths[0] - 1):
                    a = (0, pos, pos + 1)
                    b = (1, lengths[1] - 1, lengths[1])
                    (single,) = run(a, decode=True)
                    first, _ = run(a, b, decode=True)
                    other, last = run(b, a, decode=True)
                    for actual in (single, first, last):
                        torch.testing.assert_close(
                            actual.view(torch.int16),
                            solo[pos : pos + 1].view(torch.int16),
                            rtol=0,
                            atol=0,
                        )
                    torch.testing.assert_close(
                        other.view(torch.int16),
                        neighbour_solo[-1:].view(torch.int16),
                        rtol=0,
                        atol=0,
                    )
                target_first, _ = run(target, neighbour)
                _, target_last = run(neighbour, target)
                (single_token,) = run((0, 0, 1))
                single_with_neighbour, _ = run((0, 0, 1), neighbour)
                for output in (single_token, single_with_neighbour):
                    torch.testing.assert_close(
                        output.view(torch.int16),
                        solo[:1].view(torch.int16),
                        rtol=0,
                        atol=0,
                    )
                # Exercise a partial first page, an unaligned prefix and a
                # one-query tail; the visible KV grows at each chunk boundary.
                boundaries = (0, 63, 128, 192, lengths[0])
                chunked = torch.cat(
                    [
                        run((0, start, end))[0]
                        for start, end in zip(boundaries, boundaries[1:])
                    ]
                )
                for name, comparison in (
                    ("target first", target_first),
                    ("target last", target_last),
                    ("chunked", chunked),
                ):
                    with self.subTest(comparison=name):
                        torch.testing.assert_close(
                            comparison.view(torch.int16),
                            solo.view(torch.int16),
                            rtol=0,
                            atol=0,
                        )
                torch.testing.assert_close(
                    solo.float().cpu(),
                    causal_gqa_reference(queries[0], keys[0], vals[0]),
                    rtol=0.008,
                    atol=0.006,
                )


if __name__ == "__main__":
    unittest.main()
