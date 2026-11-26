"""Check batch-invariant attention routing, decode graph replay and chunked prefill."""

import unittest
from copy import copy
from types import SimpleNamespace
from unittest.mock import Mock, patch, sentinel

import torch


def qwen3_attention_config(max_seq_len):
    from rtp_llm.ops import AttentionConfigs, KvCacheDataType

    config = AttentionConfigs()
    config.head_num, config.kv_head_num, config.size_per_head = 32, 8, 128
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


class BatchInvariantAttentionTest(unittest.TestCase):
    def test_sparse_gqa_uses_registered_backend_with_batch_invariant_setting(self):
        from rtp_llm.models_py.modules.factory.attention import AttnImplFactory

        config = SimpleNamespace(
            use_mla=False, is_sparse=True, use_sparse_gqa_fmha=True, max_seq_len=8192
        )
        inputs = SimpleNamespace(is_prefill=False)
        sparse = Mock()
        with patch.dict(AttnImplFactory.FMHA_IMPL_REGISTRY, sparse_gqa=sparse):
            select_attention(config, inputs, enabled=False)
            self.assertEqual(sparse.call_args.kwargs, {})
            select_attention(
                config, inputs, enabled=True, decode_cascade_max_requests=2
            )
            self.assertEqual(sparse.call_args.kwargs, {"batch_invariant": True})

    def test_factory_requires_explicit_opt_in_and_propagates_failures(self):
        from rtp_llm.config.cuda_graph import (
            GenerationPrefillCudaGraphUnsupportedBackend,
        )
        from rtp_llm.device.device_type import DeviceType
        from rtp_llm.models_py.modules.factory.attention import AttnImplFactory

        config = qwen3_attention_config(128)
        inputs = SimpleNamespace(is_prefill=False, is_target_verify=False)
        backend = (
            "rtp_llm.models_py.modules.factory.attention.cuda_impl.batch_invariant"
        )
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

    @unittest.skipUnless(
        torch.version.cuda is not None
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability() == (10, 3),
        "requires an SM103 CUDA GPU",
    )
    @torch.inference_mode()
    def test_decode_graph_refreshes_kv_pages_and_sequence_lengths(self):
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

    @unittest.skipUnless(
        torch.version.cuda is not None
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability() == (10, 3),
        "requires an SM103 CUDA GPU",
    )
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

        actual = select_attention(config, inputs).forward(qkv, kv_cache)

        # The reference uses logical K/V tensors and an explicit causal mask.
        # It never reads the paged cache or the wrapper's computed lengths.
        keys, values_per_head = (
            x.transpose(0, 1)
            .to(device="cpu", dtype=torch.float32)
            .repeat_interleave(q_heads // kv_heads, dim=0)
            for x in (k, v)
        )
        scores = q.transpose(0, 1).float().cpu() @ keys.transpose(-1, -2)
        mask = torch.arange(total_len) > prefix_len + torch.arange(q_len)[:, None]
        probs = (scores * dim**-0.5).masked_fill(mask, -torch.inf).softmax(-1)
        expected = (probs @ values_per_head).transpose(0, 1)
        torch.testing.assert_close(
            actual.float().cpu(), expected, rtol=0.008, atol=0.006
        )


if __name__ == "__main__":
    unittest.main()
