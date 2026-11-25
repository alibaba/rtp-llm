"""Check batch-invariant attention routing and compare chunked prefill with dense GQA."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch, sentinel

import torch


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
    def test_factory_requires_explicit_opt_in_and_propagates_failures(self):
        from rtp_llm.device.device_type import DeviceType
        from rtp_llm.models_py.modules.factory.attention import AttnImplFactory
        from rtp_llm.ops import AttentionConfigs, KvCacheDataType

        config = AttentionConfigs()
        config.is_causal = config.need_rope_kv_cache = True
        config.kv_cache_dtype = KvCacheDataType.BASE
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
                (True, None),
                (True, "eager"),
                (False, "decode_graph"),
                (False, "generation_prefill_graph"),
                (False, "invalid"),
            ):
                with (
                    self.subTest(graph=graph, mode=mode),
                    self.assertRaisesRegex(ValueError, "requires eager"),
                ):
                    select_attention(
                        config,
                        inputs,
                        is_cuda_graph=graph,
                        cuda_graph_selection_mode=mode,
                    )
            inputs.is_target_verify = True
            with self.assertRaisesRegex(ValueError, "requires eager"):
                select_attention(config, inputs)

            inputs.is_target_verify = False
            inputs.is_cuda_graph = True
            init.assert_not_called()
            with self.assertRaisesRegex(RuntimeError, "failed to prepare KV"):
                select_attention(config, inputs, decode_cascade_max_requests=2)
            init.assert_called_once()
            self.assertFalse(inputs.is_cuda_graph)

    @unittest.skipUnless(
        torch.version.cuda is not None
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability() == (10, 3),
        "requires an SM103 CUDA GPU",
    )
    @torch.inference_mode()
    def test_prefill_across_reordered_kv_pages_with_prefix(self):
        from rtp_llm.ops import AttentionConfigs, KvCacheDataType, RopeStyle
        from rtp_llm.ops.compute_ops import LayerKVCache, get_typemeta

        q_heads, kv_heads, dim = 32, 8, 128
        page_size, prefix_len, q_len = 64, 126, 7
        total_len = prefix_len + q_len
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

        config = AttentionConfigs()
        config.head_num = q_heads
        config.kv_head_num = kv_heads
        config.size_per_head = dim
        config.tokens_per_block = config.kernel_tokens_per_block = page_size
        config.max_seq_len = total_len
        config.is_causal = config.need_rope_kv_cache = True
        config.kv_cache_dtype = KvCacheDataType.BASE
        config.rope_config.style = RopeStyle.No

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
