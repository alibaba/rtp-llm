import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch, sentinel

import torch

from rtp_llm.models_py.modules.factory.attention.attn_factory import AttnImplFactory
from rtp_llm.models_py.modules.factory.attention.rocm_impl.beam_search_cascade import (
    AiterCascadeDecodeImpl,
)
from rtp_llm.models_py.triton_kernels.attention.beam_search_cascade import (
    RocmBeamSearchCascadeAttention,
)
from rtp_llm.ops import AttentionConfigs, FMHAConfig, KvCacheDataType
from rtp_llm.ops.compute_ops import LayerKVCache, PyAttentionInputs


def make_inputs():
    config = AttentionConfigs()
    config.head_num, config.kv_head_num, config.size_per_head = 16, 8, 128
    config.kernel_tokens_per_block = 16
    config.dtype = torch.bfloat16
    config.kv_cache_dtype = KvCacheDataType.BASE
    config.is_causal = True
    config.need_rope_kv_cache = False
    inputs = PyAttentionInputs()
    inputs.is_prefill = False
    inputs.is_target_verify = False
    inputs.is_cuda_graph = False
    inputs.input_lengths = torch.full((128,), 500, dtype=torch.int32)
    inputs.sequence_lengths = torch.full((128,), 500, dtype=torch.int32)
    inputs.kv_cache_kernel_block_id = torch.zeros((128, 32), dtype=torch.int32)
    return config, inputs


class RocmCascadeIntegrationTest(unittest.TestCase):
    def test_factory_selects_rocm_independently_of_flashinfer_disable_flag(self):
        configs, inputs = make_inputs()
        model = SimpleNamespace(
            getAttentionConfigs=lambda tp: configs,
            max_seq_len=512,
            quant_config=None,
        )
        parallel = SimpleNamespace(
            get_attn_tp_size=lambda: 1,
            prefill_cp_config=SimpleNamespace(is_enabled=lambda: False),
        )
        fmha = FMHAConfig()
        fmha.disable_flashinfer_native = True
        with patch.object(
            AiterCascadeDecodeImpl, "__init__", return_value=None
        ) as init:
            result = AttnImplFactory.get_fmha_impl(
                model,
                parallel,
                Mock(),
                inputs,
                fmha,
                is_cuda_graph=True,
                decode_cascade_max_requests=24,
            )
        self.assertIsInstance(result, AiterCascadeDecodeImpl)
        self.assertTrue(inputs.is_cuda_graph)
        self.assertEqual(init.call_args.args[3:5], (24, 512))

    def test_unsupported_modes_fall_back_to_registered_attention(self):
        for mode in ("disabled", "prefill", "verify", "quantized", "cp"):
            with self.subTest(mode=mode):
                configs, inputs = make_inputs()
                inputs.is_prefill = mode == "prefill"
                inputs.is_target_verify = mode == "verify"
                if mode == "quantized":
                    configs.kv_cache_dtype = KvCacheDataType.FP8
                model = SimpleNamespace(
                    getAttentionConfigs=lambda tp: configs,
                    max_seq_len=512,
                    quant_config=None,
                )
                parallel = SimpleNamespace(
                    get_attn_tp_size=lambda: 1,
                    prefill_cp_config=SimpleNamespace(is_enabled=lambda: mode == "cp"),
                )
                fallback = Mock(return_value=sentinel.fallback)
                with patch.dict(AttnImplFactory.FMHA_IMPL_REGISTRY, {"mha": fallback}):
                    result = AttnImplFactory.get_fmha_impl(
                        model,
                        parallel,
                        Mock(),
                        inputs,
                        decode_cascade_max_requests=0 if mode == "disabled" else 24,
                    )
                self.assertIs(result, sentinel.fallback)
                fallback.assert_called_once()

    def test_adapter_forward_and_replay_update_use_existing_layout(self):
        configs, inputs = make_inputs()
        torch.manual_seed(20261021)
        q = torch.randn(128, 16, 128, dtype=torch.bfloat16, device="cuda")
        kv = torch.randn(160, 2, 8, 16, 128, dtype=q.dtype, device=q.device)
        table = torch.arange(32, dtype=torch.int32).expand(128, -1).clone()
        table[:, -1] = torch.arange(128) + 32
        inputs.kv_cache_kernel_block_id = table
        impl = AiterCascadeDecodeImpl.__new__(AiterCascadeDecodeImpl)
        impl.configs, impl.attn_inputs = configs, inputs
        impl.need_rope_kv_cache = False
        impl.write_cache_store_impl = None
        impl.rope_params = Mock()
        impl.cascade = RocmBeamSearchCascadeAttention(
            128, 16, 8, 128, 16, 32, q.dtype, q.device
        )
        cache = LayerKVCache()
        cache.kv_cache_base = kv
        impl.prepare_cuda_graph(inputs)
        out = impl.forward(q, cache).clone()
        expected = impl.cascade.run(q, kv).reshape(128, -1).clone()
        torch.testing.assert_close(out, expected, atol=0, rtol=0)
        inputs.kv_cache_kernel_block_id_device = table.cuda()
        inputs.sequence_lengths = inputs.sequence_lengths - 128
        impl.prepare_cuda_graph(inputs)
        impl.rope_params.update_kv_cache_offset.assert_called_once_with(
            inputs.kv_cache_kernel_block_id_device
        )
        self.assertEqual(int(impl.cascade.lengths_host[0]), 373)


if __name__ == "__main__":
    unittest.main()
