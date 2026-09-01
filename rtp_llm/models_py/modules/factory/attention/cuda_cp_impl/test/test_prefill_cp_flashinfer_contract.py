import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_cp_impl import (
    prefill_cp_flashinfer,
)
from rtp_llm.models_py.modules.factory.attention.cuda_cp_impl.prefill_cp_flashinfer import (
    CPFlashInferImpl,
)


class TestCPFlashInferForwardContract(unittest.TestCase):
    @staticmethod
    def _make_impl(need_rope_kv_cache):
        method = object()
        attn_configs = SimpleNamespace(need_rope_kv_cache=need_rope_kv_cache)
        attn_inputs = SimpleNamespace(is_prefill=True, cache_store_inputs=None)
        parallelism_config = SimpleNamespace(
            prefill_cp_config=SimpleNamespace(method=method)
        )
        fmha_impl = Mock()
        rope_impl = Mock()

        with patch.dict(
            prefill_cp_flashinfer.impl_map,
            {method: Mock(return_value=fmha_impl)},
            clear=True,
        ), patch.object(
            prefill_cp_flashinfer,
            "FusedRopeKVCachePrefillOpQKVOut",
            return_value=rope_impl,
        ):
            impl = CPFlashInferImpl(attn_configs, attn_inputs, parallelism_config)

        return impl, fmha_impl, rope_impl

    def test_layer_zero_does_not_disable_rope(self):
        impl, fmha_impl, rope_impl = self._make_impl(True)

        qkv = torch.randn(2, 8)
        rope_output = torch.randn(2, 8)
        attention_output = torch.randn(2, 4)
        rope_impl.forward.return_value = rope_output
        fmha_impl.forward.return_value = attention_output

        result = impl.forward(qkv, kv_cache=None, layer_idx=0)

        rope_impl.forward.assert_called_once_with(qkv, None, impl.rope_params)
        fmha_impl.forward.assert_called_once_with(rope_output, None, impl.fmha_params)
        self.assertIs(result, attention_output)

    def test_forward_skips_rope_when_disabled(self):
        impl, fmha_impl, rope_impl = self._make_impl(False)

        qkv = torch.randn(2, 8)
        attention_output = torch.randn(2, 4)
        fmha_impl.forward.return_value = attention_output

        result = impl.forward(qkv, kv_cache=None, layer_idx=0)

        rope_impl.forward.assert_not_called()
        fmha_impl.forward.assert_called_once_with(qkv, None, impl.fmha_params)
        self.assertIs(result, attention_output)


if __name__ == "__main__":
    unittest.main()
