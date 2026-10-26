import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.modules.hybrid.causal_attention import CausalAttention
from rtp_llm.utils.model_weight import W


class CausalAttentionCenterTest(unittest.TestCase):
    def make_attention(self, mode, has_bias, has_norm, tp_size=1):
        qh, kh, dim = 4, 2, 64
        config = SimpleNamespace(
            head_num=qh, kv_head_num=kh, size_per_head=dim, fp8_kv_cache_mode=mode
        )
        parallelism = Mock()
        parallelism.get_attn_tp_size.return_value = tp_size
        weights = {}
        if has_bias:
            weights[W.attn_qkv_b] = torch.arange((qh + 2 * kh) * dim).bfloat16()
        elif has_bias is None:
            weights[W.attn_qkv_b] = None
        if has_norm:
            weights[W.q_ln_gamma] = torch.ones(dim)
            weights[W.k_ln_gamma] = torch.ones(dim)
        projection = Mock()
        with patch(
            "rtp_llm.models_py.modules.hybrid.causal_attention.LinearFactory.create_linear_from_weights",
            return_value=projection,
        ), patch(
            "rtp_llm.models_py.modules.hybrid.causal_attention.FusedQKRMSNorm",
            return_value=torch.nn.Identity(),
        ):
            return CausalAttention(config, parallelism, weights, 1e-6), weights

    def test_only_dynamic_biased_layers_without_qk_norm_are_centered(self):
        for mode in (0, 1, 2):
            for bias in (None, False, True):
                for norm in (False, True):
                    with self.subTest(mode=mode, bias=bias, norm=norm):
                        attention, weights = self.make_attention(mode, bias, norm)
                        if mode == 2 and bias and not norm:
                            expected = (
                                weights[W.attn_qkv_b][256:384].float().view(2, 64)
                            )
                            torch.testing.assert_close(
                                attention.fp8_key_center, expected
                            )
                            self.assertTrue(attention.fp8_key_center.is_contiguous())
                        else:
                            self.assertIsNone(attention.fp8_key_center)

    def test_center_uses_local_heads_and_is_reused_across_requests(self):
        attention, _ = self.make_attention(2, True, False, tp_size=2)
        center = attention.fp8_key_center
        pointer = center.data_ptr()
        fmha = Mock()
        fmha.forward.return_value = torch.zeros(1, 4, 64)
        attention.o_proj.return_value = torch.zeros(1, 256)
        # Avoid a distributed collective in this local weight-contract test.
        for _ in range(2):
            attention(
                torch.zeros(1, 256),
                fmha,
                None,
                qkv=torch.zeros(1, 512),
                skip_allreduce=True,
            )
            self.assertEqual(
                fmha.forward.call_args.kwargs["key_center"].data_ptr(), pointer
            )
            self.assertIs(attention.fp8_key_center, center)


if __name__ == "__main__":
    unittest.main()
