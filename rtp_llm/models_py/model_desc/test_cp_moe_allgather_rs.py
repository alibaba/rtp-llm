"""Unit tests for CP MoE allgather+reduce_scatter communication pattern.

Tests the skip_tp_allreduce flag in DenseMLP/PureTpRouter,
is_cp_prefill property in FMHA implementations, and the
allgather+reduce_scatter wrapping in GenericMoeDecoderLayer.
"""

import unittest
from unittest.mock import MagicMock, patch

import torch

from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import (
    FMHAImplBase,
    MlaImplBase,
)


class TestIsCpPrefillProperty(unittest.TestCase):
    """Test is_cp_prefill property on FMHA base classes."""

    def test_fmha_impl_base_default_false(self):
        """FMHAImplBase.is_cp_prefill defaults to False."""

        class ConcreteFMHA(FMHAImplBase):
            def forward(self, qkv, kv_cache):
                pass

            @staticmethod
            def support(attn_configs, attn_inputs):
                return True

        impl = ConcreteFMHA.__new__(ConcreteFMHA)
        self.assertFalse(impl.is_cp_prefill)

    def test_mla_impl_base_default_false(self):
        """MlaImplBase.is_cp_prefill defaults to False."""

        class ConcreteMLA(MlaImplBase):
            def forward(self, q, compressed_kv, k_pe, kv_cache, layer_id, topk_indices=None):
                pass

            @staticmethod
            def support(attn_configs, attn_inputs):
                return True

        impl = ConcreteMLA.__new__(ConcreteMLA)
        self.assertFalse(impl.is_cp_prefill)

    def test_cp_override_returns_true(self):
        """Subclass overriding is_cp_prefill returns True."""

        class CPFmhaImpl(FMHAImplBase):
            def __init__(self):
                self._is_cp_prefill = True

            @property
            def is_cp_prefill(self):
                return self._is_cp_prefill

            def forward(self, qkv, kv_cache):
                pass

            @staticmethod
            def support(attn_configs, attn_inputs):
                return True

        impl = CPFmhaImpl()
        self.assertTrue(impl.is_cp_prefill)

    def test_getattr_fallback(self):
        """getattr(fmha_impl, 'is_cp_prefill', False) works for plain objects."""
        plain_obj = object()
        self.assertFalse(getattr(plain_obj, "is_cp_prefill", False))


class TestSkipTpAllreduce(unittest.TestCase):
    """Test skip_tp_allreduce behavior in DenseMLP and PureTpRouter."""

    def test_dense_mlp_skip_allreduce(self):
        """DenseMLP with skip_tp_allreduce=True should not call all_reduce."""
        from rtp_llm.models_py.modules.hybrid.dense_mlp import DenseMLP

        mlp = DenseMLP.__new__(DenseMLP)
        mlp.skip_tp_allreduce = True
        mlp.parallelism_config = MagicMock()
        mlp.parallelism_config.get_ffn_tp_size.return_value = 4
        mlp.is_gated = True
        mlp.up_proj = MagicMock(return_value=torch.randn(2, 8))
        mlp.act_fn = MagicMock(return_value=torch.randn(2, 4))
        mlp.down_proj = MagicMock(return_value=torch.randn(2, 4))

        with patch(
            "rtp_llm.models_py.modules.hybrid.dense_mlp.all_reduce"
        ) as mock_ar:
            result = mlp.forward(torch.randn(2, 4))
            mock_ar.assert_not_called()

    def test_dense_mlp_normal_allreduce(self):
        """DenseMLP with skip_tp_allreduce=False should call all_reduce when tp>1."""
        from rtp_llm.models_py.modules.hybrid.dense_mlp import DenseMLP

        mlp = DenseMLP.__new__(DenseMLP)
        mlp.skip_tp_allreduce = False
        mlp.parallelism_config = MagicMock()
        mlp.parallelism_config.get_ffn_tp_size.return_value = 4
        mlp.is_gated = True
        mlp.up_proj = MagicMock(return_value=torch.randn(2, 8))
        mlp.act_fn = MagicMock(return_value=torch.randn(2, 4))
        mlp.down_proj = MagicMock(return_value=torch.randn(2, 4))

        with patch(
            "rtp_llm.models_py.modules.hybrid.dense_mlp.all_reduce"
        ) as mock_ar:
            mock_ar.return_value = torch.randn(2, 4)
            result = mlp.forward(torch.randn(2, 4))
            mock_ar.assert_called_once()

    def test_pure_tp_router_skip_allreduce(self):
        """PureTpRouterBase.finalize respects skip_tp_allreduce in extra_finalize_args."""
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_tp_router import (
            PureTpRouterBase,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
            CombineForwardPayload,
        )

        router = PureTpRouterBase.__new__(PureTpRouterBase)
        router.tp_size = 4

        payload = CombineForwardPayload(fused_expert_output=torch.randn(4, 8))

        with patch(
            "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_tp_router.all_reduce"
        ) as mock_ar:
            # With skip_tp_allreduce=True, all_reduce should NOT be called
            result = router.finalize(
                payload,
                topk_weights=torch.randn(4, 2),
                topk_ids=torch.randint(0, 8, (4, 2)),
                apply_router_weight_on_input=False,
                extra_finalize_args={"skip_tp_allreduce": True},
            )
            mock_ar.assert_not_called()

        with patch(
            "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_tp_router.all_reduce"
        ) as mock_ar:
            mock_ar.return_value = torch.randn(4, 8)
            # Without skip_tp_allreduce, all_reduce SHOULD be called
            result = router.finalize(
                payload,
                topk_weights=torch.randn(4, 2),
                topk_ids=torch.randint(0, 8, (4, 2)),
                apply_router_weight_on_input=False,
                extra_finalize_args=None,
            )
            mock_ar.assert_called_once()


class TestDecoderLayerCpMoeWrapping(unittest.TestCase):
    """Test allgather+reduce_scatter wrapping in GenericMoeDecoderLayer."""

    def _make_decoder_layer(self, use_cp_moe_allgather_rs=True):
        """Create a minimal GenericMoeDecoderLayer with mocked internals."""
        from rtp_llm.models_py.model_desc.generic_moe import GenericMoeDecoderLayer

        layer = GenericMoeDecoderLayer.__new__(GenericMoeDecoderLayer)
        layer.layer_idx = 1
        layer.use_cp_moe_allgather_rs = use_cp_moe_allgather_rs

        layer.input_layernorm = MagicMock(side_effect=lambda h, r: h)
        layer.self_attn = MagicMock(side_effect=lambda hidden_states, fmha_impl, kv_cache: hidden_states)
        layer.post_attention_layernorm = MagicMock(side_effect=lambda h, r: h)
        layer.mlp = MagicMock(side_effect=lambda x: x * 2)
        return layer

    def test_cp_active_calls_allgather_and_reduce_scatter(self):
        """When CP is active, allgather and reduce_scatter are called."""
        layer = self._make_decoder_layer(use_cp_moe_allgather_rs=True)

        fmha_impl = MagicMock()
        fmha_impl.is_cp_prefill = True

        hidden = torch.randn(4, 8)
        residual = torch.randn(4, 8)

        with patch(
            "rtp_llm.models_py.model_desc.generic_moe.all_gather"
        ) as mock_ag, patch(
            "rtp_llm.models_py.model_desc.generic_moe.reduce_scatter"
        ) as mock_rs:
            mock_ag.side_effect = lambda t, group: torch.cat([t, t], dim=0)
            mock_rs.side_effect = lambda t, group: t[:t.shape[0] // 2]

            output = layer.forward(hidden, residual, fmha_impl)

            # all_gather called twice (hidden + residual)
            self.assertEqual(mock_ag.call_count, 2)
            # reduce_scatter called twice (hidden + residual)
            self.assertEqual(mock_rs.call_count, 2)

    def test_cp_inactive_no_allgather_reduce_scatter(self):
        """When CP is not active, no allgather/reduce_scatter calls."""
        layer = self._make_decoder_layer(use_cp_moe_allgather_rs=True)

        fmha_impl = MagicMock()
        fmha_impl.is_cp_prefill = False

        hidden = torch.randn(4, 8)
        residual = torch.randn(4, 8)

        with patch(
            "rtp_llm.models_py.model_desc.generic_moe.all_gather"
        ) as mock_ag, patch(
            "rtp_llm.models_py.model_desc.generic_moe.reduce_scatter"
        ) as mock_rs:
            output = layer.forward(hidden, residual, fmha_impl)
            mock_ag.assert_not_called()
            mock_rs.assert_not_called()

    def test_flag_disabled_no_allgather_reduce_scatter(self):
        """When use_cp_moe_allgather_rs is False, no wrapping even with CP fmha."""
        layer = self._make_decoder_layer(use_cp_moe_allgather_rs=False)

        fmha_impl = MagicMock()
        fmha_impl.is_cp_prefill = True

        hidden = torch.randn(4, 8)
        residual = torch.randn(4, 8)

        with patch(
            "rtp_llm.models_py.model_desc.generic_moe.all_gather"
        ) as mock_ag, patch(
            "rtp_llm.models_py.model_desc.generic_moe.reduce_scatter"
        ) as mock_rs:
            output = layer.forward(hidden, residual, fmha_impl)
            mock_ag.assert_not_called()
            mock_rs.assert_not_called()

    def test_mlp_receives_full_tokens_during_cp(self):
        """MLP should receive gathered (full) tokens when CP is active."""
        layer = self._make_decoder_layer(use_cp_moe_allgather_rs=True)

        fmha_impl = MagicMock()
        fmha_impl.is_cp_prefill = True

        scattered_size = 4
        hidden = torch.randn(scattered_size, 8)
        residual = torch.randn(scattered_size, 8)

        tp_size = 2

        with patch(
            "rtp_llm.models_py.model_desc.generic_moe.all_gather"
        ) as mock_ag, patch(
            "rtp_llm.models_py.model_desc.generic_moe.reduce_scatter"
        ) as mock_rs:
            mock_ag.side_effect = lambda t, group: torch.cat([t] * tp_size, dim=0)
            mock_rs.side_effect = lambda t, group: t[:t.shape[0] // tp_size]

            output = layer.forward(hidden, residual, fmha_impl)

            # Verify MLP was called with gathered (full) size
            mlp_input = layer.mlp.call_args[0][0]
            self.assertEqual(mlp_input.shape[0], scattered_size * tp_size)

    def test_fmha_without_is_cp_prefill_attr(self):
        """fmha_impl without is_cp_prefill should not trigger wrapping."""
        layer = self._make_decoder_layer(use_cp_moe_allgather_rs=True)

        # Plain object without is_cp_prefill
        fmha_impl = object()

        hidden = torch.randn(4, 8)
        residual = torch.randn(4, 8)

        # Must add the forward-compatible attributes
        layer.self_attn = MagicMock(
            side_effect=lambda hidden_states, fmha_impl, kv_cache: hidden_states
        )

        with patch(
            "rtp_llm.models_py.model_desc.generic_moe.all_gather"
        ) as mock_ag, patch(
            "rtp_llm.models_py.model_desc.generic_moe.reduce_scatter"
        ) as mock_rs:
            output = layer.forward(hidden, residual, fmha_impl)
            mock_ag.assert_not_called()
            mock_rs.assert_not_called()


if __name__ == "__main__":
    unittest.main()
