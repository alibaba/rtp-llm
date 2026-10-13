"""Decoder construction and forwarding into the real generic routed-only layer."""

from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch

import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models.qwen3_next.qwen3_next import Qwen3Next, Qwen35Moe
from rtp_llm.models_py.model_desc import generic_moe, qwen3_next
from rtp_llm.ops import HybridAttentionType, MoeConfig, ParallelismConfig
from rtp_llm.utils.model_weight import W


class _Attention(nn.Module):
    def forward(self, hidden_states, **kwargs):
        return hidden_states


class _Norm(nn.Module):
    def forward(self, hidden_states, residual):
        return hidden_states, residual


class _RoutedBackend(nn.Module):
    includes_shared_expert = False
    topk_ids_dtype = torch.int32
    router = SimpleNamespace(tp_collective_size=1, supports_skip_tp_allreduce=False)

    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, hidden_states, topk_ids, topk_weights, **kwargs):
        self.calls += 1
        return hidden_states + topk_weights.sum(dim=-1, keepdim=True)


def _select_topk(logits, ids, weights):
    values, selected = logits.softmax(dim=-1).topk(ids.shape[-1])
    ids.copy_(selected)
    weights.copy_(values / values.sum(dim=-1, keepdim=True))


class Qwen3NextRoutedOnlyTest(TestCase):
    def test_routed_only_decoder_constructs_and_forwards_generic_moe(self):
        for model_cls in (Qwen3Next, Qwen35Moe):
            for shared_width in (None, 0):
                with self.subTest(model=model_cls.__name__, shared_width=shared_width):
                    config = ModelConfig()
                    config.num_layers = 1
                    config.hidden_size = 8
                    config.max_seq_len = 16
                    config.activation_type = "SiGLU"
                    config.hybrid_attention_config.hybrid_attention_types = [
                        HybridAttentionType.LINEAR
                    ]
                    hf = {
                        "num_experts": 4,
                        "num_experts_per_tok": 2,
                        "moe_intermediate_size": 4,
                    }
                    if shared_width is not None:
                        hf["shared_expert_intermediate_size"] = shared_width
                    model_cls._parse_moe_config(hf, config)
                    weights = {
                        W.pre_ln_gamma: torch.ones(8),
                        W.post_ln_gamma: torch.ones(8),
                        W.moe_gate: torch.ones(4, 8),
                        W.moe_w1: torch.ones(4, 8, 8),
                        W.moe_w2: torch.ones(4, 8, 4),
                    }
                    backend = _RoutedBackend()
                    with (
                        patch.object(
                            qwen3_next,
                            "Qwen3NextGatedDeltaNet",
                            return_value=_Attention(),
                        ),
                        patch.object(
                            qwen3_next,
                            "RMSResNorm",
                            side_effect=lambda *a, **kw: _Norm(),
                        ),
                        patch.object(
                            generic_moe.LinearFactory,
                            "create_linear_from_weights",
                            return_value=nn.Linear(8, 4, bias=False),
                        ),
                        patch.object(
                            generic_moe, "SelectTopk", return_value=_select_topk
                        ),
                        patch.object(
                            generic_moe.FusedMoeFactory,
                            "create_fused_moe",
                            return_value=backend,
                        ),
                        patch.object(generic_moe, "DenseMLP") as shared_mlp,
                    ):
                        layer = qwen3_next.Qwen3NextDecoderLayer(
                            config, ParallelismConfig(), weights, 0, MoeConfig()
                        )
                        self.assertIsInstance(layer.mlp, generic_moe.GenericMoeLayer)
                        self.assertIsNone(layer.mlp.shared_expert)
                        hidden = torch.randn(3, 8)
                        residual = torch.randn_like(hidden)
                        output, result_residual = layer(hidden, residual, None)
                    shared_mlp.assert_not_called()
                    self.assertEqual(backend.calls, 1)
                    torch.testing.assert_close(output, hidden + 1)
                    self.assertIs(result_residual, residual)


if __name__ == "__main__":
    main()
