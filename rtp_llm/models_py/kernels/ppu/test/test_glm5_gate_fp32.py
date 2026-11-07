"""The gate must retain differences that disappear after BF16 output rounding."""

import os
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import Mock, patch

import torch
from torch import nn
from rtp_llm.models_py.model_desc.generic_moe import GenericMoeLayer
from rtp_llm.models_py.model_desc.glm5 import Glm5MoeDecoderLayer
from rtp_llm.models_py.modules.base.cuda.select_topk import GroupTopK
from rtp_llm.models_py.modules.factory.linear.impl.cuda.f16_linear import CudaF16Linear
from rtp_llm.utils.model_weight import W


class Glm5GateFp32Test(TestCase):
    def make_layer(self, model_type="glm_5"):
        weight = torch.zeros(256, 6144, dtype=torch.bfloat16, device="cuda")
        weight[:7, 0] = 2
        weight[7:9].fill_(2 ** -12)
        weight[8, 0] = 2 ** -12 + 2 ** -19
        config = SimpleNamespace(
            model_type=model_type, hidden_size=6144, inter_size=2048,
            expert_num=256, eplb_config=SimpleNamespace(phy_exp_num=lambda n: n),
            moe_k=8, quant_config=None, moe_style=2, has_moe_norm=True,
            moe_n_group=1, moe_topk_group=1, routed_scaling_factor=2.5,
            moe_layer_index=[3], layernorm_eps=1e-6,
            attn_config=SimpleNamespace(use_mla=False), getAttentionConfigs=lambda _: None,
        )
        parallel = SimpleNamespace(get_ffn_tp_size=lambda: 1, get_attn_tp_size=lambda: 1, ep_size=8)
        backend = SimpleNamespace(includes_shared_expert=True, topk_ids_dtype=torch.int64)
        package = "rtp_llm.models_py.model_desc.generic_moe."
        with (
            patch(package + "LinearFactory.create_linear_from_weights", return_value=CudaF16Linear(weight.t().contiguous())),
            patch(package + "SelectTopk"),
            patch.object(GenericMoeLayer, "_create_fused_moe", return_value=backend),
            patch.dict(os.environ, {"RTP_LLM_GLM5_CMP": "0"}),
            patch("rtp_llm.models_py.model_desc.glm5.CausalAttention", nn.Identity),
        ):
            weights = {W.e_score_correction_b: torch.zeros(256, device="cuda"),
                       W.pre_ln_gamma: torch.ones(6144, device="cuda"),
                       W.post_ln_gamma: torch.ones(6144, device="cuda")}
            decoder = Glm5MoeDecoderLayer(config, parallel, weights, {}, 3,
                                         SimpleNamespace(fake_balance_expert=False, moe_strategy="auto"))
            layer = decoder.mlp
        self.assertIs(type(layer), GenericMoeLayer)
        return layer

    def test_fp32_output_preserves_the_eighth_expert_decision(self):
        layer = self.make_layer()
        x = torch.ones(1, 6144, dtype=torch.bfloat16, device="cuda")
        result = layer._gate_logits(x)
        self.assertEqual(result.dtype, torch.float32)
        self.assertEqual(float(result[0, 7]), 1.5)
        self.assertGreater(float(result[0, 8]), 1.5)
        rounded = torch.nn.functional.linear(x, layer._ppu_gate_weight_nt).float()
        self.assertEqual(float(rounded[0, 7]), float(rounded[0, 8]))
        ids = torch.empty(1, 8, dtype=torch.int64, device="cuda")
        weights = torch.empty(1, 8, device="cuda")
        GroupTopK()(weights, ids, result, torch.zeros(256, device="cuda"), 1, 1, 8, True, 2.5)
        self.assertEqual(sorted(ids.cpu().flatten().tolist()), [0, 1, 2, 3, 4, 5, 6, 8])
        routed_ids, _ = layer._route(x)
        self.assertEqual(sorted(routed_ids.cpu().flatten().tolist()), [0, 1, 2, 3, 4, 5, 6, 8])

    def test_graph_replay_reads_new_inputs_and_empty_rows_are_valid(self):
        layer = self.make_layer()
        x = torch.ones(1, 6144, dtype=torch.bfloat16, device="cuda")
        for _ in range(3): layer._gate_logits(x)
        torch.cuda.synchronize()
        graph, stream = torch.cuda.CUDAGraph(), torch.cuda.Stream()
        with torch.cuda.graph(graph, stream=stream): result = layer._gate_logits(x)
        for value in (2, 3):
            x.fill_(value)
            graph.replay()
            torch.cuda.synchronize()
            self.assertEqual(float(result[0, 7]), value * 1.5)
            self.assertGreater(float(result[0, 8]), value * 1.5)
        self.assertEqual(layer._gate_logits(x[:0]).shape, (0, 256))

    def test_legacy_route_remains_available_without_ppu_weight(self):
        layer = self.make_layer()
        layer._ppu_gate_weight_nt = None
        x = torch.ones(1, 6144)
        with patch.object(layer.gate, "forward", return_value=torch.ones(1, 256, dtype=torch.bfloat16)) as forward:
            self.assertEqual(layer._gate_logits(x).dtype, torch.bfloat16)
        forward.assert_called_once_with(x)

    def test_other_models_keep_their_bf16_output_gate(self):
        layer = self.make_layer(model_type="qwen3_moe")
        self.assertIsNone(layer._ppu_gate_weight_nt)
        x = torch.ones(1, 6144, device="cuda", dtype=torch.bfloat16)
        self.assertEqual(layer._gate_logits(x).dtype, torch.bfloat16)


if __name__ == "__main__":
    main()
