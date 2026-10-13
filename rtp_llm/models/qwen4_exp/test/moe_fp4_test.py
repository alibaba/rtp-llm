import unittest

from rtp_llm.config.quant_config import ModelOptFp4Config
from rtp_llm.model_loader.ffn_weight import MoeAtomicWeight, MoeConfig, MoeWeight
from rtp_llm.model_loader.online_modelopt_fp4_quant_weight import (
    OnlineModelOptFp4MoeWeight,
)
from rtp_llm.model_loader.per_block_fp8_quant_weight import LoadQuantPerBlockFp8Weight
from rtp_llm.model_loader.weight_module import AtomicWeight
from rtp_llm.models.qwen4_exp.moe_fp4 import Qwen4MoeFp4WeightInfo
from rtp_llm.utils.model_weight import CkptWeightInfo, W


class Qwen4MoeFp4Test(unittest.TestCase):
    def test_only_routed_experts_are_quantized(self):
        cfg = MoeConfig(expert_num=2)
        children = [
            MoeAtomicWeight(n, [CkptWeightInfo("x")], config=cfg)
            for n in (W.moe_w1, W.moe_w2)
        ]
        gate = MoeAtomicWeight(W.moe_gate, [CkptWeightInfo("gate")], config=cfg)
        shared = AtomicWeight("shared", [CkptWeightInfo("shared")])
        moe = MoeWeight(sub_weights=[*children, gate], config=cfg)
        info = Qwen4MoeFp4WeightInfo([shared], [[moe, shared]])
        result = info.to_quant_weight_info(
            ModelOptFp4Config(bits=4, group_size=16, is_quanted=False)
        )
        self.assertIs(result.weights[0], shared)
        self.assertIs(result.layer_weights[0][1], shared)
        out = result.layer_weights[0][0].sub_weights
        for n in (W.moe_w1, W.moe_w2):
            self.assertIsInstance(out[n], OnlineModelOptFp4MoeWeight)
        self.assertIs(out[gate.name], gate)

    def test_attention_fp8_keeps_indexer_shared_hc_and_ple_unchanged(self):
        converted = [
            AtomicWeight(n, [CkptWeightInfo(n)])
            for n in (
                W.attn_qkv_w,
                W.attn_o_w,
                W.linear_attn_qkvz_w,
                W.linear_attn_out_w,
            )
        ]
        exceptions = [
            AtomicWeight(n, [CkptWeightInfo(n)])
            for n in (
                W.qwen4_indexer_qk_proj_w,
                W.shared_expert_gate,
                W.qwen4_hc_attn_mix_down,
                W.qwen4_ple_key_proj_w,
                W.linear_attn_ba_w,
                W.lm_head,
                W.embedding,
            )
        ]
        info = Qwen4MoeFp4WeightInfo(
            [], [[*converted, *exceptions]], attention_fp8=True
        )
        out = info.to_quant_weight_info(
            ModelOptFp4Config(bits=4, group_size=16, is_quanted=False)
        )
        for weight in out.layer_weights[0][:4]:
            self.assertIsInstance(weight, LoadQuantPerBlockFp8Weight)
            self.assertEqual(weight.quant_config.get_method(), "FP8_PER_BLOCK")
        for actual, original in zip(out.layer_weights[0][4:], exceptions):
            self.assertIs(actual, original)

    def test_gdn_bf16_keeps_full_attention_fp8_and_routed_nvfp4(self):
        cfg = MoeConfig(expert_num=2)
        routed = MoeWeight(
            sub_weights=[
                MoeAtomicWeight(n, [CkptWeightInfo(n)], config=cfg)
                for n in (W.moe_w1, W.moe_w2)
            ],
            config=cfg,
        )
        names = (W.attn_qkv_w, W.attn_o_w, W.linear_attn_qkvz_w, W.linear_attn_out_w)
        linears = [AtomicWeight(n, [CkptWeightInfo(n)]) for n in names]
        info = Qwen4MoeFp4WeightInfo(
            [], [[*linears, routed]], attention_fp8=True, gdn_fp8=False
        )
        out = info.to_quant_weight_info(
            ModelOptFp4Config(bits=4, group_size=16, is_quanted=False)
        ).layer_weights[0]
        for weight in out[:2]:
            self.assertIsInstance(weight, LoadQuantPerBlockFp8Weight)
        for actual, original in zip(out[2:4], linears[2:]):
            self.assertIs(actual, original)
        for name in (W.moe_w1, W.moe_w2):
            self.assertIsInstance(out[4].sub_weights[name], OnlineModelOptFp4MoeWeight)

    def test_full_fp8_wraps_dense_and_router_but_preserves_nvfp4_experts(self):
        from rtp_llm.models.qwen4_exp.full_fp8_weight import Qwen4Fp8StorageWeight

        cfg = MoeConfig(expert_num=1)
        routed = MoeWeight(
            [
                MoeAtomicWeight(n, [CkptWeightInfo(n)], config=cfg)
                for n in (W.moe_w1, W.moe_w2, W.moe_gate)
            ],
            config=cfg,
        )
        dense = AtomicWeight(W.qwen4_hc_attn_mix_down, [CkptWeightInfo("hc")])
        global_weight = AtomicWeight(W.embedding, [CkptWeightInfo("embedding")])
        info = Qwen4MoeFp4WeightInfo([global_weight], [[routed, dense]], full_fp8=True)
        converted = info.to_quant_weight_info(
            ModelOptFp4Config(bits=4, group_size=16, is_quanted=False)
        )
        self.assertIsInstance(converted.weights[0], Qwen4Fp8StorageWeight)
        self.assertIsInstance(converted.layer_weights[0][1], Qwen4Fp8StorageWeight)
        children = converted.layer_weights[0][0].sub_weights
        self.assertIsInstance(children[W.moe_gate], Qwen4Fp8StorageWeight)
        for n in (W.moe_w1, W.moe_w2):
            self.assertIsInstance(children[n], OnlineModelOptFp4MoeWeight)

    def test_refuses_prequantized_or_wrong_block(self):
        for group, pre in ((32, False), (16, True)):
            with self.assertRaisesRegex(ValueError, "BF16 source and block16"):
                Qwen4MoeFp4WeightInfo([], []).to_quant_weight_info(
                    ModelOptFp4Config(bits=4, group_size=group, is_quanted=pre)
                )


if __name__ == "__main__":
    unittest.main()
