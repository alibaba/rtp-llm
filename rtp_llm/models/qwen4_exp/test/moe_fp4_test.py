import os
import unittest
from unittest.mock import patch

from rtp_llm.config.quant_config import QWEN4_FP8_NVFP4_RECIPE, ModelOptFp4Config
from rtp_llm.model_loader.ffn_weight import MoeAtomicWeight, MoeConfig, MoeWeight
from rtp_llm.model_loader.online_modelopt_fp4_quant_weight import (
    OnlineModelOptFp4MoeWeight,
)
from rtp_llm.model_loader.weight_module import AtomicWeight
from rtp_llm.models.qwen4_exp.moe_fp4 import Qwen4MoeFp4WeightInfo
from rtp_llm.utils.model_weight import CkptWeightInfo, W


class Qwen4MoeFp4Test(unittest.TestCase):

    def test_explicit_dense_storage_is_independent_of_ambient_environment(self):
        from rtp_llm.model_loader.fp8_weight import Fp8StorageWeight

        dense = AtomicWeight(W.qwen4_hc_attn_mix_down, [CkptWeightInfo("hc")])
        head = AtomicWeight(W.lm_head, [CkptWeightInfo("lm_head.weight")])
        recipe = ModelOptFp4Config(
            bits=4,
            group_size=16,
            is_quanted=False,
            model_recipe=QWEN4_FP8_NVFP4_RECIPE,
            source_fp8_block_size=128,
        )
        target = Qwen4MoeFp4WeightInfo([head], [[dense]], dense_storage="fp8")
        draft = Qwen4MoeFp4WeightInfo([head], [[dense]], dense_storage="source_bf16")
        with patch.dict(os.environ, RTP_QWEN4_DENSE_STORAGE="source_bf16"):
            converted_target = target.to_quant_weight_info(recipe)
        with patch.dict(os.environ, RTP_QWEN4_DENSE_STORAGE="fp8"):
            converted_draft = draft.to_quant_weight_info(recipe)
        self.assertIsInstance(converted_target.weights[0], Fp8StorageWeight)
        self.assertIsInstance(converted_target.layer_weights[0][0], Fp8StorageWeight)
        self.assertIs(converted_draft.weights[0], head)
        self.assertIs(converted_draft.layer_weights[0][0], dense)

    def test_original_draft_dense_with_shared_fp8_vocab_preserves_experts(self):
        from rtp_llm.model_loader.fp8_weight import Fp8StorageWeight

        cfg = MoeConfig(expert_num=1)
        children = [
            MoeAtomicWeight(n, [CkptWeightInfo(n)], config=cfg)
            for n in (W.moe_w1, W.moe_w2, W.moe_gate)
        ]
        routed = MoeWeight(children, config=cfg)
        dense = AtomicWeight(W.attn_qkv_w, [CkptWeightInfo("qkv")])
        names = (
            W.embedding,
            W.lm_head,
            W.multi_tokens_predict_hnorm,
            W.qwen4_mtp_fc_hidden_w,
        )
        globals_ = [AtomicWeight(n, [CkptWeightInfo(n)]) for n in names]
        info = Qwen4MoeFp4WeightInfo(
            globals_,
            [[routed, dense]],
            dense_storage="source_bf16",
            vocab_storage="fp8",
        )
        converted = info.to_quant_weight_info(
            ModelOptFp4Config(
                bits=4,
                group_size=16,
                is_quanted=False,
                model_recipe=QWEN4_FP8_NVFP4_RECIPE,
                source_fp8_block_size=128,
            )
        )
        for index in (0, 1):
            self.assertIsInstance(converted.weights[index], Fp8StorageWeight)
        for index in (2, 3):
            self.assertIs(converted.weights[index], globals_[index])
        self.assertIs(converted.layer_weights[0][1], dense)
        sub = converted.layer_weights[0][0].sub_weights
        self.assertIs(sub[W.moe_gate], children[-1])
        for name in (W.moe_w1, W.moe_w2):
            self.assertIsInstance(sub[name], OnlineModelOptFp4MoeWeight)

    def test_source_bf16_retains_original_nonexpert_descriptors(self):
        cfg = MoeConfig(expert_num=1)
        children = [
            MoeAtomicWeight(n, [CkptWeightInfo(n)], config=cfg)
            for n in (W.moe_w1, W.moe_w2, W.moe_gate)
        ]
        routed = MoeWeight(children, config=cfg)
        names = (W.qwen4_hc_attn_mix_down, W.attn_qkv_w, W.q_ln_gamma)
        dense = [AtomicWeight(n, [CkptWeightInfo(n)]) for n in names]
        embedding = AtomicWeight(W.embedding, [CkptWeightInfo("embedding")])
        info = Qwen4MoeFp4WeightInfo([embedding], [[routed, *dense]])
        with patch.dict(os.environ, RTP_QWEN4_DENSE_STORAGE="source_bf16"):
            converted = info.to_quant_weight_info(
                ModelOptFp4Config(
                    bits=4,
                    group_size=16,
                    is_quanted=False,
                    model_recipe=QWEN4_FP8_NVFP4_RECIPE,
                    source_fp8_block_size=128,
                )
            )
        self.assertIs(converted.weights[0], embedding)
        for original, actual in zip(dense, converted.layer_weights[0][1:]):
            self.assertIs(actual, original)
        actual = converted.layer_weights[0][0].sub_weights
        self.assertIs(actual[W.moe_gate], children[-1])
        for name in (W.moe_w1, W.moe_w2):
            self.assertIsInstance(actual[name], OnlineModelOptFp4MoeWeight)

    def test_full_fp8_wraps_dense_and_router_but_preserves_nvfp4_experts(self):
        from rtp_llm.model_loader.fp8_weight import Fp8StorageWeight

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
        info = Qwen4MoeFp4WeightInfo([global_weight], [[routed, dense]])
        converted = info.to_quant_weight_info(
            ModelOptFp4Config(
                bits=4,
                group_size=16,
                is_quanted=False,
                model_recipe=QWEN4_FP8_NVFP4_RECIPE,
                source_fp8_block_size=128,
            )
        )
        self.assertIsInstance(converted.weights[0], Fp8StorageWeight)
        self.assertIsInstance(converted.layer_weights[0][1], Fp8StorageWeight)
        children = converted.layer_weights[0][0].sub_weights
        self.assertIsInstance(children[W.moe_gate], Fp8StorageWeight)
        for n in (W.moe_w1, W.moe_w2):
            self.assertIsInstance(children[n], OnlineModelOptFp4MoeWeight)

    def test_refuses_prequantized_or_wrong_block(self):
        for group, pre in ((32, False), (16, True)):
            with self.assertRaisesRegex(ValueError, "complete FP8/NVFP4 recipe"):
                Qwen4MoeFp4WeightInfo([], []).to_quant_weight_info(
                    ModelOptFp4Config(bits=4, group_size=group, is_quanted=pre)
                )


if __name__ == "__main__":
    unittest.main()
