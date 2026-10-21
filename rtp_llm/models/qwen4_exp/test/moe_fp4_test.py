import unittest

from rtp_llm.config.quant_config import QWEN4_FP8_NVFP4_RECIPE, ModelOptFp4Config
from rtp_llm.model_loader.ffn_weight import MoeAtomicWeight, MoeConfig, MoeWeight
from rtp_llm.model_loader.online_modelopt_fp4_quant_weight import (
    OnlineModelOptFp4MoeWeight,
)
from rtp_llm.model_loader.weight_module import AtomicWeight
from rtp_llm.models.qwen4_exp.moe_fp4 import Qwen4MoeFp4WeightInfo
from rtp_llm.utils.model_weight import CkptWeightInfo, W


class Qwen4MoeFp4Test(unittest.TestCase):

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
