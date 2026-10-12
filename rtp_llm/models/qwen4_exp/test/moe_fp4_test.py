import unittest
from rtp_llm.config.quant_config import ModelOptFp4Config
from rtp_llm.model_loader.ffn_weight import MoeAtomicWeight, MoeConfig, MoeWeight
from rtp_llm.model_loader.online_modelopt_fp4_quant_weight import (
    OnlineModelOptFp4MoeWeight,
)
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

    def test_refuses_prequantized_or_wrong_block(self):
        for group, pre in ((32, False), (16, True)):
            with self.assertRaisesRegex(ValueError, "BF16 source and block16"):
                Qwen4MoeFp4WeightInfo([], []).to_quant_weight_info(
                    ModelOptFp4Config(bits=4, group_size=group, is_quanted=pre)
                )


if __name__ == "__main__":
    unittest.main()
