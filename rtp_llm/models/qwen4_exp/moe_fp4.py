"""Selective online NVFP4 conversion for Qwen routed experts."""

from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig, ModelOptFp4Config
from rtp_llm.model_loader.ffn_weight import MoeWeight
from rtp_llm.model_loader.model_weight_info import ModelWeightInfo
from rtp_llm.utils.model_weight import W


class Qwen4MoeFp4WeightInfo(ModelWeightInfo):
    _fp8_attention_weights = frozenset(
        {W.attn_qkv_w, W.attn_o_w, W.linear_attn_qkvz_w, W.linear_attn_out_w}
    )

    def __init__(self, *args, attention_fp8=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention_fp8 = attention_fp8

    def to_quant_weight_info(self, quant_config):
        if (
            not isinstance(quant_config, ModelOptFp4Config)
            or quant_config.is_quanted()
            or quant_config.group_size() != 16
        ):
            raise ValueError(
                "Qwen4 online expert NVFP4 requires BF16 source and block16"
            )
        layers = []
        for weights in self.layer_weights:
            layer = []
            for weight in weights:
                if isinstance(weight, MoeWeight):
                    children = [
                        (
                            child.create(child, quant_config)
                            if child.name in (W.moe_w1, W.moe_w2)
                            else child
                        )
                        for child in weight.sub_weights.values()
                    ]
                    weight = MoeWeight(sub_weights=children, config=weight.config)
                elif self.attention_fp8 and weight.name in self._fp8_attention_weights:
                    weight = weight.create(
                        weight, Fp8BlockWiseQuantConfig(is_quanted=False)
                    )
                layer.append(weight)
            layers.append(layer)
        return ModelWeightInfo(list(self.weights), layers)
