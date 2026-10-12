"""Selective online NVFP4 conversion for Qwen routed experts."""

from rtp_llm.config.quant_config import ModelOptFp4Config
from rtp_llm.model_loader.ffn_weight import MoeWeight
from rtp_llm.model_loader.model_weight_info import ModelWeightInfo
from rtp_llm.utils.model_weight import W


class Qwen4MoeFp4WeightInfo(ModelWeightInfo):
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
                layer.append(weight)
            layers.append(layer)
        return ModelWeightInfo(list(self.weights), layers)
