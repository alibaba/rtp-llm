"""Complete official-FP8 to dense-FP8/routed-NVFP4 weight recipe."""

from rtp_llm.config.quant_config import (
    QWEN4_FP8_NVFP4_RECIPE,
    Fp8BlockWiseQuantConfig,
    ModelOptFp4Config,
)
from rtp_llm.model_loader.ffn_weight import MoeWeight
from rtp_llm.model_loader.model_weight_info import ModelWeightInfo
from rtp_llm.utils.model_weight import W


class Qwen4MoeFp4WeightInfo(ModelWeightInfo):
    _fp8_attention_weights = frozenset(
        {W.attn_qkv_w, W.attn_o_w, W.linear_attn_qkvz_w, W.linear_attn_out_w}
    )

    def to_quant_weight_info(self, quant_config):
        if (
            not isinstance(quant_config, ModelOptFp4Config)
            or quant_config.model_recipe != QWEN4_FP8_NVFP4_RECIPE
            or quant_config.is_quanted()
            or quant_config.bits != 4
            or quant_config.group_size() != 16
            or quant_config.source_fp8_block_size != 128
        ):
            raise ValueError(
                "Qwen4 requires the complete FP8/NVFP4 recipe from official FP8 source"
            )
        from rtp_llm.model_loader.fp8_weight import Fp8StorageWeight

        layers = []
        for weights in self.layer_weights:
            layer = []
            for weight in weights:
                if isinstance(weight, MoeWeight):
                    children = [
                        (
                            child.create(child, quant_config)
                            if child.name in (W.moe_w1, W.moe_w2)
                            else Fp8StorageWeight(child)
                        )
                        for child in weight.sub_weights.values()
                    ]
                    weight = MoeWeight(sub_weights=children, config=weight.config)
                elif weight.name in self._fp8_attention_weights:
                    weight = weight.create(
                        weight, Fp8BlockWiseQuantConfig(is_quanted=False)
                    )
                elif weight.name != W.qwen4_ple_ngram_shards:
                    weight = Fp8StorageWeight(weight)
                layer.append(weight)
            layers.append(layer)
        return ModelWeightInfo([Fp8StorageWeight(w) for w in self.weights], layers)
