"""Preserve ModelOpt NVFP4 experts and the checkpoint's BF16 dense tensors."""

import re

import torch

from rtp_llm.config.quant_config import ModelOptFp4Config
from rtp_llm.model_loader.ffn_weight import MoeWeight
from rtp_llm.model_loader.model_weight_info import ModelWeightInfo
from rtp_llm.model_loader.per_group_fp4_quant_weight import PerGroupFp4Weight
from rtp_llm.utils.model_weight import W


def validate_nvfp4_manifest(config, weight_map):
    quant = config.get("quantization_config") or {}
    if (
        quant.get("quant_method") != "modelopt"
        or str(quant.get("quant_algo", "")).upper() != "NVFP4"
        or type(quant.get("group_size")) is not int
        or quant["group_size"] != 16
        or quant.get("with_input_scale") is not True
    ):
        raise ValueError("Qwen4 NVFP4 requires ModelOpt group-16 quantization")
    text = config["text_config"]
    prefixes = [("model.language_model", int(text["num_hidden_layers"]))]
    if any(name.startswith("mtp.") for name in weight_map):
        prefixes.append(("mtp", 1))
    expected = {
        f"{prefix}.layers.{layer}.mlp.experts.{expert}.{projection}_proj.weight"
        for prefix, layers in prefixes
        for layer in range(layers)
        for expert in range(int(text["num_experts"]))
        for projection in ("gate", "up", "down")
    }
    experts = {
        name
        for name in weight_map
        if re.fullmatch(
            r"(?:model\.language_model|mtp)\.layers\.\d+\.mlp\.experts\.\d+\.(?:gate|up|down)_proj\.weight",
            name,
        )
    }
    if experts != expected:
        raise ValueError("Qwen4 NVFP4 requires complete split expert projections")
    for suffix in (".weight_scale", ".weight_scale_2", ".input_scale"):
        actual = {name for name in weight_map if name.endswith(suffix)}
        if actual != {name.removesuffix(".weight") + suffix for name in expected}:
            raise ValueError("Qwen4 NVFP4 scales must cover exactly the routed experts")


class Qwen4Nvfp4MoeWeight(PerGroupFp4Weight):
    supports_fastsafetensors_iteration = False

    def __init__(self, src_weight_info, quant_config):
        super().__init__(src_weight_info, quant_config, name=src_weight_info.name)

    @classmethod
    def support(cls, quant_config, src_weight_info):
        # Installed only by the model-owned weight mapping.
        return False

    def _load_raw_tensor(self, tensor_source, layer_id, device, load_config):
        if self.kernel.name == W.moe_w1:
            # The common fused descriptor has one outer scale for up/gate.
            # Preserve the checkpoint exactly; never replace unequal scales
            # with their maximum without rescaling the packed weights.
            for expert in load_config.get_selected_experts(
                layer_id, self.kernel.config.expert_num
            ):
                scales = [
                    tensor_source.load_tensor(
                        self.kernel._ckpt_name(weight, layer_id, expert).removesuffix(
                            ".weight"
                        )
                        + ".weight_scale_2",
                        torch.float32,
                    )
                    for weight in self.kernel.weights
                ]
                if (
                    len(scales) != 2
                    or any(len(scale) != 1 or scale[0].numel() != 1 for scale in scales)
                    or not torch.equal(scales[0][0], scales[1][0])
                ):
                    raise ValueError("Qwen4 NVFP4 requires equal up/gate outer scales")
        return super()._load_raw_tensor(tensor_source, layer_id, device, load_config)


class Qwen4Nvfp4WeightInfo(ModelWeightInfo):
    def to_quant_weight_info(self, quant_config):
        if (
            not isinstance(quant_config, ModelOptFp4Config)
            or not quant_config.is_quanted()
            or quant_config.group_size() != 16
        ):
            raise ValueError("Qwen4 NVFP4 requires a prequantized group-16 recipe")
        layers = []
        for layer in self.layer_weights:
            converted = []
            for weight in layer:
                if isinstance(weight, MoeWeight):
                    weight = MoeWeight(
                        [
                            (
                                Qwen4Nvfp4MoeWeight(child, quant_config)
                                if child.name in (W.moe_w1, W.moe_w2)
                                else child
                            )
                            for child in weight.sub_weights.values()
                        ],
                        config=weight.config,
                    )
                converted.append(weight)
            layers.append(converted)
        return ModelWeightInfo(self.weights, layers)
