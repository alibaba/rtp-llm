"""Loader for compressed-tensors W8A8 INT8 per-channel checkpoints."""

import torch

from rtp_llm.config.quant_config import CompressedW8A8Int8PerChannelQuantConfig
from rtp_llm.model_loader.per_channel_fp8_quant_weight import (
    PerChannelFp8Weight,
    _ckpt_base_matches_quant_exclude,
    _ckpt_base_matches_regex_exclude,
)
from rtp_llm.model_loader.weight_module import WeightModule


class CompressedW8A8Int8PerChannelWeight(PerChannelFp8Weight):
    """Load INT8 kernels and FP32 channel scales without re-quantization.

    Tensor mappings, TP/EP split rules and exclude handling are identical to the
    per-channel FP8 path, so only the checkpoint dtype and the device
    post-processing differ: INT8 kernels stay byte-exact and do not pass through
    FP8 layout conversion.
    """

    weight_dtype = torch.int8
    apply_fp8_device_conversion = False
    supported_quant_config_types = (CompressedW8A8Int8PerChannelQuantConfig,)

    @classmethod
    def support(
        cls,
        quant_config: CompressedW8A8Int8PerChannelQuantConfig,
        src_weight_info: WeightModule,
    ) -> bool:
        if (
            not quant_config.is_quanted()
            or not isinstance(quant_config, cls.supported_quant_config_types)
            or src_weight_info.name not in cls.w8a8_weight_list
        ):
            return False
        for ckpt_w in src_weight_info.weights:
            base_name = ckpt_w.name.rsplit(".", 1)[0]
            if (
                base_name not in quant_config.exclude_modules
                and _ckpt_base_matches_quant_exclude(
                    base_name, quant_config.exclude_modules
                )
                and not _ckpt_base_matches_regex_exclude(
                    base_name, quant_config.exclude_modules
                )
            ):
                raise ValueError(
                    "W8A8 compressed-tensors ignore targets only some instances "
                    f"of quantized weight template {base_name}; per-layer fallback "
                    "is not supported"
                )
        return super().support(quant_config, src_weight_info)
