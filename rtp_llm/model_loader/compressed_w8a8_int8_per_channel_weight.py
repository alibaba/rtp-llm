"""Loader for compressed-tensors W8A8 INT8 per-channel checkpoints."""

import torch

from rtp_llm.config.quant_config import CompressedW8A8Int8PerChannelQuantConfig
from rtp_llm.model_loader.per_channel_fp8_quant_weight import PerChannelFp8Weight


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
