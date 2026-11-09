"""Adapt compressed-tensors NVFP4 storage to the existing native FP4 backend."""

import functools

from rtp_llm.config.nvfp4_utils import reciprocal_global_scale
from rtp_llm.config.quant_config import CompressedTensorsNvfp4Config
from rtp_llm.model_loader.per_group_fp4_quant_weight import (
    PerGroupFp4Weight,
    _is_fp4_weight_excluded,
)
from rtp_llm.utils.model_weight import CkptWeightInfo, W, is_v4_weight, pad


class CompressedTensorsNvfp4Weight(PerGroupFp4Weight):
    @classmethod
    def support(cls, quant_config, src_weight_info):
        if not isinstance(quant_config, CompressedTensorsNvfp4Config):
            return False
        if _is_fp4_weight_excluded(quant_config, src_weight_info):
            return False
        if src_weight_info.name in (W.moe_w1, W.moe_w2) or is_v4_weight(
            src_weight_info
        ):
            raise ValueError(
                "compressed-tensors NVFP4 currently supports dense models only"
            )
        return src_weight_info.name in cls.w4a4_weight_list

    def __init__(self, src_weight_info, quant_config, *args, **kwargs):
        super().__init__(src_weight_info, quant_config, *args, **kwargs)
        for atomic, old, new in (
            (self.kernel, ".weight", ".weight_packed"),
            (self.scale_2, ".weight_scale_2", ".weight_global_scale"),
            (self.input_scale, ".input_scale", ".input_global_scale"),
        ):
            atomic.weights = [
                CkptWeightInfo(w.name.removesuffix(old) + new, w.merge_fun)
                for w in atomic.weights
            ]
        # CT stores divisors; the backend stores dequantization multipliers.
        # Different fused globals require requantization, not max()/averaging.
        self.scale_2.process_fun = reciprocal_global_scale
        self.input_scale.process_fun = reciprocal_global_scale
        if src_weight_info.name == W.ffn_w2:
            # align_size counts original K channels, not bytes or scale groups.
            align = src_weight_info.config.align_size
            if align % 16:
                raise ValueError("NVFP4 K padding alignment must be divisible by 16")
            self.kernel.process_fun = functools.partial(
                pad, align_size=align // 2, dim=1
            )
            self.scale.process_fun = functools.partial(
                pad, align_size=align // 16, dim=1
            )

    def _load_raw_tensor(self, tensor_source, layer_id, device, load_config):
        tensors = super()._load_raw_tensor(tensor_source, layer_id, device, load_config)
        packed = tensors[self.kernel.name]
        scales = tensors[self.scale.name]
        if (
            packed.ndim != 2
            or scales.ndim != 2
            or packed.shape[0] != scales.shape[0]
            or packed.shape[1] != scales.shape[1] * 8
        ):
            raise ValueError("NVFP4 expects packed [N,K/2] and local scales [N,K/16]")
        return tensors
