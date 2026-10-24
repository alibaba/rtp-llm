"""Load prequantized 1x32 MXFP8 weights without the 128x128 FP8 requantizer."""

from typing import Dict, Union

import torch

from rtp_llm.config.quant_config import Fp8MxBlockWiseQuantConfig, QuantizationConfig
from rtp_llm.model_loader.load_config import LoadConfig
from rtp_llm.model_loader.per_block_fp8_quant_weight import PerBlockFp8Weight
from rtp_llm.model_loader.weight_module import CompositeWeight, WeightModule
from rtp_llm.utils.model_weight import is_v4_weight


class Mxfp8Weight(PerBlockFp8Weight):
    def __init__(
        self,
        src_weight_info: WeightModule,
        quant_config: QuantizationConfig,
        *args,
        **kwargs,
    ):
        super().__init__(src_weight_info, quant_config, *args, **kwargs)
        if self.scale is not None:
            kernel_split = self.kernel._get_split_func()
            self.scale._get_split_func = lambda _split=kernel_split: _split

    @classmethod
    def support(
        cls, quant_config: QuantizationConfig, src_weight_info: WeightModule
    ) -> bool:
        return (
            quant_config.is_quanted()
            and isinstance(quant_config, Fp8MxBlockWiseQuantConfig)
            and src_weight_info.name in cls.w8a8_weight_list
            and not getattr(src_weight_info, "is_prepacked_nvfp4", False)
            and not is_v4_weight(src_weight_info)
        )

    def _postprocess(
        self,
        tensor: Union[torch.Tensor, Dict[str, torch.Tensor]],
        device: str,
        load_config: LoadConfig,
    ):
        # Preserve the serialized E4M3 values; only decode the UE8M0 exponent
        # bytes. DeepGEMM scale packing is cached by the Linear at first use.
        result = CompositeWeight._postprocess(self, tensor, device, load_config)
        kernel = load_config.exported_device.maybe_rewrite_weight_by_key(
            "weight", result[self.kernel.name]
        )
        result[self.kernel.name] = kernel
        if self.scale is not None:
            raw_scale = load_config.exported_device.maybe_rewrite_weight_by_key(
                "scale", result[self.scale.name]
            )
            result[self.scale.name] = torch.exp2(
                raw_scale.to(device=kernel.device, dtype=torch.float32) - 127.0
            ).contiguous()
        return result
