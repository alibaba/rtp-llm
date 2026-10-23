"""Native official block-FP8 experts, with complete QSA/PLE FP8 consumers."""

import torch

from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
from rtp_llm.model_loader.per_block_fp8_quant_weight import PerBlockFp8Weight
from rtp_llm.models.qwen4_exp.moe_fp4 import Qwen4MoeFp4WeightInfo
from rtp_llm.utils.model_weight import W


class Qwen4NativeFp8MoeWeight(PerBlockFp8Weight):
    """Preserve E4M3 bytes and source scales across block-aligned TP shards.

    Qwen4's 640-wide experts have five blocks. A naive TP4 split would divide
    640 weights into 160-row pieces but five scales into one-block pieces.
    Pad each up/gate half and the down K axis before the existing TP splitter.
    Padding adds zero channels; no source values are dequantized/requantized.
    """

    @classmethod
    def support(cls, quant_config, src_weight_info):
        # Model-local recipe; do not compete with the generic registry.
        return False

    def __init__(self, source, quant_config):
        super().__init__(source, quant_config, name=source.name)
        # Generic pre-sharding cannot express five blocks split across TP4.
        # Keep the bounded per-layer read and jointly shard weights/scales below.
        self.kernel.enable_pure_tp_preshard = False
        self.scale.enable_pure_tp_preshard = False

    def _split(self, tensor, load_config):
        if not load_config.moe_pure_tp_mode or load_config.tp_size == 1:
            return super()._split(tensor, load_config)
        kernel, scale = tensor[self.kernel.name], tensor[self.scale.name]
        upgate = self.kernel.name == W.moe_w1
        dim = 1 if upgate else 2
        parts = 2 if upgate else 1
        width = kernel.shape[dim] // parts
        blocks = scale.shape[dim] // parts
        if width != blocks * 128:
            raise ValueError(
                "Qwen4 native FP8 experts must have complete 128-wide blocks"
            )
        padded_blocks = (
            (blocks + load_config.tp_size - 1) // load_config.tp_size
        ) * load_config.tp_size
        if padded_blocks != blocks:

            def pad_parts(value, size, padded, fill):
                shape = list(value.shape)
                shape[dim] = parts * padded
                result = torch.full(shape, fill, dtype=value.dtype, device=value.device)
                for part in range(parts):
                    result.narrow(dim, part * padded, size).copy_(
                        value.narrow(dim, part * size, size)
                    )
                return result

            tensor = dict(tensor)
            tensor[self.kernel.name] = pad_parts(kernel, width, padded_blocks * 128, 0)
            tensor[self.scale.name] = pad_parts(scale, blocks, padded_blocks, 1)
        return super()._split(tensor, load_config)


class Qwen4NativeFp8WeightInfo(Qwen4MoeFp4WeightInfo):
    """Reuse dense/QSA/PLE FP8 wiring without converting routed experts to FP4."""

    def _validate_quant_config(self, quant_config):
        if (
            not isinstance(quant_config, Fp8BlockWiseQuantConfig)
            or not quant_config.is_quanted()
            or quant_config.bits != 8
            or quant_config.group_size() != 128
        ):
            raise ValueError(
                "Qwen4 native FP8 requires the official block128 FP8 recipe"
            )

    def _expert_weight(self, weight, quant_config):
        return Qwen4NativeFp8MoeWeight(weight, quant_config)
