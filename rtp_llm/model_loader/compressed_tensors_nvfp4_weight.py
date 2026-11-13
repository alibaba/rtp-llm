"""Adapt compressed-tensors NVFP4 storage to the existing native FP4 backend."""

import functools

from rtp_llm.config.nvfp4_utils import reciprocal_global_scale
from rtp_llm.config.quant_config import CompressedTensorsNvfp4Config
from rtp_llm.model_loader.per_group_fp4_quant_weight import (
    PerGroupFp4Weight,
    _is_fp4_weight_excluded,
)
from rtp_llm.utils.model_weight import CkptWeightInfo, W, is_v4_weight, pad


def _validate_packed_scale_shapes(packed_shape, scale_shape):
    if (
        len(packed_shape) != 2
        or len(scale_shape) != 2
        or packed_shape[0] != scale_shape[0]
        or packed_shape[1] != scale_shape[1] * 8
    ):
        raise ValueError("NVFP4 expects packed [N,K/2] and local scales [N,K/16]")


class _ShapeCheckedSource:
    """Check paired checkpoint shapes before padding can conceal a mismatch.

    Forward the original tensors unchanged; retain only per-shard shapes.
    This wrapper is local to a single load and does not mutate shared atomics.
    """

    def __init__(self, source):
        self._source = source
        self._shapes = {}

    def __getattr__(self, name):
        return getattr(self._source, name)

    def load_tensor(self, name, data_type):
        tensors = self._source.load_tensor(name, data_type)
        for kind, suffix in (("packed", ".weight_packed"), ("scales", ".weight_scale")):
            if not name.endswith(suffix):
                continue
            pair = self._shapes.setdefault(name.removesuffix(suffix), {})
            pair[kind] = [tuple(t.shape) for t in tensors]
            if "packed" in pair and "scales" in pair:
                if len(pair["packed"]) != len(pair["scales"]):
                    raise ValueError(
                        "NVFP4 packed weights and local scales have different shard counts"
                    )
                for packed_shape, scale_shape in zip(pair["packed"], pair["scales"]):
                    _validate_packed_scale_shapes(packed_shape, scale_shape)
            break
        return tensors


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
        try:
            tensors = super()._load_raw_tensor(
                _ShapeCheckedSource(tensor_source), layer_id, device, load_config
            )
        except ValueError as error:
            names = [w.tensor_name(layer_id) for w in self.kernel.weights]
            raise ValueError(
                f"NVFP4 layer {layer_id}, modules {names}: {error}"
            ) from error
        packed = tensors[self.kernel.name]
        scales = tensors[self.scale.name]
        _validate_packed_scale_shapes(packed.shape, scales.shape)
        return tensors
