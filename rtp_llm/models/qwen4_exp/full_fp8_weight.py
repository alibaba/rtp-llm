"""FP8 storage for the released checkpoint's unquantized dense parameters."""

import torch

from rtp_llm.model_loader.weight_module import (
    CompositeWeight,
    QuantWeight,
    WeightModule,
)

FP8_STORAGE_SCALE_SUFFIX = ".qwen4_fp8_storage_scale"


class Qwen4Fp8StorageWeight(CompositeWeight, QuantWeight):
    """Quantize after the original descriptor's TP split and layout conversion."""

    supports_fastsafetensors_iteration = False

    def __init__(self, source):
        WeightModule.__init__(self, name=source.name)
        self.sub_weights = {source.name: source}
        self.resident_device = source.resident_device
        self.pinned_memory = source.pinned_memory
        self.resident_output_devices = dict(source.resident_output_devices)
        self.additional_output_names = tuple(
            c.name + FP8_STORAGE_SCALE_SUFFIX for c in source.get_components()
        )

    @classmethod
    def support(cls, quant_config, src_weight_info):
        return False

    def update(self, *args, **kwargs):
        raise PermissionError(
            "Qwen full FP8 parameters require a service restart to update"
        )

    def _postprocess(self, tensor, device, load_config):
        processed = super()._postprocess(tensor, device, load_config)
        result = {}

        def quantize(items):
            for name, value in items.items():
                if isinstance(value, dict):
                    quantize(value)
                elif value.dtype in (torch.float16, torch.bfloat16, torch.float32):
                    values = value.float()
                    scale = values.abs().amax().clamp_min(1e-12).reshape(1) / 448.0
                    result[name] = (
                        (values / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
                    )
                    result[name + FP8_STORAGE_SCALE_SUFFIX] = scale
                else:
                    result[name] = value

        quantize(processed)
        return result
