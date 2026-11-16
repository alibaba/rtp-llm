"""Load ModelOpt NVFP4 experts without dequantization or requantization."""

import torch

from rtp_llm.model_loader.online_modelopt_fp4_quant_weight import (
    OnlineMegaMoeFp4FromNvfp4Weight,
)
from rtp_llm.utils.model_weight import W


class NativeMegaMoeNvfp4Weight(OnlineMegaMoeFp4FromNvfp4Weight):
    """Keep packed E2M1, E4M3 block scales and independent projection scales."""

    def __init__(self, src_kernel, **kwargs):
        super().__init__(src_kernel, **kwargs)
        self.global_scale_name = W.moe_w1_s2 if self.name == W.moe_w1 else W.moe_w2_s2

    def _load_raw_tensor(self, tensor_source, layer_id, device, load_config):
        experts = list(
            load_config.get_selected_experts(layer_id, self.kernel.config.expert_num)
        )
        if not experts:
            raise ValueError("Native NVFP4 requires at least one local expert")
        packed_out = scales_out = None
        globals_out = []
        for local_id, expert_id in enumerate(experts):
            packed_parts, scale_parts, globals_parts = [], [], []
            for ckpt in self.kernel.weights:
                name = ckpt.name.format(
                    i=layer_id, i_1=layer_id + 1, expert_id=expert_id
                )
                base = name.removesuffix(".weight")
                packed = ckpt.merge_fun(
                    tensor_source.load_tensor(name, torch.uint8)
                ).to(device)
                scale = ckpt.merge_fun(
                    tensor_source.load_tensor(
                        base + ".weight_scale", torch.float8_e4m3fn
                    )
                ).to(device)
                global_scale = ckpt.merge_fun(
                    tensor_source.load_tensor(base + ".weight_scale_2", torch.float32)
                ).to(device)
                if (
                    packed.ndim != 2
                    or packed.shape[1] % 8
                    or scale.shape != (packed.shape[0], packed.shape[1] // 8)
                    or global_scale.numel() != 1
                ):
                    raise ValueError(
                        f"Invalid native NVFP4 block-16 projection: {name}"
                    )
                packed_parts.append(packed)
                scale_parts.append(scale.view(torch.uint8))
                globals_parts.append(global_scale.reshape(()))
            # RTP W1 is [up | gate]. Preserve the same order for both scales.
            packed = torch.cat(packed_parts, dim=0)
            scale = torch.cat(scale_parts, dim=0)
            if packed_out is None:
                packed_out = torch.empty(
                    (len(experts), *packed.shape), dtype=torch.uint8, device=device
                )
                scales_out = torch.empty(
                    (len(experts), *scale.shape), dtype=torch.uint8, device=device
                )
            packed_out[local_id].copy_(packed)
            scales_out[local_id].copy_(scale)
            globals_out.append(torch.stack(globals_parts))
        globals_tensor = torch.stack(globals_out)
        if self.name == W.moe_w2:
            globals_tensor = globals_tensor[:, 0].contiguous()
        return {
            self.kernel.name: packed_out.view(torch.int8),
            self.scale.name: scales_out.view(torch.float8_e4m3fn),
            self.global_scale_name: globals_tensor,
        }

    def _split(self, tensor, load_config):
        # Packed bytes and block scales use existing TP/EP splitting. Scalar
        # scales already follow selected-expert order and are replicated in TP.
        result = super()._split(tensor, load_config)
        result[self.global_scale_name] = tensor[self.global_scale_name]
        return result

    def _postprocess(self, tensor, device, load_config):
        return tensor
