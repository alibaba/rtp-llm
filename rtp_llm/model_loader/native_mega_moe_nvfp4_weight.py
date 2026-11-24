"""Native NVFP4 experts, preserved from checkpoints or quantized at load time."""

import torch

from rtp_llm.model_loader.offline_modelopt_fp4_quant_weight import (
    OfflineMegaMoeFp4MoeWeight,
)
from rtp_llm.model_loader.online_modelopt_fp4_quant_weight import (
    MEGA_MOE_FP4_BLOCK,
    NVFP4_BLOCK_SIZE,
    OnlineMegaMoeFp4FromFp8Weight,
    OnlineMegaMoeFp4FromNvfp4Weight,
    OnlineMegaMoeFp4Weight,
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


def quantize_projection_to_native_nvfp4(weight):
    """Use DeepGEMM's block-16 E4M3 format and a projection-wide FP32 scale."""
    try:
        from deep_gemm.utils import per_block_cast_to_nvfp4
        from deep_gemm.utils.math import nvfp4_global_scale
    except ImportError as error:
        raise RuntimeError(
            "NVFP4 expert conversion requires DeepGEMM 2.8.2+161614f.cu132 "
            "or compatible NVFP4 quantizers"
        ) from error

    gsf = nvfp4_global_scale(weight.float().abs().amax())
    packed, scale = per_block_cast_to_nvfp4(weight, weight_gsf=gsf)
    return packed, scale, gsf


def _quantize_experts(name, weight_shape, device, read_expert):
    experts, rows, hidden = weight_shape
    parts = 2 if name == W.moe_w1 else 1
    if not experts or not rows or not hidden or hidden % NVFP4_BLOCK_SIZE:
        raise ValueError(
            "NVFP4 expert weights require nonempty, block-16-aligned shapes"
        )
    if rows % parts:
        raise ValueError("NVFP4 W1 must contain equal up/gate projections")
    packed = torch.empty((experts, rows, hidden // 2), dtype=torch.int8, device=device)
    scales = torch.empty(
        (experts, rows, hidden // 16), dtype=torch.float8_e4m3fn, device=device
    )
    globals_out = torch.empty((experts, parts), dtype=torch.float32, device=device)
    for expert in range(experts):
        value = read_expert(expert)
        for part, projection in enumerate(value.chunk(parts, dim=0)):
            p, s, g = quantize_projection_to_native_nvfp4(projection)
            start, end = part * (rows // parts), (part + 1) * (rows // parts)
            packed[expert, start:end].copy_(p)
            scales[expert, start:end].copy_(s)
            globals_out[expert, part].copy_(g)
    return packed, scales, globals_out if parts == 2 else globals_out[:, 0].contiguous()


class _NativeOutput:
    def _split(self, tensor, load_config):
        result = super()._split(tensor, load_config)
        key = W.moe_w1_s2 if self.name == W.moe_w1 else W.moe_w2_s2
        result[key] = tensor[key]
        return result

    def _postprocess(self, tensor, device, load_config):
        return tensor

    def _output(self, weight_shape, device, scale_name, read_expert):
        packed, scale, gsf = _quantize_experts(
            self.name, weight_shape, device, read_expert
        )
        key = W.moe_w1_s2 if self.name == W.moe_w1 else W.moe_w2_s2
        return {self.name: packed, scale_name: scale, key: gsf}


def dequantize_mxfp4_expert(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Decode packed E2M1 with one MX scale per 32 logical input values."""
    if packed.ndim != 2 or packed.dtype not in (torch.int8, torch.uint8):
        raise ValueError("MXFP4 experts require 2D packed FP4 weight bytes")
    rows, packed_width = packed.shape
    width = packed_width * 2
    groups = width // MEGA_MOE_FP4_BLOCK
    if width % MEGA_MOE_FP4_BLOCK or scale.shape != (rows, groups):
        raise ValueError(
            f"Invalid MXFP4 scale shape {tuple(scale.shape)} for {(rows, width)}"
        )

    if scale.dtype == torch.uint8:
        scale = scale.view(torch.float8_e8m0fnu)
    if scale.dtype not in (torch.float8_e8m0fnu, torch.float32):
        raise ValueError(f"Unsupported MXFP4 scale dtype {scale.dtype}")
    codes = packed.view(torch.uint8)
    dtype_values = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
        device=packed.device,
    )
    codes = torch.stack((codes & 0xF, codes >> 4), dim=-1).reshape(
        rows, groups, MEGA_MOE_FP4_BLOCK
    )
    values = dtype_values[(codes & 7).long()]
    values = torch.where(codes & 8 != 0, -values, values)
    values.mul_(scale.float().unsqueeze(-1))
    return values.reshape(rows, width).to(torch.bfloat16)


class OnlineMegaMoeNvfp4FromMxfp4Weight(_NativeOutput, OfflineMegaMoeFp4MoeWeight):
    """Requantize checkpoint MXFP4 once, one local expert at a time."""

    def _load_raw_tensor(self, tensor_source, layer_id, device, load_config):
        raw = super()._load_raw_tensor(tensor_source, layer_id, device, load_config)
        packed, scale = raw[self.kernel.name], raw[self.scale.name]
        experts, rows, packed_width = packed.shape
        return self._output(
            (experts, rows, packed_width * 2),
            packed.device,
            self.scale.name,
            lambda expert: dequantize_mxfp4_expert(packed[expert], scale[expert]),
        )


class OnlineMegaMoeNvfp4FromFp8Weight(_NativeOutput, OnlineMegaMoeFp4FromFp8Weight):
    """Convert FP8 experts once at load time; inference uses NVFP4 x NVFP4."""

    def _load_raw_tensor(self, tensor_source, layer_id, device, load_config):
        weight = self.kernel._load_raw_tensor(
            tensor_source, layer_id, device, load_config
        )[self.kernel.name]
        scale = self.fp8_scale._load_raw_tensor(
            tensor_source, layer_id, device, load_config
        )[self.fp8_scale.name]
        experts, rows, hidden = weight.shape
        block = self._source_block_size
        if hidden % block:
            raise ValueError("FP8 source width must be divisible by its block size")
        cols = hidden // block
        # Source scales can be row-wise (MXFP8) or 2-D block-wise (DeepSeek FP8).
        if scale.numel() == experts * rows * cols:
            scale = scale.reshape(experts, rows, cols)
        elif scale.numel() == experts * ((rows + block - 1) // block) * cols:
            scale = scale.reshape(experts, (rows + block - 1) // block, cols)
            scale = scale.repeat_interleave(block, dim=1)[:, :rows]
        else:
            raise ValueError(
                f"Invalid FP8 scale shape {tuple(scale.shape)} for {tuple(weight.shape)}"
            )

        def dequantize(expert):
            sf = scale[expert].to(device=device, dtype=torch.float32)
            if self._scale_is_ue8m0_exponent:
                sf = torch.exp2(sf - 127.0)
            return (
                (weight[expert].float().reshape(rows, cols, block) * sf.unsqueeze(-1))
                .reshape(rows, hidden)
                .to(torch.bfloat16)
            )

        return self._output(
            weight.shape, weight.device, self.out_scale.name, dequantize
        )


class OnlineMegaMoeNvfp4Weight(_NativeOutput, OnlineMegaMoeFp4Weight):
    """Convert BF16 experts, including unquantized MTP layers, to native NVFP4."""

    def _load_raw_tensor(self, tensor_source, layer_id, device, load_config):
        weight = self.kernel._load_raw_tensor(
            tensor_source, layer_id, device, load_config
        )[self.kernel.name]
        return self._output(
            weight.shape, weight.device, self.scale.name, weight.__getitem__
        )
