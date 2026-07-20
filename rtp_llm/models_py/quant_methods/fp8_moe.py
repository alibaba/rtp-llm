"""FP8 MoE quant methods for the new loader.

This module owns the registered FP8 MoE loading paths. The per-tensor fusion
uses the runtime layout consumed by ``BaseMoEExperts`` and intentionally
uses max-scale rescaling for merged gate/up/down shards, so it is not a
byte-for-byte transform of each checkpoint shard. Forward execution
runs through BaseMoEExperts.forward and fused_moe after these buffers are built.
"""

import logging
from typing import Any, Dict, List

import torch
import torch.nn as nn
from rtp_llm.models_py.quant_methods.base import (
    FusedMoEMethodBase,
    register_moe_quant_method,
)
from rtp_llm.utils.model_weight import W

logger = logging.getLogger(__name__)


def _fp8_min_scale(fp8_max: float) -> float:
    return 1.0 / (fp8_max * 512.0)


def _runtime_fp8_dtype(device: torch.device) -> torch.dtype:
    from rtp_llm.models_py.quant_methods.fp8 import _runtime_fp8_dtype as resolve

    return resolve(device)


def _requant_per_tensor_to_runtime_fp8(
    weight: torch.Tensor, scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    from rtp_llm.models_py.quant_methods.fp8 import (
        _requant_per_tensor_to_runtime_fp8 as convert,
    )

    runtime_weight = torch.empty_like(weight, dtype=_runtime_fp8_dtype(weight.device))
    runtime_scale = torch.empty_like(scale, dtype=torch.float32)
    for expert_id in range(weight.shape[0]):
        converted, converted_scale = convert(weight[expert_id], scale[expert_id])
        runtime_weight[expert_id].copy_(converted)
        runtime_scale[expert_id] = converted_scale.reshape(())
    return runtime_weight, runtime_scale


def _requant_per_channel_to_runtime_fp8(
    weight: torch.Tensor, scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if scale.shape != weight.shape[:2]:
        raise ValueError(
            f"MoE FP8 per-channel scale/weight mismatch: "
            f"weight={tuple(weight.shape)} scale={tuple(scale.shape)}"
        )
    from rtp_llm.models_py.quant_methods.fp8 import (
        _requant_per_channel_to_runtime_fp8 as convert,
    )

    runtime_weight = torch.empty_like(weight, dtype=_runtime_fp8_dtype(weight.device))
    runtime_scale = torch.empty_like(scale, dtype=torch.float32)
    for expert_id in range(weight.shape[0]):
        converted, converted_scale = convert(weight[expert_id], scale[expert_id])
        runtime_weight[expert_id].copy_(converted)
        runtime_scale[expert_id].copy_(converted_scale.reshape(-1))
    return runtime_weight, runtime_scale


@register_moe_quant_method(
    # Prequantized per-tensor families.
    "fp8",
    "FP8_PER_TENSOR_COMPRESSED",
    "FP8_DYNAMIC_PER_TENSOR",
    # Prequantized per-block families.
    "FP8_PER_BLOCK",
    "fp8_block",
    # Prequantized per-channel families.
    "FP8_PER_CHANNEL_COMPRESSED",
    "fp8_per_channel",
    "FP8_PER_CHANNEL_QUARK",
    # Online BF16/FP16 to FP8 families.
    "fp8_online",
    "fp8_block_online",
    "fp8_per_channel_online",
)
class Fp8MoEMethod(FusedMoEMethodBase):
    def __init__(self, quant_config: Any = None):
        super().__init__(quant_config)
        quant_type = getattr(quant_config, "quant_type", "")
        self.requires_staged_device_postprocess = quant_type in (
            "fp8_online",
            "fp8_block_online",
            "fp8_per_channel_online",
        )

    def validate_runtime_device(self, device: torch.device) -> None:
        device = torch.device(device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError(
                f"FP8 MoE requires a supported accelerator, got {device}"
            )
        index = torch.cuda.current_device() if device.index is None else device.index
        properties = torch.cuda.get_device_properties(index)
        if getattr(torch.version, "hip", None) is not None:
            architecture = getattr(properties, "gcnArchName", "")
            if not any(name in architecture for name in ("gfx942", "gfx950")):
                raise RuntimeError(
                    f"FP8 MoE is not supported on ROCm architecture {architecture!r}"
                )
        elif torch.cuda.get_device_capability(index) < (8, 9):
            raise RuntimeError("FP8 MoE requires CUDA compute capability 8.9 or newer")
        _runtime_fp8_dtype(device)

    def required_aux_parameters(self):
        quant_type = getattr(self.quant_config, "quant_type", "")
        if quant_type in (
            "fp8",
            "FP8_PER_TENSOR_COMPRESSED",
            "FP8_DYNAMIC_PER_TENSOR",
            "FP8_PER_CHANNEL_COMPRESSED",
            "fp8_per_channel",
            "FP8_PER_CHANNEL_QUARK",
        ):
            return ("weight_scale",)
        if quant_type in ("FP8_PER_BLOCK", "fp8_block"):
            return ("weight_scale_inv",)
        return ()

    def dispatch_weight(self, layer, local_id, proj, param_name, tensor):
        if param_name != "weight":
            return False
        qf = layer._quant_family
        prequantized = qf in (
            "fp8_per_tensor",
            "fp8_per_channel",
            "fp8_per_block",
        )
        fp8_dtypes = {torch.float8_e4m3fn}
        fnuz = getattr(torch, "float8_e4m3fnuz", None)
        if fnuz is not None:
            fp8_dtypes.add(fnuz)
        if prequantized and tensor.dtype not in fp8_dtypes:
            raise TypeError(
                f"Prequantized FP8 MoE {proj}.weight requires FP8 storage, "
                f"got {tensor.dtype}"
            )
        if not prequantized and (
            not tensor.is_floating_point() or tensor.dtype in fp8_dtypes
        ):
            raise TypeError(
                f"Online FP8 MoE {proj}.weight requires an unquantized floating "
                f"tensor, got {tensor.dtype}"
            )
        return False

    @staticmethod
    def _ceil_div(x: int, y: int) -> int:
        return (x + y - 1) // y

    def _weight_block_size(self, layer) -> List[int]:
        block_size = getattr(
            getattr(layer, "_quant_config", None), "weight_block_size", None
        )
        if block_size is None:
            block_size = getattr(self.quant_config, "weight_block_size", None)
        if block_size is None:
            block_size = [128, 128]
        if not isinstance(block_size, (list, tuple)) or len(block_size) != 2:
            raise ValueError(f"weight_block_size must have 2 values, got {block_size}")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in block_size
        ):
            raise TypeError(
                f"weight_block_size values must be integers, got {block_size}"
            )
        block_n, block_k = block_size
        if block_n <= 0 or block_k <= 0:
            raise ValueError(
                f"weight_block_size values must be positive, got {block_size}"
            )
        return [block_n, block_k]

    def _requant_block_to_runtime_fp8(
        self, weight: torch.Tensor, scale: torch.Tensor, block_size: List[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        runtime_dtype = _runtime_fp8_dtype(weight.device)
        if weight.dtype == runtime_dtype:
            return weight.contiguous(), scale.contiguous()

        block_n, block_k = block_size
        expected = (
            weight.shape[0],
            self._ceil_div(weight.shape[1], block_n),
            self._ceil_div(weight.shape[2], block_k),
        )
        if tuple(scale.shape) != expected:
            raise ValueError(
                f"MoE FP8 block scale shape must be {expected}, got "
                f"{tuple(scale.shape)}"
            )
        if not bool(torch.isfinite(scale).all()) or bool((scale <= 0).any()):
            raise ValueError("MoE FP8 block scales must be finite and positive")
        if (
            weight.dtype == torch.float8_e4m3fn
            and runtime_dtype == torch.float8_e4m3fnuz
        ):
            from rtp_llm.models_py.quant_methods.fp8 import _convert_e4m3fn_to_fnuz

            return _convert_e4m3fn_to_fnuz(weight, scale)
        raise TypeError(
            f"Unsupported MoE FP8 conversion from {weight.dtype} to {runtime_dtype}"
        )

    # ------------------------------------------------------------------ #
    # Weight allocation.
    # ------------------------------------------------------------------ #
    def create_weights(
        self,
        layer,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype,
        **kwargs,
    ):
        E, M_tp, H = num_experts, intermediate_size, hidden_size
        qf = layer._quant_family
        dt = torch.float8_e4m3fn
        BS = layer._FP8_BLOCK_SIZE
        block_n, block_k = self._weight_block_size(layer)

        if qf == "fp8_per_tensor":
            layer.w13 = nn.Parameter(
                torch.empty(E, 2 * M_tp, H, dtype=dt), requires_grad=False
            )
            layer.w2 = nn.Parameter(
                torch.empty(E, H, M_tp, dtype=dt), requires_grad=False
            )
            layer.register_buffer("_gate_scales", torch.zeros(E, dtype=torch.float32))
            layer.register_buffer("_up_scales", torch.zeros(E, dtype=torch.float32))
            layer.register_buffer("_down_scales", torch.zeros(E, dtype=torch.float32))
            layer.register_buffer("w13_scale", torch.zeros(E, dtype=torch.float32))
            layer.register_buffer("w2_scale", torch.zeros(E, dtype=torch.float32))

        elif qf == "fp8_per_channel":
            layer.w13 = nn.Parameter(
                torch.empty(E, 2 * M_tp, H, dtype=dt), requires_grad=False
            )
            layer.w2 = nn.Parameter(
                torch.empty(E, H, M_tp, dtype=dt), requires_grad=False
            )
            layer.register_buffer(
                "_gate_ch_scales", torch.zeros(E, M_tp, dtype=torch.float32)
            )
            layer.register_buffer(
                "_up_ch_scales", torch.zeros(E, M_tp, dtype=torch.float32)
            )
            layer.register_buffer(
                "_down_ch_scales", torch.zeros(E, H, dtype=torch.float32)
            )
            layer.register_buffer(
                "w13_scale", torch.zeros(E, 2 * M_tp, dtype=torch.float32)
            )
            layer.register_buffer("w2_scale", torch.zeros(E, H, dtype=torch.float32))

        elif qf == "fp8_per_block":
            layer.w13 = nn.Parameter(
                torch.empty(E, 2 * M_tp, H, dtype=dt), requires_grad=False
            )
            layer.w2 = nn.Parameter(
                torch.empty(E, H, M_tp, dtype=dt), requires_grad=False
            )
            if layer.moe_expert_tp_size > 1 and (
                M_tp % block_n != 0 or M_tp % block_k != 0
            ):
                raise ValueError(
                    f"MoE FP8 block scale TP shard requires block-aligned "
                    f"moe_intermediate per rank, got moe_inter_tp={M_tp}, "
                    f"block_n={block_n}, block_k={block_k}, "
                    f"tp_size={layer.moe_expert_tp_size}"
                )
            nb = self._ceil_div(M_tp, block_n)
            hb = self._ceil_div(H, block_k)
            h_nb = self._ceil_div(H, block_n)
            k_nb = self._ceil_div(M_tp, block_k)
            layer._n_scale_blocks_per_proj = nb
            layer._k_scale_blocks_per_proj = k_nb
            layer._fp8_moe_weight_block_size = [block_n, block_k]
            if [block_n, block_k] == [BS, BS]:
                layer.register_buffer(
                    "w13_scale",
                    torch.zeros(E, 2 * nb, hb, dtype=torch.float32),
                )
                layer.register_buffer(
                    "w2_scale",
                    torch.zeros(E, h_nb, k_nb, dtype=torch.float32),
                )
            else:
                # Keep custom source scales on CPU and move one expert at a
                # time during conversion to bound accelerator peak memory.
                layer.w13_scale = torch.zeros(
                    E, 2 * nb, hb, dtype=torch.float32, device="cpu"
                )
                layer.w2_scale = torch.zeros(
                    E, h_nb, k_nb, dtype=torch.float32, device="cpu"
                )

        elif qf in (
            "fp8_per_tensor_online",
            "fp8_per_channel_online",
            "fp8_per_block_online",
        ):
            # Staging buffers keep the checkpoint dtype until post-load quantization.
            layer.w13 = nn.Parameter(
                torch.empty(E, 2 * M_tp, H, dtype=params_dtype), requires_grad=False
            )
            layer.w2 = nn.Parameter(
                torch.empty(E, H, M_tp, dtype=params_dtype), requires_grad=False
            )
            if qf == "fp8_per_tensor_online":
                layer.register_buffer("w13_scale", torch.zeros(E, dtype=torch.float32))
                layer.register_buffer("w2_scale", torch.zeros(E, dtype=torch.float32))
            elif qf == "fp8_per_channel_online":
                layer.register_buffer(
                    "w13_scale", torch.zeros(E, 2 * M_tp, dtype=torch.float32)
                )
                layer.register_buffer(
                    "w2_scale", torch.zeros(E, H, dtype=torch.float32)
                )
            else:  # fp8_per_block_online
                nb = self._ceil_div(M_tp, block_n)
                hb = self._ceil_div(H, block_k)
                h_nb = self._ceil_div(H, block_n)
                k_nb = self._ceil_div(M_tp, block_k)
                layer._n_scale_blocks_per_proj = nb
                layer._k_scale_blocks_per_proj = k_nb
                layer._fp8_moe_weight_block_size = [block_n, block_k]
                layer.register_buffer(
                    "w13_scale",
                    torch.zeros(E, 2 * nb, hb, dtype=torch.float32),
                )
                layer.register_buffer(
                    "w2_scale",
                    torch.zeros(E, h_nb, k_nb, dtype=torch.float32),
                )
        else:
            raise ValueError(f"Fp8MoEMethod does not support quant family {qf!r}")

    # ------------------------------------------------------------------ #
    # Streamed auxiliary-scale dispatch.
    # ------------------------------------------------------------------ #
    def dispatch_scale(self, layer, local_id, proj, param_name, tensor):
        qf = layer._quant_family

        # Online families consume floating-point weights and must not receive
        # prequantized checkpoint scales.
        if param_name in ("weight_scale", "weight_scale_inv") and qf in (
            "fp8_per_tensor_online",
            "fp8_per_channel_online",
            "fp8_per_block_online",
        ):
            raise RuntimeError(
                f"[Fp8MoEMethod] Got {param_name!r} for proj={proj!r} while in "
                f"online quant family {qf!r}; use the matching prequantized "
                "FP8 quantization method for this checkpoint"
            )

        if param_name in ("weight_scale", "weight_scale_inv"):
            if not tensor.is_floating_point():
                raise TypeError(
                    f"FP8 MoE {param_name} must be floating point, got "
                    f"{tensor.dtype}"
                )
            if not bool(torch.isfinite(tensor).all()) or bool((tensor <= 0).any()):
                raise ValueError(
                    f"FP8 MoE {param_name} values must be finite and positive"
                )

        if param_name == "weight_scale" and qf == "fp8_per_tensor":
            if tensor.numel() != 1:
                raise ValueError(
                    f"FP8 MoE per-tensor scale must contain one value, got "
                    f"shape {tuple(tensor.shape)}"
                )
            scale_val = tensor.float().reshape(()).item()
            if proj == "gate_proj":
                layer._gate_scales[local_id] = scale_val
            elif proj == "up_proj":
                layer._up_scales[local_id] = scale_val
            elif proj == "down_proj":
                layer._down_scales[local_id] = scale_val
            else:
                return False
            return True

        elif param_name == "weight_scale" and qf == "fp8_per_channel":
            self._copy_per_channel_scale(layer, local_id, proj, tensor)
            return True

        elif param_name == "weight_scale_inv" and qf == "fp8_per_block":
            self._copy_block_scale(layer, local_id, proj, tensor)
            return True
        return False

    def _copy_per_channel_scale(self, layer, expert_id, proj, tensor):
        # Checkpoints commonly store channel scales as [N], [N, 1], or [1, N].
        # Flattening preserves the valid single-channel case instead of turning
        # it into a scalar as squeeze() would.
        scale = tensor.float().reshape(-1)
        if proj == "gate_proj":
            if scale.dim() != 1 or scale.shape[0] != layer.moe_inter:
                raise ValueError(
                    f"gate_proj per-channel scale must have shape "
                    f"({layer.moe_inter},), got {tuple(scale.shape)}"
                )
            start = layer.moe_expert_tp_rank * layer.moe_inter_tp
            layer._gate_ch_scales.data[expert_id].copy_(
                scale.narrow(0, start, layer.moe_inter_tp)
            )
        elif proj == "up_proj":
            if scale.dim() != 1 or scale.shape[0] != layer.moe_inter:
                raise ValueError(
                    f"up_proj per-channel scale must have shape "
                    f"({layer.moe_inter},), got {tuple(scale.shape)}"
                )
            start = layer.moe_expert_tp_rank * layer.moe_inter_tp
            layer._up_ch_scales.data[expert_id].copy_(
                scale.narrow(0, start, layer.moe_inter_tp)
            )
        elif proj == "down_proj":
            if scale.dim() != 1 or scale.shape[0] != layer.hidden_size:
                raise ValueError(
                    f"down_proj per-channel scale must have shape "
                    f"({layer.hidden_size},), got {tuple(scale.shape)}"
                )
            layer._down_ch_scales.data[expert_id].copy_(scale)

    def _copy_block_scale(self, layer, expert_id, proj, tensor):
        block_n, block_k = getattr(
            layer, "_fp8_moe_weight_block_size", [layer._FP8_BLOCK_SIZE] * 2
        )
        nb = layer._n_scale_blocks_per_proj
        shard_start = layer.moe_expert_tp_rank * layer.moe_inter_tp
        if proj in ("gate_proj", "up_proj"):
            if layer.moe_expert_tp_size > 1:
                self._check_block_aligned(
                    shard_start,
                    layer.moe_inter_tp,
                    block_n,
                    f"MoE FP8 block scale {proj} TP shard",
                )
            start_block = shard_start // block_n
            expected = (
                self._ceil_div(layer.moe_inter, block_n),
                self._ceil_div(layer.hidden_size, block_k),
            )
            if tuple(tensor.shape) != expected:
                raise ValueError(
                    f"{proj} block scale must have shape {expected}, got "
                    f"{tuple(tensor.shape)}"
                )
            sliced = tensor.narrow(0, start_block, nb).contiguous()
            row_start = nb if proj == "gate_proj" else 0
            layer.w13_scale.data[expert_id, row_start : row_start + nb].copy_(sliced)
        elif proj == "down_proj":
            if layer.moe_expert_tp_size > 1:
                self._check_block_aligned(
                    shard_start,
                    layer.moe_inter_tp,
                    block_k,
                    "MoE FP8 block scale down_proj TP shard",
                )
            start_block = shard_start // block_k
            k_nb = getattr(layer, "_k_scale_blocks_per_proj", nb)
            expected = (
                self._ceil_div(layer.hidden_size, block_n),
                self._ceil_div(layer.moe_inter, block_k),
            )
            if tuple(tensor.shape) != expected:
                raise ValueError(
                    f"down_proj block scale must have shape {expected}, got "
                    f"{tuple(tensor.shape)}"
                )
            sliced = tensor.narrow(1, start_block, k_nb).contiguous()
            layer.w2_scale.data[expert_id].copy_(sliced)

    @staticmethod
    def _check_block_aligned(start: int, size: int, block: int, what: str):
        if start % block != 0 or size % block != 0:
            raise ValueError(
                f"{what} requires block-aligned shard boundaries, got "
                f"start={start}, size={size}, block={block}. Non-aligned "
                f"FP8 block scales would overlap checkpoint blocks and cannot "
                f"be copied safely without requantization."
            )

    # ------------------------------------------------------------------ #
    # Post-load scale fusion and online quantization.
    # ------------------------------------------------------------------ #
    def process_weights_after_loading(self, layer):
        qf = layer._quant_family
        if qf == "fp8_per_tensor":
            self._fuse_per_tensor(layer)
        elif qf == "fp8_per_channel":
            self._fuse_per_channel(layer)
        elif qf == "fp8_per_tensor_online":
            self._online_per_tensor(layer)
        elif qf == "fp8_per_channel_online":
            self._online_per_channel(layer)
        elif qf == "fp8_per_block_online":
            self._online_per_block(layer)
            self._requant_per_block_if_needed(layer)
        elif qf == "fp8_per_block":
            self._requant_per_block_if_needed(layer)

    def _fuse_per_tensor(self, layer):
        M_tp = layer.moe_inter_tp
        up_scales = layer._up_scales
        gate_scales = layer._gate_scales
        down_scales = layer._down_scales
        all_scales = torch.stack((up_scales, gate_scales, down_scales))
        scales_valid = torch.isfinite(all_scales).all() & (all_scales > 0).all()
        if not bool(scales_valid):
            raise ValueError("MoE FP8 per-tensor scales must be finite and positive")

        fused_scales = torch.maximum(up_scales, gate_scales)

        for e in range(layer.num_local_experts):
            up_ratio = up_scales[e] / fused_scales[e]
            gate_ratio = gate_scales[e] / fused_scales[e]
            up_half = layer.w13.data[e, :M_tp].to(torch.float16) * up_ratio
            gate_half = layer.w13.data[e, M_tp:].to(torch.float16) * gate_ratio
            layer.w13.data[e, :M_tp].copy_(up_half.to(layer.w13.dtype))
            layer.w13.data[e, M_tp:].copy_(gate_half.to(layer.w13.dtype))

        layer.w13_scale.copy_(fused_scales)
        layer.w2_scale.copy_(down_scales)

        layer.w13.data, layer.w13_scale.data = _requant_per_tensor_to_runtime_fp8(
            layer.w13.data, layer.w13_scale
        )
        layer.w2.data, layer.w2_scale.data = _requant_per_tensor_to_runtime_fp8(
            layer.w2.data, layer.w2_scale
        )

        del layer._gate_scales
        del layer._up_scales
        del layer._down_scales

    def _fuse_per_channel(self, layer):
        for e in range(layer.num_local_experts):
            layer.w13_scale.data[e, : layer.moe_inter_tp] = layer._up_ch_scales[e]
            layer.w13_scale.data[e, layer.moe_inter_tp :] = layer._gate_ch_scales[e]
            layer.w2_scale.data[e] = layer._down_ch_scales[e]
        layer.w13.data, layer.w13_scale.data = _requant_per_channel_to_runtime_fp8(
            layer.w13.data, layer.w13_scale
        )
        layer.w2.data, layer.w2_scale.data = _requant_per_channel_to_runtime_fp8(
            layer.w2.data, layer.w2_scale
        )
        del layer._gate_ch_scales
        del layer._up_ch_scales
        del layer._down_ch_scales

    def _online_per_tensor(self, layer):
        E = layer.num_local_experts
        device = layer.w13.data.device
        runtime_dtype = _runtime_fp8_dtype(device)
        new_w13 = torch.empty_like(layer.w13.data, dtype=runtime_dtype)
        new_w2 = torch.empty_like(layer.w2.data, dtype=runtime_dtype)
        from rtp_llm.models_py.quant_methods.fp8 import _resolve_per_tensor_quant

        quant = _resolve_per_tensor_quant()

        for e in range(E):
            qw, sc = quant(layer.w13.data[e].contiguous())
            new_w13[e].copy_(qw)
            layer.w13_scale[e] = sc.view(-1)[0]

            qw, sc = quant(layer.w2.data[e].contiguous())
            new_w2[e].copy_(qw)
            layer.w2_scale[e] = sc.view(-1)[0]

        layer.w13 = nn.Parameter(new_w13.to(device), requires_grad=False)
        layer.w2 = nn.Parameter(new_w2.to(device), requires_grad=False)

    def _online_per_channel(self, layer):
        E = layer.num_local_experts
        device = layer.w13.data.device
        runtime_dtype = _runtime_fp8_dtype(device)
        new_w13 = torch.empty_like(layer.w13.data, dtype=runtime_dtype)
        new_w2 = torch.empty_like(layer.w2.data, dtype=runtime_dtype)
        fp8_max = float(torch.finfo(runtime_dtype).max)
        min_scale = _fp8_min_scale(fp8_max)

        for e in range(E):
            w13_e = layer.w13.data[e].float()
            row_max = w13_e.abs().amax(dim=1)
            row_scale = (row_max / fp8_max).clamp_min(min_scale)
            new_w13[e] = (w13_e / row_scale.unsqueeze(1)).to(runtime_dtype)
            layer.w13_scale.data[e] = row_scale

            w2_e = layer.w2.data[e].float()
            row_max = w2_e.abs().amax(dim=1)
            row_scale = (row_max / fp8_max).clamp_min(min_scale)
            new_w2[e] = (w2_e / row_scale.unsqueeze(1)).to(runtime_dtype)
            layer.w2_scale.data[e] = row_scale

        layer.w13 = nn.Parameter(new_w13.to(device), requires_grad=False)
        layer.w2 = nn.Parameter(new_w2.to(device), requires_grad=False)

    def _online_per_block(self, layer):
        BS = layer._FP8_BLOCK_SIZE
        block_size = self._weight_block_size(layer)
        if block_size != [BS, BS]:
            raise ValueError(
                f"fp8_per_block_online MoE only supports [{BS}, {BS}] "
                f"runtime block size, got {block_size}"
            )
        from rtp_llm.models_py.quant_methods.fp8 import per_block_quant_like_legacy

        runtime_dtype = _runtime_fp8_dtype(layer.w13.device)
        new_w13 = torch.empty_like(layer.w13.data, dtype=runtime_dtype)
        new_w2 = torch.empty_like(layer.w2.data, dtype=runtime_dtype)
        for expert_id in range(layer.num_local_experts):
            weight, scale = per_block_quant_like_legacy(
                layer.w13.data[expert_id].contiguous(), BS
            )
            new_w13[expert_id].copy_(weight)
            layer.w13_scale[expert_id].copy_(scale)
            weight, scale = per_block_quant_like_legacy(
                layer.w2.data[expert_id].contiguous(), BS
            )
            new_w2[expert_id].copy_(weight)
            layer.w2_scale[expert_id].copy_(scale)

        layer.w13 = nn.Parameter(new_w13.contiguous(), requires_grad=False)
        layer.w2 = nn.Parameter(new_w2.contiguous(), requires_grad=False)

    def _requant_per_block_if_needed(self, layer):
        block_size = getattr(
            layer, "_fp8_moe_weight_block_size", [layer._FP8_BLOCK_SIZE] * 2
        )
        from rtp_llm.models_py.quant_methods.fp8 import (
            _resolve_requant_weight_ue8m0,
            is_deep_gemm_e8m0_used,
            per_block_quant_like_legacy,
        )

        if list(block_size) == [layer._FP8_BLOCK_SIZE, layer._FP8_BLOCK_SIZE]:
            runtime_dtype = _runtime_fp8_dtype(layer.w13.device)
            if layer.w13.dtype != runtime_dtype or layer.w2.dtype != runtime_dtype:
                layer.w13.data, layer.w13_scale = self._requant_block_to_runtime_fp8(
                    layer.w13.data.contiguous(),
                    layer.w13_scale.to(
                        layer.w13.device, non_blocking=True
                    ).contiguous(),
                    list(block_size),
                )
                layer.w2.data, layer.w2_scale = self._requant_block_to_runtime_fp8(
                    layer.w2.data.contiguous(),
                    layer.w2_scale.to(layer.w2.device, non_blocking=True).contiguous(),
                    list(block_size),
                )
                logger.info(
                    "[Fp8MoEMethod] requantized MoE fp8 block weights to runtime dtype: "
                    "dtype=%s w13=%s w13_scale=%s w2=%s w2_scale=%s",
                    runtime_dtype,
                    tuple(layer.w13.shape),
                    tuple(layer.w13_scale.shape),
                    tuple(layer.w2.shape),
                    tuple(layer.w2_scale.shape),
                )

            self._requant_block_scales_to_ue8m0_if_needed(
                layer, is_deep_gemm_e8m0_used, _resolve_requant_weight_ue8m0
            )
            return

        def requant_weight_fp8_block_float(weight, weight_scale):
            block_n, block_k = block_size
            expanded = weight_scale.float().repeat_interleave(block_n, dim=0)
            expanded = expanded.repeat_interleave(block_k, dim=1)
            weight_dequant = (
                weight.float() * expanded[: weight.shape[0], : weight.shape[1]]
            )
            return per_block_quant_like_legacy(
                weight_dequant.to(torch.bfloat16), layer._FP8_BLOCK_SIZE
            )

        src_w13_scale = layer.w13_scale
        src_w2_scale = layer.w2_scale
        if src_w13_scale.is_meta or src_w2_scale.is_meta:
            raise RuntimeError(
                "FP8 MoE custom block source scales must be materialized before "
                "post-load requantization"
            )
        new_w13_scale = []
        new_w2_scale = []
        device = layer.w13.device
        runtime_dtype = _runtime_fp8_dtype(device)
        new_w13 = torch.empty_like(layer.w13.data, dtype=runtime_dtype)
        new_w2 = torch.empty_like(layer.w2.data, dtype=runtime_dtype)
        for e in range(layer.num_local_experts):
            w, s = requant_weight_fp8_block_float(
                layer.w13.data[e].contiguous(),
                src_w13_scale[e].to(device, non_blocking=True).contiguous(),
            )
            if w.dtype != runtime_dtype:
                raise TypeError(
                    f"MoE FP8 block quantizer returned {w.dtype}, expected "
                    f"runtime dtype {runtime_dtype}"
                )
            new_w13[e].copy_(w)
            new_w13_scale.append(s)
            del w
            w, s = requant_weight_fp8_block_float(
                layer.w2.data[e].contiguous(),
                src_w2_scale[e].to(device, non_blocking=True).contiguous(),
            )
            if w.dtype != runtime_dtype:
                raise TypeError(
                    f"MoE FP8 block quantizer returned {w.dtype}, expected "
                    f"runtime dtype {runtime_dtype}"
                )
            new_w2[e].copy_(w)
            new_w2_scale.append(s)
            del w

        layer.w13 = nn.Parameter(new_w13.contiguous(), requires_grad=False)
        layer.w2 = nn.Parameter(new_w2.contiguous(), requires_grad=False)
        del layer.w13_scale
        del layer.w2_scale
        layer.register_buffer(
            "w13_scale", torch.stack(new_w13_scale, dim=0).contiguous()
        )
        layer.register_buffer("w2_scale", torch.stack(new_w2_scale, dim=0).contiguous())
        logger.info(
            "[Fp8MoEMethod] requantized MoE fp8 block scales from %s to [128, 128]: "
            "w13=%s w13_scale=%s w2=%s w2_scale=%s",
            block_size,
            tuple(layer.w13.shape),
            tuple(layer.w13_scale.shape),
            tuple(layer.w2.shape),
            tuple(layer.w2_scale.shape),
        )
        self._requant_block_scales_to_ue8m0_if_needed(
            layer, is_deep_gemm_e8m0_used, _resolve_requant_weight_ue8m0
        )

    @staticmethod
    def _requant_block_scales_to_ue8m0_if_needed(
        layer, is_ue8m0_used, resolve_requant
    ) -> None:
        if not is_ue8m0_used(layer.w13.device):
            return

        requant_weight_ue8m0 = resolve_requant()
        layer.w13.data, layer.w13_scale = requant_weight_ue8m0(
            layer.w13.data.contiguous(),
            layer.w13_scale.to(layer.w13.device, non_blocking=True).contiguous(),
        )
        layer.w2.data, layer.w2_scale = requant_weight_ue8m0(
            layer.w2.data.contiguous(),
            layer.w2_scale.to(layer.w2.device, non_blocking=True).contiguous(),
        )
        logger.info(
            "[Fp8MoEMethod] requantized MoE fp8 block scales to UE8M0: "
            "w13=%s w13_scale=%s/%s w2=%s w2_scale=%s/%s",
            tuple(layer.w13.shape),
            layer.w13_scale.dtype,
            tuple(layer.w13_scale.shape),
            tuple(layer.w2.shape),
            layer.w2_scale.dtype,
            tuple(layer.w2_scale.shape),
        )

    # ------------------------------------------------------------------ #
    # Add runtime scales to the fused-MoE weight view.
    # ------------------------------------------------------------------ #
    def add_weight_tensors(self, layer, weights_dict: Dict[str, Any]) -> None:
        weights_dict[W.moe_s1] = layer.w13_scale
        weights_dict[W.moe_s2] = layer.w2_scale
