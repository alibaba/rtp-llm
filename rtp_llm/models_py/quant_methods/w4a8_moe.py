from typing import Any, Dict

import torch
import torch.nn as nn
from rtp_llm.models_py.quant_methods.base import (
    FusedMoEMethodBase,
    register_moe_quant_method,
)
from rtp_llm.models_py.quant_methods.w4a8_utils import (
    quantize_weight_to_int4b,
    repack_compressed_int4_to_cutlass,
)
from rtp_llm.utils.model_weight import W


def _group_size(source_config: Any, default: int = 128) -> int:
    if source_config is None:
        return default
    group_size = getattr(source_config, "group_size", None)
    if callable(group_size):
        group_size = group_size()
    if group_size in (-1, 0, None):
        return default
    if isinstance(group_size, bool) or not isinstance(group_size, int):
        raise TypeError(f"W4A8 INT4 group_size must be an integer, got {group_size!r}")
    if group_size <= 0:
        raise ValueError(f"W4A8 INT4 group_size must be positive, got {group_size}")
    return group_size


@register_moe_quant_method(
    "W4A8_INT4_PER_CHANNEL",
    "W4A8_INT4_PER_CHANNEL_COMPRESSED",
)
class W4A8Int4MoEMethod(FusedMoEMethodBase):
    """W4A8 INT4 MoE loading and runtime layout conversion.

    - W4A8_INT4_PER_CHANNEL: load BF16/FP16 ``*.weight`` tensors, then call
      ``quantize_weight_to_int4b`` per expert.
    - W4A8_INT4_PER_CHANNEL_COMPRESSED: load pre-packed
      ``*.weight_packed`` + ``*.weight_scale`` tensors, then
      compressed-tensors repack to cutlass layout.
    """

    def __init__(self, quant_config: Any = None):
        super().__init__(quant_config)
        self.source_config = getattr(quant_config, "source_config", None)
        self.quant_type = getattr(quant_config, "quant_type", "")
        self.group_size = _group_size(self.source_config)
        self.compressed = self.quant_type == "W4A8_INT4_PER_CHANNEL_COMPRESSED"
        # Both paths invoke CUDA-only conversion helpers. Stage one expert
        # module at a time so pending checkpoint buffers do not all migrate to
        # the GPU before they are repacked and released.
        self.requires_staged_device_postprocess = True

    def validate_runtime_device(self, device: torch.device) -> None:
        device = torch.device(device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError(f"W4A8 MoE requires a CUDA accelerator, got {device}")
        if getattr(torch.version, "hip", None) is not None:
            raise RuntimeError("W4A8 MoE is not supported by the ROCm runtime")
        index = torch.cuda.current_device() if device.index is None else device.index
        if torch.cuda.get_device_capability(index) < (8, 9):
            raise RuntimeError("W4A8 MoE requires CUDA compute capability 8.9 or newer")
        try:
            from rtp_kernel.w4a8_group_gemm import (
                pack_scale_fp8,
                reorder_tensor,
                unified_encode_int4b,
            )
        except (ImportError, OSError, RuntimeError, AttributeError) as exc:
            raise RuntimeError("W4A8 MoE requires the rtp-kernel W4A8 runtime") from exc
        if not all(
            callable(value)
            for value in (pack_scale_fp8, reorder_tensor, unified_encode_int4b)
        ):
            raise RuntimeError("rtp-kernel W4A8 runtime is missing required symbols")

    def required_aux_parameters(self):
        return ("weight_scale",) if self.compressed else ()

    def create_weights(
        self,
        layer,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype,
        **kwargs,
    ):
        E = num_experts
        H = hidden_size
        M = intermediate_size
        gs = self.group_size

        if H % 2 != 0 or M % 2 != 0:
            raise ValueError(f"W4A8 INT4 expects even H/M, got H={H}, M={M}")
        if H % gs != 0 or M % gs != 0:
            raise ValueError(f"W4A8 INT4 group_size={gs} must divide H={H} and M={M}")

        if self.compressed:
            layer.w13 = nn.Parameter(
                torch.empty(E, 2 * M, H // 2, dtype=torch.int8),
                requires_grad=False,
            )
            layer.w2 = nn.Parameter(
                torch.empty(E, H, M // 2, dtype=torch.int8),
                requires_grad=False,
            )
            layer.register_buffer(
                "w13_scale",
                torch.empty(E, H // gs, 2 * M, 8, dtype=torch.float8_e4m3fn),
            )
            layer.register_buffer(
                "w2_scale",
                torch.empty(E, M // gs, H, 8, dtype=torch.float8_e4m3fn),
            )
        else:
            layer.w13 = nn.Parameter(
                torch.empty(E, 2 * M, H, dtype=params_dtype),
                requires_grad=False,
            )
            layer.w2 = nn.Parameter(
                torch.empty(E, H, M, dtype=params_dtype),
                requires_grad=False,
            )

    def dispatch_weight(self, layer, local_id: int, proj: str, param_name: str, tensor):
        if not self.compressed and param_name == "weight":
            if not tensor.is_floating_point():
                raise TypeError(
                    f"W4A8 online quantization requires floating {proj}.weight, "
                    f"got {tensor.dtype}"
                )
            return False
        if not self.compressed or param_name != "weight_packed":
            return False
        if tensor.dtype not in (torch.int8, torch.uint8, torch.int16, torch.int32):
            raise TypeError(
                f"W4A8 compressed weights require integer storage, got {tensor.dtype}"
            )
        if proj in ("gate_proj", "up_proj"):
            packed, scale = self._pending_gate_up(layer, local_id, proj)
            packed.copy_(self._slice_gate_up_packed(layer, tensor))
        elif proj == "down_proj":
            packed, scale = self._pending_down(layer, local_id)
            packed.copy_(self._slice_down_packed(layer, tensor))
        else:
            return False
        return True

    def dispatch_scale(self, layer, local_id: int, proj: str, param_name: str, tensor):
        if not self.compressed or param_name != "weight_scale":
            return False
        if not tensor.is_floating_point():
            raise TypeError(
                f"W4A8 compressed scales must be floating point, got {tensor.dtype}"
            )
        if not bool(torch.isfinite(tensor).all()) or bool((tensor <= 0).any()):
            raise ValueError("W4A8 compressed scales must be finite and positive")
        if proj in ("gate_proj", "up_proj"):
            packed, scale = self._pending_gate_up(layer, local_id, proj)
            scale.copy_(self._slice_gate_up_scale(layer, tensor))
        elif proj == "down_proj":
            packed, scale = self._pending_down(layer, local_id)
            scale.copy_(self._slice_down_scale(layer, tensor))
        else:
            return False
        return True

    def process_weights_after_loading(self, layer):
        if self.compressed:
            self._repack_compressed(layer)
        else:
            self._online_quantize(layer)

    def add_weight_tensors(self, layer, weights_dict: Dict[str, Any]) -> None:
        weights_dict[W.moe_s1] = layer.w13_scale
        weights_dict[W.moe_s2] = layer.w2_scale

    def _online_quantize(self, layer):
        E = layer.num_local_experts
        gs = self.group_size
        device = layer.w13.data.device
        w13 = layer.w13.data
        w2 = layer.w2.data
        n1, k1 = w13.shape[1], w13.shape[2]
        n2, k2 = w2.shape[1], w2.shape[2]

        q13 = torch.empty(E, n1, k1 // 2, device=device, dtype=torch.int8)
        s13 = torch.empty(E, k1 // gs, n1, 8, device=device, dtype=torch.float8_e4m3fn)
        q2 = torch.empty(E, n2, k2 // 2, device=device, dtype=torch.int8)
        s2 = torch.empty(E, k2 // gs, n2, 8, device=device, dtype=torch.float8_e4m3fn)

        for e in range(E):
            q13[e], s13[e] = quantize_weight_to_int4b(w13[e], gs)
            q2[e], s2[e] = quantize_weight_to_int4b(w2[e], gs)

        layer.w13 = nn.Parameter(q13, requires_grad=False)
        layer.w2 = nn.Parameter(q2, requires_grad=False)
        layer.register_buffer("w13_scale", s13)
        layer.register_buffer("w2_scale", s2)

    def _repack_compressed(self, layer):
        M = layer.moe_inter_tp
        for e in range(layer.num_local_experts):
            up_w, up_s = self._pending_gate_up(layer, e, "up_proj")
            gate_w, gate_s = self._pending_gate_up(layer, e, "gate_proj")
            down_w, down_s = self._pending_down(layer, e)

            qw, qs = repack_compressed_int4_to_cutlass(up_w, up_s, self.group_size)
            layer.w13.data[e, :M].copy_(qw)
            layer.w13_scale.data[e, :, :M].copy_(qs)

            qw, qs = repack_compressed_int4_to_cutlass(gate_w, gate_s, self.group_size)
            layer.w13.data[e, M : 2 * M].copy_(qw)
            layer.w13_scale.data[e, :, M : 2 * M].copy_(qs)

            qw, qs = repack_compressed_int4_to_cutlass(down_w, down_s, self.group_size)
            layer.w2.data[e].copy_(qw)
            layer.w2_scale.data[e].copy_(qs)

        del layer._w4a8_gate_packed
        del layer._w4a8_up_packed
        del layer._w4a8_down_packed
        del layer._w4a8_gate_scale
        del layer._w4a8_up_scale
        del layer._w4a8_down_scale

    def _ensure_pending(self, layer):
        if hasattr(layer, "_w4a8_gate_packed"):
            return
        E = layer.num_local_experts
        M = layer.moe_inter_tp
        H = layer.hidden_size
        gs = self.group_size
        dev = layer.w13.device
        layer.register_buffer(
            "_w4a8_gate_packed", torch.empty(E, M, H // 2, dtype=torch.int8, device=dev)
        )
        layer.register_buffer(
            "_w4a8_up_packed", torch.empty(E, M, H // 2, dtype=torch.int8, device=dev)
        )
        layer.register_buffer(
            "_w4a8_down_packed", torch.empty(E, H, M // 2, dtype=torch.int8, device=dev)
        )
        layer.register_buffer(
            "_w4a8_gate_scale",
            torch.empty(E, M, H // gs, dtype=torch.bfloat16, device=dev),
        )
        layer.register_buffer(
            "_w4a8_up_scale",
            torch.empty(E, M, H // gs, dtype=torch.bfloat16, device=dev),
        )
        layer.register_buffer(
            "_w4a8_down_scale",
            torch.empty(E, H, M // gs, dtype=torch.bfloat16, device=dev),
        )

    def _pending_gate_up(self, layer, local_id: int, proj: str):
        self._ensure_pending(layer)
        if proj == "gate_proj":
            return layer._w4a8_gate_packed[local_id], layer._w4a8_gate_scale[local_id]
        return layer._w4a8_up_packed[local_id], layer._w4a8_up_scale[local_id]

    def _pending_down(self, layer, local_id: int):
        self._ensure_pending(layer)
        return layer._w4a8_down_packed[local_id], layer._w4a8_down_scale[local_id]

    def _slice_gate_up_packed(self, layer, tensor: torch.Tensor) -> torch.Tensor:
        rows = layer.moe_inter_tp
        full_rows = rows * layer.moe_expert_tp_size
        if tensor.shape[0] != full_rows:
            raise ValueError(
                f"W4A8 gate/up packed rows {tensor.shape[0]} != expected {full_rows}"
            )
        start = layer.moe_expert_tp_rank * rows
        return tensor.narrow(0, start, rows).contiguous().view(torch.int8)

    def _slice_gate_up_scale(self, layer, tensor: torch.Tensor) -> torch.Tensor:
        rows = layer.moe_inter_tp
        full_rows = rows * layer.moe_expert_tp_size
        if tensor.shape[0] != full_rows:
            raise ValueError(
                f"W4A8 gate/up scale rows {tensor.shape[0]} != expected {full_rows}"
            )
        start = layer.moe_expert_tp_rank * rows
        return tensor.narrow(0, start, rows).contiguous()

    def _slice_down_packed(self, layer, tensor: torch.Tensor) -> torch.Tensor:
        cols = layer.moe_inter_tp
        full_cols = cols * layer.moe_expert_tp_size
        packed_cols = cols // 2
        packed = tensor.contiguous().view(torch.int8)
        if packed.shape[1] != full_cols // 2:
            raise ValueError(
                f"W4A8 down packed cols {packed.shape[1]} != expected {full_cols // 2}"
            )
        start = layer.moe_expert_tp_rank * packed_cols
        return packed.narrow(1, start, packed_cols).contiguous()

    def _slice_down_scale(self, layer, tensor: torch.Tensor) -> torch.Tensor:
        cols = layer.moe_inter_tp
        full_cols = cols * layer.moe_expert_tp_size
        groups = cols // self.group_size
        if tensor.shape[1] != full_cols // self.group_size:
            raise ValueError(
                f"W4A8 down scale cols {tensor.shape[1]} "
                f"!= expected {full_cols // self.group_size}"
            )
        start = layer.moe_expert_tp_rank * groups
        return tensor.narrow(1, start, groups).contiguous()
