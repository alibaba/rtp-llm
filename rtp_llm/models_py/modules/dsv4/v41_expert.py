"""V4.1 expert arithmetic, explicit scale layouts and the dense shared expert."""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from rtp_llm.models_py.modules.dsv4.v41_mxfp8 import (
    AMAX_FLOOR,
    GROUP_SIZE,
    V41MXFP8Linear,
    _scale_bytes,
    quantize_group32,
)


@dataclass(frozen=True)
class V41ExpertRecipe:
    """Numerical identity for future dispatch/workspace admission keys."""

    activation_group: int = GROUP_SIZE
    amax_floor: float = AMAX_FLOOR
    scale_rounding: str = "fp32_reciprocal_bitwise_ceil_pow2"
    scale_format: str = "ue8m0"
    gate_up_order: str = "gate_then_up"
    router_weight_position: str = "before_down_input_bf16"
    intermediate_dtype: str = "bfloat16"
    routed_sum_dtype: str = "float32"

    def __post_init__(self):
        if (
            self.activation_group,
            self.amax_floor,
            self.scale_rounding,
            self.scale_format,
            self.gate_up_order,
            self.router_weight_position,
            self.intermediate_dtype,
            self.routed_sum_dtype,
        ) != (
            32,
            1e-4,
            "fp32_reciprocal_bitwise_ceil_pow2",
            "ue8m0",
            "gate_then_up",
            "before_down_input_bf16",
            "bfloat16",
            "float32",
        ):
            raise ValueError("Unsupported V4.1 expert numerical recipe")


V41_EXPERT_RECIPE = V41ExpertRecipe()


def _check_limit(clamp_limit):
    if (
        type(clamp_limit) not in (int, float)
        or not math.isfinite(clamp_limit)
        or clamp_limit < 0
    ):
        raise ValueError("V4.1 SwiGLU clamp must be finite and nonnegative")


def _check_rows(shape, device, valid_rows, route_weights):
    if (
        valid_rows.shape != shape
        or valid_rows.dtype != torch.bool
        or valid_rows.device != device
        or not valid_rows.is_contiguous()
    ):
        raise ValueError("V4.1 expert requires a contiguous device bool row mask")
    if route_weights is not None and (
        route_weights.shape != shape
        or route_weights.dtype != torch.float32
        or route_weights.device != device
        or not route_weights.is_contiguous()
        or (torch.is_grad_enabled() and route_weights.requires_grad)
    ):
        raise ValueError(
            "V4.1 route weights require contiguous inference FP32 per-row values"
        )
    if route_weights is None:
        return valid_rows
    if device.type == "cpu" and not bool(
        (
            torch.isfinite(route_weights[valid_rows]) & (route_weights[valid_rows] >= 0)
        ).all()
    ):
        raise ValueError("V4.1 active route weights must be finite and nonnegative")
    return valid_rows & (route_weights != 0)


def swiglu_bf16(gate, up, valid_rows, *, route_weights=None, clamp_limit=10.0):
    """Checkpoint FP32 activation/weight order, followed by one BF16 rounding.

    Inputs are BF16 GEMM outputs, optionally strided along rows. Dead/zero-route
    rows are zeroed before arithmetic. Active inputs/weights must be finite and
    route weights nonnegative. This unfused baseline uses PyTorch on CPU/CUDA.
    """
    _check_limit(clamp_limit)
    if (
        gate.ndim < 2
        or gate.shape[-1] <= 0
        or gate.shape[-1] % GROUP_SIZE
        or up.shape != gate.shape
        or gate.dtype != torch.bfloat16
        or up.dtype != gate.dtype
        or up.device != gate.device
        or gate.device.type not in ("cpu", "cuda")
        or gate.stride(-1) != 1
        or up.stride(-1) != 1
    ):
        raise ValueError(
            "V4.1 SwiGLU requires matching BF16 [...,I] with contiguous group-32 channels"
        )
    if torch.is_grad_enabled() and (gate.requires_grad or up.requires_grad):
        raise ValueError("V4.1 expert arithmetic is inference-only")
    active = _check_rows(gate.shape[:-1], gate.device, valid_rows, route_weights)
    g = torch.where(active[..., None], gate.float(), 0.0)
    u = torch.where(active[..., None], up.float(), 0.0)
    if gate.device.type == "cpu" and not (
        bool(torch.isfinite(g).all()) and bool(torch.isfinite(u).all())
    ):
        raise ValueError("V4.1 active gate/up values must be finite")
    if clamp_limit:
        g = g.clamp(max=clamp_limit)
        u = u.clamp(-clamp_limit, clamp_limit)
    result = F.silu(g) * u
    if route_weights is not None:
        weights = torch.where(active, route_weights, 0.0)
        result = weights[..., None] * result
    return result.to(torch.bfloat16)


def quantize_swiglu(
    gate, up, valid_rows, *, route_weights=None, clamp_limit=10.0, backend="triton"
):
    """Return group-32 E4M3/linear UE8M0; unused route scales are zero."""
    if backend not in ("eager", "triton") or (backend == "eager") != (
        gate.device.type == "cpu"
    ):
        raise ValueError(
            "V4.1 SwiGLU quantization requires explicit CPU eager or CUDA Triton"
        )
    intermediate = swiglu_bf16(
        gate, up, valid_rows, route_weights=route_weights, clamp_limit=clamp_limit
    )
    values, scales = quantize_group32(intermediate, backend=backend)
    active = valid_rows if route_weights is None else valid_rows & (route_weights != 0)
    scales = torch.where(active.reshape(-1, 1), scales, 0)
    return values, scales


def pack_group32_scales(scales, *, column_major=False):
    """Pack linear UE8M0 [...,M,G] into int32 [...,M,ceil(G/4)].

    Row-major is the dispatch ABI. Column-major pads M to four (minimum four,
    including empty batches) for DeepGEMM TMA addressing. All unused scale
    bytes/rows are initialized to zero.
    FlashInfer's 128x4 swizzle is a separate ABI and must not use this helper.
    """
    raw = _scale_bytes(scales)
    if raw.ndim < 2 or raw.shape[-1] <= 0 or not raw.is_contiguous():
        raise ValueError(
            "V4.1 scale packing requires contiguous linear [...,M,G] UE8M0"
        )
    if type(column_major) is not bool:
        raise ValueError("V4.1 packed scale layout must be selected explicitly")
    rows, groups = raw.shape[-2:]
    packs = (groups + 3) // 4
    padded = torch.zeros(
        (*raw.shape[:-1], packs * 4), dtype=torch.uint8, device=raw.device
    )
    padded[..., :groups].copy_(raw)
    packed = padded.view(torch.int32)
    if not column_major:
        return packed
    storage = torch.zeros(
        (*raw.shape[:-2], packs, max(4, (rows + 3) // 4 * 4)),
        dtype=torch.int32,
        device=raw.device,
    )
    result = storage.transpose(-1, -2)[..., :rows, :]
    result.copy_(packed)
    return result


class V41SharedExpert(nn.Module):
    """Loader-merged [gate,up] group-32 FP8 weights with explicit BF16 semantics."""

    recipe = V41_EXPERT_RECIPE

    def __init__(self, expert_weights, *, swiglu_limit=10.0, backend="flashinfer"):
        super().__init__()
        _check_limit(swiglu_limit)
        w13, w2 = expert_weights["w13_w"], expert_weights["w2_w"]
        if (
            w13.ndim != 2
            or w2.ndim != 2
            or w13.shape[0] != 2 * w2.shape[1]
            or w13.shape[1] != w2.shape[0]
            or w2.shape[1] % GROUP_SIZE
        ):
            raise ValueError(
                "V4.1 shared gate/up and down projection geometry disagree"
            )
        self.dim, self.inter_dim = w2.shape
        self.w13 = V41MXFP8Linear(w13, expert_weights["w13_s"], backend=backend)
        self.w2 = V41MXFP8Linear(w2, expert_weights["w2_s"], backend=backend)
        if self.w13.weight.device != self.w2.weight.device:
            raise ValueError("V4.1 shared projections must share one device")
        self.swiglu_limit = swiglu_limit
        self.backend = "eager" if backend == "eager" else "triton"

    def forward(self, x, valid_rows, *, route_weights=None, out=None):
        if (
            x.ndim < 2
            or x.shape[-1] != self.dim
            or x.dtype != torch.bfloat16
            or not x.is_contiguous()
        ):
            raise ValueError("V4.1 shared input requires contiguous BF16 [...,D]")
        if torch.is_grad_enabled() and x.requires_grad:
            raise ValueError("V4.1 shared expert is inference-only")
        active = _check_rows(x.shape[:-1], x.device, valid_rows, route_weights)
        if x.device != self.w13.weight.device:
            raise ValueError("V4.1 shared input must be on the weight device")
        if out is not None:
            if (
                out.shape != x.shape
                or out.dtype != x.dtype
                or out.device != x.device
                or not out.is_contiguous()
                or out.requires_grad
            ):
                raise ValueError(
                    "V4.1 shared output must match contiguous BF16 input metadata"
                )
            protected = (
                x,
                valid_rows,
                route_weights,
                self.w13.weight,
                self.w13.scales,
                self.w13._packed_scales,
                self.w2.weight,
                self.w2.scales,
                self.w2._packed_scales,
            )
            if out.numel() and any(
                t is not None
                and out.untyped_storage().data_ptr() == t.untyped_storage().data_ptr()
                for t in protected
            ):
                raise ValueError(
                    "V4.1 shared output must not share input or weight storage"
                )
        safe_x = torch.where(active[..., None], x, 0.0)
        gate_up = self.w13(safe_x)
        gate, up = gate_up.split(self.inter_dim, dim=-1)
        values, scales = quantize_swiglu(
            gate,
            up,
            valid_rows,
            route_weights=route_weights,
            clamp_limit=self.swiglu_limit,
            backend=self.backend,
        )
        result = self.w2.forward_quantized(values, scales)
        result = torch.where(active[..., None], result, 0.0)
        return result if out is None else out.copy_(result)
