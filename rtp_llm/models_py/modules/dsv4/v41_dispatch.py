"""Recipe-bound V4.1 EP payloads: E4M3, packed UE8M0, FP32 weights and IDs."""

from dataclasses import dataclass

import torch

from rtp_llm.models_py.modules.dsv4.v41_expert import (
    V41_EXPERT_RECIPE,
    V41ExpertRecipe,
    pack_group32_scales,
)
from rtp_llm.models_py.modules.dsv4.v41_mxfp8 import quantize_group32


def _validate_routes(weights, indices, valid_rows, experts):
    legal_weights = (
        ~valid_rows[:, None] | (torch.isfinite(weights) & (weights >= 0))
    ).all()
    active = valid_rows[:, None] & (weights != 0)
    legal_ids = (~active | ((indices >= 0) & (indices < experts))).all()
    ordered_ids = torch.where(active, indices.long(), experts).sort(dim=-1).values
    unique_ids = (
        ~((ordered_ids[:, 1:] == ordered_ids[:, :-1]) & (ordered_ids[:, 1:] < experts))
    ).all()
    legal = legal_weights & legal_ids & unique_ids
    if weights.device.type == "cpu":
        if not bool(legal):
            raise ValueError(
                "V4.1 dispatch requires finite nonnegative weights and unique in-range active IDs"
            )
    else:
        torch._assert_async(legal, "Invalid V4.1 dispatch route values")
    return active


@dataclass(frozen=True)
class V41DispatchLayout:
    hidden: int
    topk: int
    experts: int
    recipe: V41ExpertRecipe = V41_EXPERT_RECIPE

    def __post_init__(self):
        if (
            type(self.hidden) is not int
            or self.hidden <= 0
            or self.hidden % 32
            or type(self.topk) is not int
            or type(self.experts) is not int
            or not 0 < self.topk <= self.experts <= torch.iinfo(torch.int32).max
            or type(self.recipe) is not V41ExpertRecipe
            or self.recipe != V41_EXPERT_RECIPE
        ):
            raise ValueError("Invalid V4.1 dispatch geometry or numerical recipe")

    @property
    def packed_scales(self):
        return (self.hidden // 32 + 3) // 4

    @property
    def weight_offset(self):
        return self.hidden + self.packed_scales * 4

    @property
    def index_offset(self):
        return self.weight_offset + self.topk * 4

    @property
    def payload_bytes(self):
        return self.index_offset + self.topk * 4

    def views(self, payload):
        """Return strided x, packed-scale, weight and ID views without copies.

        Shape validation is not peer admission: workers must agree on this full
        immutable layout, including recipe and expert count, before collectives.
        """
        if (
            payload.ndim != 2
            or payload.shape[1] != self.payload_bytes
            or payload.dtype != torch.uint8
            or not payload.is_contiguous()
            or payload.device.type not in ("cpu", "cuda")
        ):
            raise ValueError(
                "V4.1 dispatch payload must be contiguous uint8 with exact row width"
            )
        return (
            payload[:, : self.hidden].view(torch.float8_e4m3fn),
            payload[:, self.hidden : self.weight_offset].view(torch.int32),
            payload[:, self.weight_offset : self.index_offset].view(torch.float32),
            payload[:, self.index_offset :].view(torch.int32),
        )

    def allocate(self, rows, *, device):
        if type(rows) is not int or rows < 0:
            raise ValueError("V4.1 dispatch row capacity must be nonnegative")
        return torch.empty((rows, self.payload_bytes), device=device, dtype=torch.uint8)

    def unpack(self, payload):
        """Copy received fields to expert-ready rows without numerical conversion.

        The caller must finish transport writes before use and retain the input
        through stream completion. Peer layout admission precedes this method;
        row geometry alone cannot establish the sender's numerical recipe.
        """
        values, packed, weights, indices = self.views(payload)
        valid_rows = torch.ones(
            payload.shape[0], dtype=torch.bool, device=payload.device
        )
        _validate_routes(weights, indices, valid_rows, self.experts)
        return (
            values.view(torch.uint8).contiguous().view(torch.float8_e4m3fn),
            packed.contiguous().view(torch.uint8)[:, : self.hidden // 32].contiguous(),
            weights.contiguous(),
            indices.contiguous(),
        )


def pack_dispatch(
    x, weights, indices, valid_rows, layout, *, out=None, backend="triton"
):
    """Canonicalize and quantize rank-local rows into an admitted EP payload.

    Active BF16 inputs must be finite. Inactive rows/routes may contain garbage:
    their payloads are zero, with IDs -1. Output capacity can exceed input rows.
    Scale packing is row-major dispatch ABI, never FlashInfer's swizzled layout.
    CUDA uses the same group-32 quantizer as expert compute, without fallback.
    """
    if type(layout) is not V41DispatchLayout:
        raise ValueError("V4.1 dispatch requires its immutable layout contract")
    if (
        x.ndim != 2
        or x.shape[1] != layout.hidden
        or x.dtype != torch.bfloat16
        or not x.is_contiguous()
        or x.device.type not in ("cpu", "cuda")
        or (torch.is_grad_enabled() and x.requires_grad)
    ):
        raise ValueError("V4.1 dispatch input requires contiguous inference BF16 [N,D]")
    rows = x.shape[0]
    if (
        weights.shape != (rows, layout.topk)
        or weights.dtype != torch.float32
        or weights.device != x.device
        or not weights.is_contiguous()
        or (torch.is_grad_enabled() and weights.requires_grad)
        or indices.shape != weights.shape
        or indices.dtype not in (torch.int32, torch.int64)
        or indices.device != x.device
        or not indices.is_contiguous()
        or valid_rows.shape != (rows,)
        or valid_rows.dtype != torch.bool
        or valid_rows.device != x.device
        or not valid_rows.is_contiguous()
    ):
        raise ValueError("V4.1 dispatch route metadata disagrees with its layout")
    if backend not in ("eager", "triton") or (backend == "eager") != (
        x.device.type == "cpu"
    ):
        raise ValueError("V4.1 dispatch requires explicit CPU eager or CUDA Triton")
    if out is not None:
        layout.views(out)
        if out.device != x.device or out.shape[0] < rows:
            raise ValueError(
                "V4.1 dispatch output device/capacity disagrees with input"
            )
        if out.numel() and any(
            t.untyped_storage().data_ptr() == out.untyped_storage().data_ptr()
            for t in (x, weights, indices, valid_rows)
        ):
            raise ValueError("V4.1 dispatch output must not share input storage")
    active = _validate_routes(weights, indices, valid_rows, layout.experts)
    active_rows = active.any(-1)
    safe_x = torch.where(active_rows[:, None], x, 0)
    values, scales = quantize_group32(safe_x, backend=backend)
    scales = torch.where(active_rows[:, None], scales, 0)
    packed = pack_group32_scales(scales)
    result = layout.allocate(rows, device=x.device) if out is None else out
    result.zero_()
    q_out, scale_out, weight_out, ids_out = layout.views(result)
    ids_out.fill_(-1)
    q_out[:rows].view(torch.uint8).copy_(values.view(torch.uint8))
    scale_out[:rows].copy_(packed)
    weight_out[:rows].copy_(torch.where(active, weights, 0))
    ids_out[:rows].copy_(torch.where(active, indices, -1).to(torch.int32))
    return result
