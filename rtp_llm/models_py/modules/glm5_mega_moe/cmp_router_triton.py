import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from .eplb_routing import map_experts, maps_supported


@triton.jit(do_not_specialize=["rank"])
def _cmp_router(
    logits,
    bias,
    out_ids,
    out_weights,
    mapping,
    counts,
    stride: tl.constexpr,
    rank,
    PDL: tl.constexpr,
):
    if PDL:
        tl.inline_asm_elementwise(
            "griddepcontrol.wait; mov.u32 $0, 0;",
            constraints="=r",
            args=[],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )
    row = tl.program_id(0).to(tl.int64)
    experts = tl.arange(0, 256)
    value = tl.load(logits + row * 256 + experts).to(tl.bfloat16).to(tl.float32)
    score = libdevice.fast_dividef(1.0, 1.0 + libdevice.fast_expf(-value))
    corrected = tl.where(
        tl.abs(score) < float("inf"), score + tl.load(bias + experts), -float("inf")
    )
    tl.store(logits + row * 256 + experts, 0.0)
    slots = tl.arange(0, 8)
    selected_scores = tl.full((8,), 0, tl.float32)
    selected_ids = tl.full((8,), 0, tl.int32)
    available = tl.full((256,), True, tl.int1)
    for k in tl.static_range(8):
        best = tl.max(corrected, 0)
        winner = tl.min(
            tl.where((corrected == best) & available, experts, 2147483647), 0
        )
        selected_ids = tl.where(slots == k, winner, selected_ids)
        selected_scores = tl.where(
            slots == k,
            tl.sum(tl.where(experts == winner, score, 0), 0),
            selected_scores,
        )
        corrected = tl.where(experts == winner, -float("inf"), corrected)
        available = available & (experts != winner)
    partial = tl.sum(selected_scores.reshape(2, 2, 2), 0)
    total = tl.sum(tl.sum(partial, 0), 0) + 1.0e-20
    physical = map_experts(
        selected_ids, row, slots < 8, mapping, counts, stride, rank, 0
    )
    tl.store(out_ids + row * 8 + slots, physical.to(tl.int64))
    tl.store(
        out_weights + row * 8 + slots,
        libdevice.fast_dividef(selected_scores, total) * 2.5,
    )
    if PDL:
        tl.debug_barrier()
        tl.inline_asm_elementwise(
            "griddepcontrol.launch_dependents; mov.u32 $0, 0;",
            constraints="=r",
            args=[],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )


def is_supported(logits, bias, ids, weights, layer) -> bool:
    return (
        maps_supported(layer, logits.device)
        and logits.dtype == bias.dtype == weights.dtype == torch.float32
        and ids.dtype == torch.int64
        and logits.ndim == 2
        and logits.shape[1] == 256
        and bias.shape == (256,)
        and layer.logical_count.numel() == 256
        and ids.shape == weights.shape == (logits.shape[0], 8)
        and all(
            t.is_contiguous() and t.device == logits.device
            for t in (logits, bias, ids, weights)
        )
    )


def router_topk(logits, bias, ids, weights, layer, *, pdl=False):
    if not is_supported(logits, bias, ids, weights, layer):
        raise ValueError("unsupported GLM CMP fused EPLB routing tensors")
    if logits.shape[0] == 0:
        return
    _cmp_router[(logits.shape[0],)](
        logits,
        bias,
        ids,
        weights,
        layer.logical_to_physical,
        layer.logical_count,
        layer.logical_to_physical.stride(0),
        layer.replica_offset,
        PDL=pdl,
        launch_pdl=pdl,
        num_warps=4,
        enable_fp_fusion=False,
        enable_reflect_ftz=True,
    )
