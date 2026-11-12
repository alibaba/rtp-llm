import os

import torch
import triton
import triton.language as tl


def enabled() -> bool:
    return os.environ.get("RTP_LLM_GLM5_EPLB_FUSED_ROUTING", "1") == "1"


def maps_supported(layer, device) -> bool:
    if layer is None or not enabled():
        return False
    mapping, counts = layer.logical_to_physical, layer.logical_count
    return (
        mapping.device == counts.device == device
        and device.type == "cuda"
        and mapping.dtype == counts.dtype == torch.int32
        and mapping.is_contiguous()
        and counts.is_contiguous()
    )


@triton.jit
def map_experts(
    ids, rows, valid, mapping, counts, stride: tl.constexpr, rank, token_offset
):
    valid = valid & (ids >= 0)
    count = tl.load(counts + ids, mask=valid, other=1).to(tl.uint32)
    rotation = rank.to(tl.uint32) + (rows + token_offset).to(tl.uint32) * 23333
    replica = rotation % count
    physical = tl.load(
        mapping + ids.to(tl.int64) * stride + replica, mask=valid, other=-1
    )
    return physical.to(ids.dtype)
