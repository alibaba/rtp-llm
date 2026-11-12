"""DeepGEMM MegaGate adapter shared by GLM prefill and CMP decode.

The native epilogue computes sigmoid, corrected TopK, normalization and EPLB
mapping without materializing logits. Map/count and optional output buffers are
borrowed, so graph replay observes in-place placement updates.
"""

import os
from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def _native_gate():
    try:
        import deep_gemm
    except ImportError:
        return None
    return getattr(deep_gemm, "bf16_mega_gate", None)


def enabled(eplb) -> bool:
    mode = os.environ.get("RTP_LLM_GLM5_MEGA_GATE", "auto")
    return mode == "1" or (mode == "auto" and eplb is not None)


def is_supported(x, weight, bias, config, eplb) -> bool:
    from .eplb_routing import maps_supported

    if not enabled(eplb):
        return False
    if weight is None or bias is None:
        return False
    if eplb is not None and not maps_supported(eplb, x.device):
        return False
    return (
        x.ndim == 2
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
        and 0 < x.shape[0] <= (1 << 20)
        and weight.ndim == 2
        and x.shape[1] == weight.shape[1]
        and weight.shape[1] % 256 == 0
        and 0 < weight.shape[0] <= 512
        and weight.shape[0] % 4 == 0
        and weight.dtype == torch.bfloat16
        and weight.is_contiguous()
        and bias.dtype == torch.float32
        and bias.shape == (weight.shape[0],)
        and bias.is_contiguous()
        and weight.device == bias.device == x.device
        and int(config.moe_n_group) == int(config.moe_topk_group) == 1
        and bool(config.has_moe_norm)
        and 0 < int(config.moe_k) <= min(weight.shape[0], 32)
        and torch.cuda.get_device_capability(x.device)[0] == 10
        and _native_gate() is not None
    )


def mega_gate(x, weight, bias, config, eplb, *, out=None):
    kernel = _native_gate()
    if kernel is None:
        raise RuntimeError("DeepGEMM does not provide bf16_mega_gate")
    if out is None:
        shape = (x.shape[0], int(config.moe_k))
        out = (
            torch.empty(shape, device=x.device, dtype=torch.int64),
            torch.empty(shape, device=x.device, dtype=torch.float32),
        )
    kernel(
        x,
        weight,
        int(config.moe_k),
        use_shared_as_routed=False,
        num_shared_experts=0,
        routed_scaling_factor=float(config.routed_scaling_factor),
        ep_rank=eplb.replica_offset if eplb is not None else 0,
        scoring_func="sigmoid",
        bias=bias,
        to_physical_map=eplb.logical_to_physical if eplb is not None else None,
        logical_count=eplb.logical_count if eplb is not None else None,
        out=out,
    )
    indices, weights = out
    return weights, indices
