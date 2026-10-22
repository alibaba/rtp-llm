"""V4.1 FA4 forward with native BSHD producer/consumer layout.

No package-path mutation, automatic installation, or silent retry after a
supported launch fails. FA4 is the model default and requires its Python package
to be importable. The attention selector can explicitly choose SDPA instead.
"""

from functools import lru_cache

import torch


def _supported(q, k, v):
    return (
        not torch.is_grad_enabled()
        and all(
            isinstance(t, torch.Tensor) and t.layout == torch.strided for t in (q, k, v)
        )
        and q.device.type == "cuda"
        and q.dtype == torch.bfloat16
        and q.ndim == 4
        and q.shape[0] > 0
        and q.shape[1] > 0
        and q.shape[2:] == (16, 64)
        and all(
            t.shape == q.shape
            and t.device == q.device
            and t.dtype == q.dtype
            and t.stride(-1) == 1
            for t in (q, k, v)
        )
        and torch.cuda.get_device_capability(q.device)[0] == 10
    )


@lru_cache(maxsize=1)
def _fa4_forward():
    # Keep optional imports off CPU/SDPA paths and resolve before capture
    # during normal warmup. A missing plugin is a visible configuration error.
    from flash_attn.cute import flash_attn_func

    return flash_attn_func


def apply_vision_fa4(q, k, v):
    """Fresh [B,S,16,64] output, or None for an unsupported input contract.

    B isolates independent images. Non-contiguous outer strides are intentional:
    Q/K may share a rotary allocation and V remains a view into packed QKV.
    Only the head dimension must be contiguous. Inputs are never mutated.
    """
    if not _supported(q, k, v):
        return None
    with torch.cuda.device(q.device):
        output, _ = _fa4_forward()(q, k, v, softmax_scale=0.125, causal=False)
    return output


@lru_cache(maxsize=1)
def _fa4_varlen_forward():
    from flash_attn.cute import flash_attn_varlen_func

    return flash_attn_varlen_func


def apply_vision_fa4_varlen(q, k, v, cu_seqlens, max_seqlen):
    """Attend within each image in packed THD storage, without regrouping copies.

    The caller owns the cumulative image boundaries and their stream readiness.
    They are built once per forward, or retained with an exact-shape graph.
    """
    if q.ndim != 3 or not _supported(*(t.unsqueeze(0) for t in (q, k, v))):
        return None
    if (
        cu_seqlens.device != q.device
        or cu_seqlens.dtype != torch.int32
        or cu_seqlens.ndim != 1
        or not cu_seqlens.is_contiguous()
        or cu_seqlens.numel() < 2
        or max_seqlen <= 0
    ):
        raise ValueError("invalid packed vision attention boundaries")
    with torch.cuda.device(q.device):
        output, _ = _fa4_varlen_forward()(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            softmax_scale=0.125,
            causal=False,
        )
    return output


@torch.inference_mode()
def warmup_vision_fa4_varlen(device):
    """Compile the short/long sequence variants using native producer strides."""
    with torch.cuda.device(device):
        for length in (8, 512):
            rows = 2 * length
            qk = torch.zeros(rows, 32, 64, device=device, dtype=torch.bfloat16)
            qkv = torch.zeros(rows, 3072, device=device, dtype=torch.bfloat16)
            v = qkv[:, 2048:].view(rows, 16, 64)
            cu = torch.tensor([0, length, rows], device=device, dtype=torch.int32)
            apply_vision_fa4_varlen(qk[:, :16], qk[:, 16:], v, cu, length)
