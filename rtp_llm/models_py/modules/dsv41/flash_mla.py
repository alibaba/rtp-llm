"""FlashMLA head padding for arbitrary valid V4.1 tensor-parallel widths.

Each attention head is independent. Padding unused query heads to the kernel's
64/128-head tiles keeps the model's head partition unrestricted and slices only
unobserved heads from the result.
"""

import importlib

import torch
import torch.nn.functional as F


def _backend(q):
    sm90 = (
        q.is_cuda
        and torch.version.hip is None
        and torch.cuda.get_device_capability(q.device) == (9, 0)
    )
    return importlib.import_module("flash_mla_sm90" if sm90 else "flash_mla"), sm90


def _pad_heads(q, attn_sink):
    heads = q.shape[-2]
    kernel_heads = ((heads + 63) // 64) * 64
    if kernel_heads == heads:
        return q, attn_sink, heads
    q = F.pad(q, (0, 0, 0, kernel_heads - heads))
    if attn_sink is not None:
        attn_sink = F.pad(attn_sink, (0, kernel_heads - heads), value=float("inf"))
    return q, attn_sink, heads


def flash_mla_sparse_fwd(
    q, kv, indices, sm_scale, d_v=512, attn_sink=None, topk_length=None
):
    backend, _ = _backend(q)
    native = backend.flash_mla_sparse_fwd

    q, sink, heads = _pad_heads(q, attn_sink)
    output, maximum, lse = native(
        q, kv, indices, sm_scale, d_v=d_v, attn_sink=sink, topk_length=topk_length
    )
    return output[:, :heads], maximum[:, :heads], lse[:, :heads]


def flash_mla_with_kvcache(*, q, attn_sink=None, **kwargs):
    backend, sm90 = _backend(q)
    native = backend.flash_mla_with_kvcache
    if sm90:
        # Each wheel owns a distinct planner type. Preserve the caller's
        # per-forward/graph lifetime rather than caching plans on the module.
        owner = kwargs["tile_scheduler_metadata"]
        plan = getattr(owner, "_dsv41_sm90_plan", None)
        if plan is None:
            plan = backend.get_mla_metadata()[0]
            owner._dsv41_sm90_plan = plan
        kwargs["tile_scheduler_metadata"] = plan

    q, sink, heads = _pad_heads(q, attn_sink)
    output, lse = native(q=q, attn_sink=sink, **kwargs)
    return output[..., :heads, :], lse[:, :heads, :]
