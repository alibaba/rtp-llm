"""FlashMLA head padding for arbitrary valid V4.1 tensor-parallel widths.

Each attention head is independent. Padding unused query heads to the kernel's
64/128-head tiles keeps the model's head partition unrestricted and slices only
unobserved heads from the result.
"""

import torch
import torch.nn.functional as F

from rtp_llm.device.device_type import is_h20


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
    from flash_mla import flash_mla_sparse_fwd as native

    q, sink, heads = _pad_heads(q, attn_sink)
    output, maximum, lse = native(
        q, kv, indices, sm_scale, d_v=d_v, attn_sink=sink, topk_length=topk_length
    )
    return output[:, :heads], maximum[:, :heads], lse[:, :heads]


def flash_mla_with_kvcache(*, q, attn_sink=None, **kwargs):
    from flash_mla import flash_mla_with_kvcache as native

    q, sink, heads = _pad_heads(q, attn_sink)
    if (
        q.is_cuda
        and q.shape[1] > 1
        and kwargs.get("indices") is not None
        and is_h20(q.device)
    ):
        output, lse = _single_query_decode(native, q, sink, kwargs)
    else:
        output, lse = native(q=q, attn_sink=sink, **kwargs)
    return output[..., :heads, :], lse[:, :heads, :]


def _single_query_decode(native, q, sink, kwargs):
    """Keep H20 verification on the single-step sparse decode arithmetic.

    Native multi-query planning changes split/merge boundaries with B*S.
    Small BF16 differences can then amplify through downstream quantization.
    Sparse indices already encode each query's causal visibility. Reuse the
    existing kernel and per-forward planner ownership, including graph capture.
    """
    from flash_mla import get_mla_metadata

    owner = kwargs["tile_scheduler_metadata"]
    plans = getattr(owner, "_dsv41_query_plans", None)
    if plans is None:
        plans = tuple(get_mla_metadata()[0] for _ in range(q.shape[1]))
        owner._dsv41_query_plans = plans
    if len(plans) != q.shape[1]:
        raise ValueError("FlashMLA query count changed on an existing planner")
    outputs, lses = [], []
    for index, plan in enumerate(plans):
        args = dict(kwargs, tile_scheduler_metadata=plan)
        for name in ("indices", "extra_indices_in_kvcache"):
            value = args.get(name)
            if value is not None:
                args[name] = value[:, index : index + 1].contiguous()
        output, lse = native(
            q=q[:, index : index + 1].contiguous(), attn_sink=sink, **args
        )
        outputs.append(output)
        lses.append(lse)
    return torch.cat(outputs, dim=1), torch.cat(lses, dim=2)
