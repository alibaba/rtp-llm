"""Bounded, nonmutating BF16 CUDA finite check; caller performs all-rank vote."""
import torch


def batched_finite(gradients, max_views=32, view_numel=1048576):
    if type(max_views) is not int or type(view_numel) is not int or min(max_views, view_numel) < 1:
        raise ValueError('invalid finite-check batch limits')
    device = torch.device('cuda', torch.cuda.current_device())
    finite = torch.ones((), dtype=torch.int32, device=device)
    batch = []

    def flush():
        if batch:
            norms = torch._foreach_norm(batch, float('inf'), dtype=torch.float32)
            finite.bitwise_and_(torch.isfinite(torch.stack(norms)).all().to(torch.int32))
            batch.clear()

    for grad in gradients:
        if grad is None:
            continue
        if grad.device != device or grad.dtype != torch.bfloat16 or not grad.is_contiguous():
            raise ValueError('finite-check input must be a contiguous local CUDA BF16 shard')
        flat = grad.reshape(-1)
        for start in range(0, flat.numel(), view_numel):
            batch.append(flat[start:start + view_numel])
            if len(batch) == max_views:
                flush()
    flush()
    return finite
