"""Batch pure-copy downsampling without changing the aligner GEMM layout.

DeepInfer's batch-downsample approach removes per-image pad/unfold launches.
Here each image still produces a contiguous [C * ratio**2, output_patches]
column matrix, so the existing per-image W1 sees exactly its original strides.
Only homogeneous BF16 SM10x inference batches use this specialization.
"""

import torch

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError as error:
    if error.name != "triton":
        raise
    triton = None


if triton is not None:

    @triton.jit
    def _batch_downsample_columns(
        X,
        Y,
        H: tl.constexpr,
        W: tl.constexpr,
        C: tl.constexpr,
        R: tl.constexpr,
        L: tl.constexpr,
        OW: tl.constexpr,
        BC: tl.constexpr,
        BL: tl.constexpr,
    ):
        # Large production batches exceed 2 GiB; all base offsets are int64.
        image = tl.program_id(2).to(tl.int64)
        channel = tl.program_id(0) * BC + tl.arange(0, BC)
        patch = tl.program_id(1) * BL + tl.arange(0, BL)
        row = patch // OW * R
        col = patch % OW * R
        for dy in tl.static_range(R):
            for dx in tl.static_range(R):
                source = (
                    image * H * W + (row[None, :] + dy) * W + col[None, :] + dx
                ) * C + channel[:, None]
                values = tl.load(
                    X + source,
                    (channel[:, None] < C)
                    & (patch[None, :] < L)
                    & (row[None, :] + dy < H)
                    & (col[None, :] + dx < W),
                    other=0,
                )
                target = (
                    image * C * R * R + channel[:, None] * R * R + dy * R + dx
                ) * L + patch[None, :]
                tl.store(
                    Y + target, values, (channel[:, None] < C) & (patch[None, :] < L)
                )


def is_supported(x, grids, ratio):
    """Return false for unsupported device, geometry, dtype or autograd."""
    if not (
        triton is not None
        and isinstance(x, torch.Tensor)
        and x.is_cuda
        and x.layout == torch.strided
        and not torch.is_grad_enabled()
        and x.dtype == torch.bfloat16
        and x.ndim == 2
        and x.shape[1] == 1024
        and x.is_contiguous()
        and ratio == 3
        and len(grids) >= 2
    ):
        return False
    height, width = grids[0]
    return (
        height > 0
        and width > 0
        and all(grid == grids[0] for grid in grids)
        and x.shape[0] == len(grids) * height * width
        and torch.cuda.get_device_capability(x.device)[0] == 10
    )


def apply_vision_downsample(x, grids, ratio):
    """Return ordered column views, or None for the unchanged per-image path."""
    if not is_supported(x, grids, ratio):
        return None
    height, width = grids[0]
    length = triton.cdiv(height, ratio) * triton.cdiv(width, ratio)
    columns = torch.empty(
        (len(grids), x.shape[1] * ratio * ratio, length),
        dtype=x.dtype,
        device=x.device,
    )
    _batch_downsample_columns[
        (triton.cdiv(x.shape[1], 32), triton.cdiv(length, 32), len(grids))
    ](
        x,
        columns,
        height,
        width,
        x.shape[1],
        ratio,
        length,
        triton.cdiv(width, ratio),
        32,
        32,
        num_warps=4,
    )
    return list(columns.unbind(0))
