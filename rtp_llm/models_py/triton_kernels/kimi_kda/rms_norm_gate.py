"""Kimi KDA fused RMSNorm with a sigmoid output gate.

The K3 reference model uses FLA's ``FusedRMSNormGated`` operator.  Keeping the
normalization and gate in one Triton kernel is numerically significant: a
sequence of eager Torch operations rounds at different intermediate points and
can change later MoE routing decisions.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _kimi_kda_rms_norm_sigmoid_gate_kernel(
    x,
    gate,
    output,
    weight,
    eps,
    token_rows,
    feature_size: tl.constexpr,
    block_rows: tl.constexpr,
    block_features: tl.constexpr,
):
    row_block = tl.program_id(0)
    feature_offsets = tl.arange(0, block_features)
    feature_mask = feature_offsets < feature_size

    x_pointer = tl.make_block_ptr(
        x,
        (token_rows, feature_size),
        (feature_size, 1),
        (row_block * block_rows, 0),
        (block_rows, block_features),
        (1, 0),
    )
    x_block = tl.load(
        x_pointer,
        boundary_check=(0, 1),
    ).to(tl.float32)
    x_bar = tl.where(feature_mask[None, :], x_block, 0.0)
    variance = tl.sum(x_bar * x_bar, axis=1) / feature_size
    inverse_rms = 1.0 / tl.sqrt(variance + eps)

    norm_weight = tl.load(
        weight + feature_offsets,
        mask=feature_mask,
    ).to(tl.float32)
    gate_pointer = tl.make_block_ptr(
        gate,
        (token_rows, feature_size),
        (feature_size, 1),
        (row_block * block_rows, 0),
        (block_rows, block_features),
        (1, 0),
    )
    gate_block = tl.load(
        gate_pointer,
        boundary_check=(0, 1),
    ).to(tl.float32)
    result = (
        x_block * inverse_rms[:, None] * norm_weight[None, :] * tl.sigmoid(gate_block)
    )

    output_pointer = tl.make_block_ptr(
        output,
        (token_rows, feature_size),
        (feature_size, 1),
        (row_block * block_rows, 0),
        (block_rows, block_features),
        (1, 0),
    )
    tl.store(
        output_pointer,
        result.to(output_pointer.dtype.element_ty),
        boundary_check=(0, 1),
    )


@torch.compiler.disable
def kimi_kda_rms_norm_sigmoid_gate(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Apply K3's per-head RMSNorm and sigmoid output gate.

    The launch configurations are the deterministic choices selected by the
    FLA 0.5.1/Triton 3.7 Golden runtime on SM103 for the K3 short/decode and
    8192-token shapes.  They also reproduce the sealed Golden bit-for-bit under
    RTP's Triton 3.6 runtime.
    """

    if x.shape != gate.shape:
        raise ValueError(
            f"KDA norm input and gate shapes must match, got {x.shape} and "
            f"{gate.shape}"
        )
    if x.ndim < 2:
        raise ValueError(f"KDA norm input must have at least two dims, got {x.ndim}")
    feature_size = x.shape[-1]
    if weight.shape != (feature_size,):
        raise ValueError(
            f"KDA norm weight must have shape {(feature_size,)}, got {weight.shape}"
        )
    if x.numel() == 0:
        return torch.empty_like(x)
    if not x.is_cuda:
        x_float = x.float()
        return (
            x_float
            * torch.rsqrt(x_float.square().mean(dim=-1, keepdim=True) + eps)
            * weight.float()
            * torch.sigmoid(gate.float())
        ).to(dtype=x.dtype)
    if not gate.is_cuda or not weight.is_cuda:
        raise ValueError("KDA norm input, gate, and weight must share a CUDA device")
    if x.device != gate.device or x.device != weight.device:
        raise ValueError("KDA norm input, gate, and weight must be on the same device")
    if x.stride(-1) != 1 or gate.stride(-1) != 1 or weight.stride(-1) != 1:
        raise ValueError("KDA norm tensors must be contiguous in the feature dim")

    token_rows = x.numel() // feature_size
    block_features = triton.next_power_of_2(feature_size)
    if feature_size > min(65536 // x.element_size(), block_features):
        raise RuntimeError(
            f"KDA fused RMSNorm does not support feature size {feature_size}"
        )

    # FLA keys its autotune cache by NB = ceil(T / (2048 * 32)).
    if triton.cdiv(token_rows, 2048 * 32) == 1:
        block_rows = 16
        num_warps = 16
    else:
        block_rows = 32
        num_warps = 4

    output = torch.empty_like(x)
    _kimi_kda_rms_norm_sigmoid_gate_kernel[(triton.cdiv(token_rows, block_rows),)](
        x.reshape(token_rows, feature_size),
        gate.reshape(token_rows, feature_size),
        output.reshape(token_rows, feature_size),
        weight,
        eps,
        token_rows,
        feature_size=feature_size,
        block_rows=block_rows,
        block_features=block_features,
        num_warps=num_warps,
        num_stages=3,
    )
    return output


__all__ = ["kimi_kda_rms_norm_sigmoid_gate"]
