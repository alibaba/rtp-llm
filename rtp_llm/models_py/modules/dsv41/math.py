"""Readable V4.1 formulas for component execution and kernel comparisons.

Quantized GEMM wrappers must preserve these conversion boundaries. These
functions do not replace real checkpoint or GPU acceptance measurements.
"""

import torch
import torch.nn.functional as F


def dequantize_block32(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if weight.ndim != 2 or scale.ndim != 2:
        raise ValueError("block32 weight and scale must be matrices")
    rows, columns = weight.shape
    if scale.shape != ((rows + 31) // 32, (columns + 31) // 32):
        raise ValueError("dense FP8 requires a scale for each 32x32 block")
    expanded = scale.float().repeat_interleave(32, 0).repeat_interleave(32, 1)
    return (weight.float() * expanded[:rows, :columns]).to(torch.bfloat16)


def grouped_wo_a(attention: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """BF16 attention [..., groups, heads_per_group * head_dim] projection."""
    if attention.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError(
            "V4.1 wo_a consumes BF16 attention and converter-decoded BF16 weights"
        )
    if weight.ndim != 3 or attention.shape[-2:] != (weight.shape[0], weight.shape[2]):
        raise ValueError("wo_a group geometry does not match attention output")
    return torch.einsum("...gd,grd->...gr", attention, weight)


def swiglu_activation(
    gate: torch.Tensor,
    up: torch.Tensor,
    route_weight: torch.Tensor | None = None,
    limit: float = 10.0,
) -> torch.Tensor:
    if gate.shape != up.shape:
        raise ValueError("gate and up shapes must match")
    result = F.silu(gate.float().clamp(max=limit)) * up.float().clamp(-limit, limit)
    if route_weight is not None:
        if route_weight.shape != gate.shape[:-1]:
            raise ValueError("route_weight must contain one weight per routed row")
        result = result * route_weight.float().unsqueeze(-1)
    return result.to(torch.bfloat16)


def moe_gate(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    bias_vl: torch.Tensor,
    image_mask: torch.Tensor | None,
    topk: int,
    route_scale: float = 1.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = F.softplus(F.linear(hidden.float(), weight.float())).sqrt()
    selection_bias = bias.float()
    if image_mask is not None:
        if image_mask.dtype != torch.bool or image_mask.shape != hidden.shape[:-1]:
            raise ValueError(
                "image_mask must cover every text, delimiter and patch row"
            )
        selection_bias = torch.where(
            image_mask.unsqueeze(-1), bias_vl.float(), selection_bias
        )
    indices = (scores + selection_bias).topk(topk, dim=-1).indices
    weights = scores.gather(-1, indices)
    if topk > 1:
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
    return weights * route_scale, indices


def rms_norm(
    hidden: torch.Tensor, weight: torch.Tensor, eps: float = 1e-20
) -> torch.Tensor:
    values = hidden.float()
    return (
        values
        * torch.rsqrt(values.square().mean(-1, keepdim=True) + eps)
        * weight.float()
    ).to(hidden.dtype)


def identity_pre_mix(hidden: torch.Tensor) -> torch.Tensor:
    pre = hidden.new_zeros(hidden.shape[:-1], dtype=torch.float32)
    pre[..., 0] = 1.0
    return pre


def hc_pre(hidden: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
    if pre_mix.shape != hidden.shape[:-1] or pre_mix.dtype != torch.float32:
        raise ValueError("pre_mix must be FP32 with one coefficient per HC stream")
    return (hidden.float() * pre_mix.unsqueeze(-1)).sum(-2).to(hidden.dtype)


def hc_post(
    output: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    # comb's first HC axis is the source stream, its second the destination.
    mixed = (comb.float().unsqueeze(-1) * residual.float().unsqueeze(-2)).sum(-3)
    return (post.float().unsqueeze(-1) * output.float().unsqueeze(-2) + mixed).to(
        output.dtype
    )


def hc_mixes(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    iterations: int = 20,
    norm_eps: float = 1e-20,
    hc_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hc = hidden.shape[-2]
    flat = hidden.flatten(-2).float()
    mixes = F.linear(flat, weight.float()) * torch.rsqrt(
        flat.square().mean(-1, keepdim=True) + norm_eps
    )
    pre = (mixes[..., :hc] * scale[0] + base[:hc]).sigmoid() + hc_eps
    post = (mixes[..., hc : 2 * hc] * scale[1] + base[hc : 2 * hc]).sigmoid() * 2
    comb = (mixes[..., 2 * hc :] * scale[2] + base[2 * hc :]).unflatten(-1, (hc, hc))
    comb = comb.softmax(-1) + hc_eps
    comb = comb / (comb.sum(-2, keepdim=True) + hc_eps)
    for _ in range(iterations - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(-2, keepdim=True) + hc_eps)
    return pre, post, comb


def engram_inject(
    hidden: torch.Tensor,
    projected: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    token_mask: torch.Tensor | None = None,
    eps: float = 1e-20,
) -> torch.Tensor:
    hc, dim = hidden.shape[-2:]
    key, value = projected.split((hc * dim, dim), dim=-1)
    key = key.float().unflatten(-1, (hc, dim))
    values = hidden.float()
    rstd = torch.rsqrt(values.square().mean(-1) + eps) * torch.rsqrt(
        key.square().mean(-1) + eps
    )
    dot = (
        (values * q_weight.float() * k_weight.float() * key).sum(-1) * rstd * dim**-0.5
    )
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
    if token_mask is not None:
        if token_mask.shape != hidden.shape[:-2] or token_mask.dtype != torch.bool:
            raise ValueError(
                "Engram token_mask must contain one boolean per canonical token"
            )
        gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
    return (values + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(hidden.dtype)
