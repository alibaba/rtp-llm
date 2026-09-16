"""Readable V4.1 formulas for component execution and kernel comparisons.

Quantized GEMM wrappers must preserve these conversion boundaries. These
functions do not replace real checkpoint or GPU acceptance measurements.
"""

import importlib
import os

import torch
import torch.nn.functional as F

_HC_SPLIT_SINKHORN_KERNELS = {}
_HC_POINTWISE_KERNELS = {}
_HC_PRENORM_READY = {}


def _hc_prenorm_supported(hidden, weight):
    return (
        os.environ.get("DSV41_MHC_PRENORM", "1") == "1"
        and not torch.is_grad_enabled()
        and not torch.is_autocast_enabled()
        and hidden.is_cuda
        and hidden.dtype == torch.bfloat16
        and hidden.ndim >= 3
        and hidden.shape[-2:] == (4, 5120)
        and hidden.numel() > 0
        and hidden.is_contiguous()
        and weight.shape == (24, 20480)
        and weight.dtype == torch.float32
        and weight.device == hidden.device
        and weight.is_contiguous()
        and torch.version.cuda is not None
        and torch.version.cuda.split(".")[0] == "13"
        and torch.cuda.get_device_capability(hidden.device)[0] == 10
    )


def _hc_prenorm(hidden, weight, norm_eps):
    from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import tf32_hc_prenorm_gemm
    from rtp_llm.models_py.modules.dsv41._hc_prenorm_triton import (
        hc_prenorm_reduce_kernel,
    )

    rows = hidden.numel() // 20480
    key = (hidden.device.index, rows, norm_eps)
    with torch.cuda.device(hidden.device):
        splits = _HC_PRENORM_READY.get(key)
        if splits is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("V4.1 HC prenorm must be warmed before capture")
            sms = torch.cuda.get_device_properties(hidden.device).multi_processor_count
            splits = max(1, min(sms // max((rows + 63) // 64, 1), 80))
        products = hidden.new_empty((splits, rows, 24), dtype=torch.float32)
        squares = hidden.new_empty((splits, rows), dtype=torch.float32)
        mixes = hidden.new_empty((rows, 24), dtype=torch.float32)
        # BF16 activations are exact TF32 inputs; FP32 weights use the existing
        # upstream TF32 multiply with FP32 accumulation and square reduction.
        tf32_hc_prenorm_gemm(hidden.view(rows, 20480), weight, products, squares, splits)
        hc_prenorm_reduce_kernel[(rows,)](
            products,
            squares,
            mixes,
            ROWS=rows,
            HIDDEN=20480,
            SPLITS=splits,
            NORM_EPS=norm_eps,
            BLOCK_S=1 << (splits - 1).bit_length(),
        )
        _HC_PRENORM_READY[key] = splits
    return mixes.view(*hidden.shape[:-2], 24)


def _hc_pointwise_supported(hidden, *mixes):
    return (
        os.environ.get("DSV41_MHC_POINTWISE", "1") == "1"
        and not torch.is_grad_enabled()
        and hidden.is_cuda
        and hidden.dtype == torch.bfloat16
        and hidden.ndim >= 3
        and hidden.shape[-2] == 4
        and hidden.numel() > 0
        and hidden.is_contiguous()
        and all(
            mix.dtype == torch.float32
            and mix.device == hidden.device
            and mix.is_contiguous()
            for mix in mixes
        )
    )


def _hc_pointwise_kernels(hidden):
    key = (hidden.device.index, hidden.shape[-1])
    kernels = _HC_POINTWISE_KERNELS.get(key)
    if kernels is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("V4.1 HC pointwise kernels must be warmed before capture")
        from rtp_llm.models_py.modules.dsv4 import tilelang_kernels  # noqa: F401

        prefix = "rtp_llm.models_py.3rdparty.tile_kernels.mhc."
        pre = importlib.import_module(prefix + "pre_apply_mix_kernel")
        post = importlib.import_module(prefix + "post_kernel")
        kernels = (
            pre._mhc_pre_apply_mix_fwd(4, hidden.shape[-1]),
            post._mhc_post_fwd(4, hidden.shape[-1]),
        )
        _HC_POINTWISE_KERNELS[key] = kernels
    return kernels


def _hc_split_sinkhorn(mixes, scale, base, iterations, eps):
    key = (mixes.device.index, iterations, eps)
    kernels = _HC_SPLIT_SINKHORN_KERNELS.get(key)
    if kernels is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("V4.1 HC split/Sinkhorn must be warmed before capture")
        # Reuse the existing TileLang environment and forward-only kernels.
        from rtp_llm.models_py.modules.dsv4 import tilelang_kernels  # noqa: F401

        prefix = "rtp_llm.models_py.3rdparty.tile_kernels.mhc."
        split = importlib.import_module(prefix + "pre_split_mixes_kernel")
        sinkhorn = importlib.import_module(prefix + "sinkhorn_kernel")
        kernels = (
            split._mhc_pre_split_mixes_fwd(4, 2.0, eps, token_block_size=32),
            sinkhorn._mhc_sinkhorn_fwd(4, 1, iterations, eps),
        )
        _HC_SPLIT_SINKHORN_KERNELS[key] = kernels
    rows = mixes.numel() // 24
    pre = mixes.new_empty(rows, 4)
    post = mixes.new_empty(rows, 4)
    comb_raw = mixes.new_empty(rows, 16)
    comb = mixes.new_empty(rows, 4, 4)
    kernels[0](mixes.view(rows, 24), scale, base, pre, post, comb_raw)
    kernels[1](comb_raw.view(rows, 4, 4), comb)
    batch = mixes.shape[:-1]
    return pre.view(*batch, 4), post.view(*batch, 4), comb.view(*batch, 4, 4)


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


def _rms_norm_native_supported(hidden, weight):
    # Code-default on (R4-3): <=1 BF16 ulp vs the reference path measured on
    # B300 SM103 (W10-032) and GB200 SM100 (W10-070); the env stays only as a
    # diagnostic override.
    return (
        os.environ.get("DSV41_RMSNORM_NATIVE", "1") == "1"
        and not torch.is_grad_enabled()
        and not torch.is_autocast_enabled()
        and hidden.is_cuda
        and hidden.dtype == torch.bfloat16
        and hidden.ndim >= 2
        and hidden.numel() > 0
        and hidden.is_contiguous()
        and weight.ndim == 1
        and weight.shape == (hidden.shape[-1],)
        and weight.dtype == torch.bfloat16
        and weight.device == hidden.device
        and weight.is_contiguous()
        and torch.version.cuda is not None
        and torch.version.cuda.split(".")[0] == "13"
        and torch.cuda.get_device_capability(hidden.device)[0] == 10
    )


def rms_norm(
    hidden: torch.Tensor, weight: torch.Tensor, eps: float = 1e-20
) -> torch.Tensor:
    # The native kernel preserves the fp32-upcast formula; measured contract is
    # at most one BF16 ulp from the reference path with no worse FP64 error.
    if _rms_norm_native_supported(hidden, weight):
        from rtp_llm.ops.compute_ops import rtp_llm_ops

        flat = hidden.view(-1, hidden.shape[-1])
        out = torch.empty_like(flat)
        rtp_llm_ops.rmsnorm(
            out, flat, weight, eps, torch.cuda.current_stream().cuda_stream
        )
        return out.view(hidden.shape)
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
    if _hc_pointwise_supported(hidden, pre_mix):
        dim = hidden.shape[-1]
        out = hidden.new_empty((*hidden.shape[:-2], dim))
        _hc_pointwise_kernels(hidden)[0](
            hidden.view(-1, 4, dim), pre_mix.view(-1, 4), out.view(-1, dim)
        )
        return out
    return (hidden.float() * pre_mix.unsqueeze(-1)).sum(-2).to(hidden.dtype)


def hc_post(
    output: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    if (
        _hc_pointwise_supported(residual, post, comb)
        and output.dtype == residual.dtype
        and output.device == residual.device
        and output.is_contiguous()
        and output.shape == residual.shape[:-2] + (residual.shape[-1],)
        and post.shape == residual.shape[:-1]
        and comb.shape == residual.shape[:-2] + (4, 4)
    ):
        dim = residual.shape[-1]
        out = torch.empty_like(residual)
        _hc_pointwise_kernels(residual)[1](
            comb.view(-1, 4, 4),
            residual.view(-1, 4, dim),
            post.view(-1, 4),
            output.view(-1, dim),
            out.view(-1, 4, dim),
        )
        return out
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
    if _hc_prenorm_supported(hidden, weight):
        mixes = _hc_prenorm(hidden, weight, norm_eps)
    else:
        flat = hidden.flatten(-2).float()
        mixes = F.linear(flat, weight.float()) * torch.rsqrt(
            flat.square().mean(-1, keepdim=True) + norm_eps
        )
    if (
        os.environ.get("DSV41_HC_SPLIT_SINKHORN", "0") == "1"
        and hc == 4
        and iterations >= 1
        and not torch.is_grad_enabled()
        and mixes.is_cuda
        and mixes.numel() > 0
        and mixes.is_contiguous()
        and scale.dtype == base.dtype == mixes.dtype == torch.float32
        and scale.device == base.device == mixes.device
        and scale.shape == (3,)
        and base.shape == (24,)
        and scale.is_contiguous()
        and base.is_contiguous()
    ):
        return _hc_split_sinkhorn(mixes, scale, base, iterations, hc_eps)
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
    if token_mask is not None and (
        token_mask.shape != hidden.shape[:-2] or token_mask.dtype != torch.bool
    ):
        raise ValueError(
            "Engram token_mask must contain one boolean per canonical token"
        )
    if (
        os.environ.get("DSV41_ENGRAM_FUSED_INJECT", "1") == "1"
        and not torch.is_grad_enabled()
        and hidden.is_cuda
        and hidden.numel() > 0
        and (hc, dim) == (4, 5120)
        and hidden.ndim >= 3
        and projected.shape == hidden.shape[:-2] + ((hc + 1) * dim,)
        and q_weight.shape == k_weight.shape == (hc, dim)
        and all(
            value.dtype == torch.bfloat16
            and value.device == hidden.device
            and value.is_contiguous()
            for value in (hidden, projected, q_weight, k_weight)
        )
        and (
            token_mask is None
            or (token_mask.device == hidden.device and token_mask.is_contiguous())
        )
    ):
        from rtp_llm.models_py.modules.dsv41._engram_inject_triton import (
            engram_inject_kernel,
        )

        out = torch.empty_like(hidden)
        engram_inject_kernel[(hidden.numel() // (hc * dim), hc)](
            hidden,
            projected,
            q_weight,
            k_weight,
            hidden if token_mask is None else token_mask,
            out,
            DIM=dim,
            HC=hc,
            HAS_MASK=token_mask is not None,
            EPS=eps,
            BLOCK=8192,
            enable_fp_fusion=False,
        )
        return out
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
        gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
    return (values + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(hidden.dtype)
