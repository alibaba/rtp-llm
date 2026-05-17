from __future__ import annotations

import os

import torch

_DEFAULT_Q_D128_MAX_M = 2048
_DEFAULT_Q_D128_LARGE_MIN_M = _DEFAULT_Q_D128_MAX_M + 1
_DEFAULT_KV_D512_MAX_M = 32768


def indexed_rope_enabled() -> bool:
    return os.environ.get("DSV4_INDEXED_ROPE_CUDA", "0") == "1"


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value[:1].lower() not in ("0", "f", "n")


def _strict_position_check(freqs_cis: torch.Tensor, position_ids: torch.Tensor) -> None:
    if (
        os.environ.get("DSV4_INDEXED_ROPE_STRICT", "0") == "0"
        or position_ids.numel() == 0
    ):
        return
    min_pos = int(position_ids.min().item())
    max_pos = int(position_ids.max().item())
    if min_pos < 0 or max_pos >= int(freqs_cis.shape[0]):
        raise ValueError(
            f"position_ids out of range for freqs_cis: min={min_pos} max={max_pos} "
            f"max_pos={int(freqs_cis.shape[0])}"
        )


def _load_rtp_llm_ops():
    try:
        import libth_transformer_config  # noqa: F401
        from librtp_compute_ops import rtp_llm_ops

        return rtp_llm_ops
    except Exception:
        from rtp_llm.ops.compute_ops import rtp_llm_ops

        return rtp_llm_ops


def _rmsnorm_rope_token_rows(x: torch.Tensor) -> int:
    d = int(x.shape[-1])
    return int(x.shape[0] * x.shape[1]) if x.dim() == 4 else int(x.numel() // d)


def _is_q_no_weight_path(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    rope_head_dim: int,
) -> bool:
    return (
        weight is None
        and x.dim() == 4
        and int(x.shape[2]) == 64
        and int(x.shape[-1]) in (128, 512)
        and int(rope_head_dim) == 64
    )


def _is_kv_d512_weight_path(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    rope_head_dim: int,
) -> bool:
    return (
        weight is not None
        and x.dim() in (2, 3)
        and int(x.shape[-1]) == 512
        and int(rope_head_dim) == 64
    )


def _within_max_m(token_rows: int, max_m: int) -> bool:
    return max_m <= 0 or token_rows <= max_m


def indexed_rmsnorm_rope_path(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    rope_head_dim: int,
) -> str:
    """Return the runtime implementation selected by the Python wrapper."""
    if not indexed_rope_enabled():
        return "materialized_fallback"
    token_rows = _rmsnorm_rope_token_rows(x)
    if _is_q_no_weight_path(x, weight, rope_head_dim):
        if int(x.shape[-1]) != 128:
            return "indexed_small"
        large_min_m = _env_int(
            "DSV4_INDEXED_RMSNORM_ROPE_Q_LARGE_MIN_M",
            _DEFAULT_Q_D128_LARGE_MIN_M,
        )
        if (
            _env_flag("DSV4_INDEXED_RMSNORM_ROPE_Q_LARGE_M", False)
            and token_rows >= large_min_m
        ):
            return "indexed_large"
        return "indexed_small"
    if _is_kv_d512_weight_path(x, weight, rope_head_dim):
        return "indexed_small"
    return "materialized_fallback"


def _is_indexed_rmsnorm_rope_semantically_supported(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    freqs_cis: torch.Tensor,
    position_ids: torch.Tensor,
    rope_head_dim: int,
) -> bool:
    if not (
        x.is_cuda
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
        and x.dim() in (2, 3, 4)
        and freqs_cis.is_cuda
        and freqs_cis.dtype == torch.complex64
        and freqs_cis.is_contiguous()
        and position_ids.is_cuda
        and position_ids.dtype in (torch.int32, torch.int64)
        and position_ids.is_contiguous()
        and x.device == freqs_cis.device == position_ids.device
    ):
        return False
    d = int(x.shape[-1])
    token_rows = _rmsnorm_rope_token_rows(x)
    if (
        d <= 0
        or d > 512
        or rope_head_dim <= 0
        or rope_head_dim > d
        or rope_head_dim % 2 != 0
        or freqs_cis.dim() != 2
        or int(freqs_cis.shape[1]) != rope_head_dim // 2
        or int(position_ids.numel()) != token_rows
    ):
        return False
    if weight is not None and not (
        weight.is_cuda
        and weight.dtype == torch.bfloat16
        and weight.is_contiguous()
        and weight.device == x.device
        and tuple(weight.shape) == (d,)
    ):
        return False
    return True


def is_indexed_rmsnorm_rope_supported(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    freqs_cis: torch.Tensor,
    position_ids: torch.Tensor,
    rope_head_dim: int,
) -> bool:
    return (
        indexed_rope_enabled()
        and _is_indexed_rmsnorm_rope_semantically_supported(
            x, weight, freqs_cis, position_ids, rope_head_dim
        )
        and indexed_rmsnorm_rope_path(x, weight, rope_head_dim).startswith("indexed")
    )


def _materialized_rmsnorm_rope_fallback(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    freqs_cis: torch.Tensor,
    position_ids: torch.Tensor,
    rope_head_dim: int,
    eps: float,
) -> torch.Tensor:
    from rtp_llm.models_py.modules.dsv4._fused_rmsnorm_rope_triton import (
        fused_rmsnorm_rope,
    )

    selected = freqs_cis.index_select(0, position_ids.to(dtype=torch.long)).contiguous()
    return fused_rmsnorm_rope(x, weight, selected, int(rope_head_dim), eps=float(eps))


def dsv4_indexed_rmsnorm_rope(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    freqs_cis: torch.Tensor,
    position_ids: torch.Tensor,
    rope_head_dim: int,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    if not _is_indexed_rmsnorm_rope_semantically_supported(
        x, weight, freqs_cis, position_ids, rope_head_dim
    ):
        raise ValueError(
            "unsupported dsv4_indexed_rmsnorm_rope input: "
            f"x_shape={tuple(x.shape)} x_dtype={x.dtype} x_contig={x.is_contiguous()} "
            f"weight_shape={None if weight is None else tuple(weight.shape)} "
            f"freqs_shape={tuple(freqs_cis.shape)} freqs_dtype={freqs_cis.dtype} "
            f"position_shape={tuple(position_ids.shape)} position_dtype={position_ids.dtype} "
            f"rope_head_dim={rope_head_dim}"
        )
    _strict_position_check(freqs_cis, position_ids)
    if not indexed_rmsnorm_rope_path(x, weight, rope_head_dim).startswith("indexed"):
        return _materialized_rmsnorm_rope_fallback(
            x,
            weight,
            freqs_cis,
            position_ids,
            int(rope_head_dim),
            float(eps),
        )

    rtp_llm_ops = _load_rtp_llm_ops()

    output = torch.empty_like(x)
    if weight is None:
        weight_arg = torch.empty(0, dtype=torch.bfloat16, device=x.device)
        has_weight = False
    else:
        weight_arg = weight
        has_weight = True
    rtp_llm_ops.dsv4_indexed_rmsnorm_rope(
        x,
        weight_arg,
        freqs_cis,
        position_ids,
        output,
        int(rope_head_dim),
        float(eps),
        bool(has_weight),
    )
    return output


def _alloc_inv_rope_outputs(
    m: int,
    n_groups: int,
    heads_per_group: int,
    head_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    d_per_group = heads_per_group * head_dim
    tma_m = ((m + 3) // 4) * 4
    fp8_work = torch.empty(
        (n_groups, m, d_per_group), dtype=torch.float8_e4m3fn, device=device
    )
    scale_work = torch.empty(
        n_groups * heads_per_group * tma_m, dtype=torch.int32, device=device
    ).as_strided(
        (n_groups, m, heads_per_group),
        (heads_per_group * tma_m, 1, tma_m),
    )
    return fp8_work.transpose(0, 1), scale_work.transpose(0, 1)


def is_indexed_inv_rope_fp8_quant_supported(
    o: torch.Tensor,
    freqs_cis: torch.Tensor,
    position_ids: torch.Tensor,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_head_dim: int,
    quant_group_size: int = 128,
) -> bool:
    if not indexed_rope_enabled():
        return False
    if not (
        o.is_cuda
        and o.dtype == torch.bfloat16
        and o.is_contiguous()
        and o.dim() in (3, 4)
        and freqs_cis.is_cuda
        and freqs_cis.dtype == torch.complex64
        and freqs_cis.is_contiguous()
        and position_ids.is_cuda
        and position_ids.dtype in (torch.int32, torch.int64)
        and position_ids.is_contiguous()
        and o.device == freqs_cis.device == position_ids.device
    ):
        return False
    m = int(o.shape[0] * o.shape[1]) if o.dim() == 4 else int(o.shape[0])
    h = int(o.shape[-2])
    d = int(o.shape[-1])
    return (
        int(position_ids.numel()) == m
        and h == int(n_groups) * int(heads_per_group)
        and d == int(nope_dim) + int(rope_head_dim)
        and d % quant_group_size == 0
        and d <= 512
        and int(rope_head_dim) > 0
        and int(rope_head_dim) % 2 == 0
        and int(nope_dim) % quant_group_size == quant_group_size - int(rope_head_dim)
        and freqs_cis.dim() == 2
        and int(freqs_cis.shape[1]) == int(rope_head_dim) // 2
    )


def dsv4_indexed_inv_rope_fp8_quant(
    o: torch.Tensor,
    freqs_cis: torch.Tensor,
    position_ids: torch.Tensor,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_head_dim: int,
    quant_group_size: int = 128,
    eps: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not is_indexed_inv_rope_fp8_quant_supported(
        o, freqs_cis, position_ids, n_groups, heads_per_group, nope_dim, rope_head_dim, quant_group_size
    ):
        raise ValueError(
            "unsupported dsv4_indexed_inv_rope_fp8_quant input: "
            f"o_shape={tuple(o.shape)} o_dtype={o.dtype} o_contig={o.is_contiguous()} "
            f"freqs_shape={tuple(freqs_cis.shape)} freqs_dtype={freqs_cis.dtype} "
            f"position_shape={tuple(position_ids.shape)} position_dtype={position_ids.dtype} "
            f"n_groups={n_groups} heads_per_group={heads_per_group} "
            f"nope_dim={nope_dim} rope_head_dim={rope_head_dim}"
        )
    _strict_position_check(freqs_cis, position_ids)

    rtp_llm_ops = _load_rtp_llm_ops()

    m = int(o.shape[0] * o.shape[1]) if o.dim() == 4 else int(o.shape[0])
    o_flat = o.reshape(m, int(o.shape[-2]), int(o.shape[-1]))
    output_q, output_s = _alloc_inv_rope_outputs(
        m, int(n_groups), int(heads_per_group), int(o.shape[-1]), o.device
    )
    rtp_llm_ops.dsv4_indexed_inv_rope_fp8_quant(
        o_flat,
        freqs_cis,
        position_ids,
        output_q,
        output_s,
        int(n_groups),
        int(heads_per_group),
        int(nope_dim),
        int(rope_head_dim),
        float(eps),
        float(torch.finfo(torch.float8_e4m3fn).max),
    )
    return output_q, output_s
