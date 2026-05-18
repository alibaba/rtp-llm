from __future__ import annotations

import os

import torch

_INV_ROPE_FP8_MODE_GENERIC = 0
_INV_ROPE_FP8_MODE_D512_TOKEN4 = 1
_INV_ROPE_FP8_MODE_D512_GROUP8 = 2
_INV_ROPE_FP8_MODE_D512_TOKEN64 = 3
_INV_ROPE_FP8_MODE_D512_TILE16 = 4
_INV_ROPE_FP8_MODE_D512_TILE32 = 5
_INV_ROPE_FP8_MODE_D512_TOKEN64_STREAM = 6
_INV_ROPE_FP8_MODE_D512_HEAD1_SMALL = 7
_INV_ROPE_FP8_MODE_D512_TOKEN64_W16 = 8
_INV_ROPE_FP8_MODE_D512_TOKEN64_W32 = 9

# Large-M indexed InvRoPE+FP8 was tested as a standalone replacement for
# materialized freqs gather + fused_inv_rope_fp8_quant.  It only wins up to
# about M=2048; beyond that the original Triton fused quant kernel dominates
# total time and the indexed CUDA variants are neutral or slower.  Keep the
# GraphFX rewrite, but dispatch back to the original kernel for large M so the
# pass is a stable optimization instead of a large-prefill regression.
_INV_ROPE_FP8_INDEXED_MAX_M = 2048


def indexed_rope_enabled() -> bool:
    return os.environ.get("DSV4_INDEXED_ROPE_CUDA", "0") == "1"


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
    fp8_buf: torch.Tensor | None = None,
    scale_buf: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    d_per_group = heads_per_group * head_dim
    tma_m = ((m + 3) // 4) * 4
    if fp8_buf is None:
        fp8_work = torch.empty(
            (n_groups, m, d_per_group), dtype=torch.float8_e4m3fn, device=device
        )
    else:
        if not (
            fp8_buf.is_cuda
            and fp8_buf.dtype == torch.float8_e4m3fn
            and fp8_buf.device == device
            and fp8_buf.dim() == 3
            and int(fp8_buf.shape[0]) == n_groups
            and int(fp8_buf.shape[1]) >= m
            and int(fp8_buf.shape[2]) == d_per_group
        ):
            raise ValueError(
                "invalid indexed inv_rope fp8_buf: "
                f"shape={tuple(fp8_buf.shape)} dtype={fp8_buf.dtype} device={fp8_buf.device}; "
                f"expected [{n_groups}, >= {m}, {d_per_group}] float8_e4m3fn on {device}"
            )
        fp8_work = fp8_buf[:, :m, :]

    if scale_buf is None:
        scale_work = torch.empty(
            n_groups * heads_per_group * tma_m, dtype=torch.int32, device=device
        ).as_strided(
            (n_groups, m, heads_per_group),
            (heads_per_group * tma_m, 1, tma_m),
        )
    else:
        if not (
            scale_buf.is_cuda
            and scale_buf.dtype == torch.int32
            and scale_buf.device == device
            and scale_buf.dim() == 3
            and int(scale_buf.shape[0]) == n_groups
            and int(scale_buf.shape[1]) >= m
            and int(scale_buf.shape[2]) == heads_per_group
        ):
            raise ValueError(
                "invalid indexed inv_rope scale_buf: "
                f"shape={tuple(scale_buf.shape)} dtype={scale_buf.dtype} device={scale_buf.device}; "
                f"expected [{n_groups}, >= {m}, {heads_per_group}] int32 on {device}"
            )
        scale_work = scale_buf[:, :m, :]
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


def _indexed_inv_rope_fp8_quant_kernel_mode(
    m: int,
    h: int,
    d: int,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_head_dim: int,
) -> int:
    if not (
        h == 64
        and d == 512
        and int(n_groups) == 8
        and int(heads_per_group) == 8
        and int(nope_dim) == 448
        and int(rope_head_dim) == 64
    ):
        return _INV_ROPE_FP8_MODE_GENERIC
    override = os.environ.get("DSV4_INDEXED_INV_ROPE_FP8_QUANT_D512_MODE", "").lower()
    if override == "generic":
        return _INV_ROPE_FP8_MODE_GENERIC
    if override == "token4":
        return _INV_ROPE_FP8_MODE_D512_TOKEN4
    if override == "group8":
        return _INV_ROPE_FP8_MODE_D512_GROUP8
    if override == "token64":
        return _INV_ROPE_FP8_MODE_D512_TOKEN64
    if override in ("token64_stream", "stream"):
        return _INV_ROPE_FP8_MODE_D512_TOKEN64_STREAM
    if override in ("head1", "head1_small", "small_head"):
        return _INV_ROPE_FP8_MODE_D512_HEAD1_SMALL
    if override in ("token64_w16", "w16"):
        return _INV_ROPE_FP8_MODE_D512_TOKEN64_W16
    if override in ("token64_w32", "w32"):
        return _INV_ROPE_FP8_MODE_D512_TOKEN64_W32
    if override == "tile16":
        return _INV_ROPE_FP8_MODE_D512_TILE16
    if override == "tile32":
        return _INV_ROPE_FP8_MODE_D512_TILE32
    if 1 < m <= 64:
        return _INV_ROPE_FP8_MODE_D512_HEAD1_SMALL
    return _INV_ROPE_FP8_MODE_D512_TOKEN4 if m <= 512 else _INV_ROPE_FP8_MODE_D512_TOKEN64


def _indexed_inv_rope_fp8_quant_use_indexed_kernel(
    m: int,
    h: int,
    d: int,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_head_dim: int,
) -> bool:
    if not (
        h == 64
        and d == 512
        and int(n_groups) == 8
        and int(heads_per_group) == 8
        and int(nope_dim) == 448
        and int(rope_head_dim) == 64
    ):
        return True
    return int(m) <= _INV_ROPE_FP8_INDEXED_MAX_M


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
    backend: str | None = None,
    fp8_buf: torch.Tensor | None = None,
    scale_buf: torch.Tensor | None = None,
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

    m = int(o.shape[0] * o.shape[1]) if o.dim() == 4 else int(o.shape[0])
    h = int(o.shape[-2])
    d = int(o.shape[-1])

    selected_backend = (
        backend
        if backend is not None
        else os.environ.get("DSV4_INDEXED_INV_ROPE_FP8_QUANT_BACKEND", "cuda")
    ).lower()
    if selected_backend == "triton":
        if fp8_buf is not None or scale_buf is not None:
            raise ValueError("indexed Triton backend does not support explicit output buffers")
        from rtp_llm.models_py.modules.dsv4._fused_inv_rope_fp8_quant_triton import (
            fused_indexed_inv_rope_fp8_quant,
        )

        return fused_indexed_inv_rope_fp8_quant(
            o,
            freqs_cis,
            position_ids,
            int(n_groups),
            int(heads_per_group),
            int(nope_dim),
            int(rope_head_dim),
            quant_group_size=int(quant_group_size),
            eps=float(eps),
        )
    if selected_backend != "cuda":
        raise ValueError(
            f"invalid DSV4_INDEXED_INV_ROPE_FP8_QUANT_BACKEND={selected_backend!r}; "
            "expected 'cuda' or 'triton'"
        )

    if not _indexed_inv_rope_fp8_quant_use_indexed_kernel(
        m,
        h,
        d,
        int(n_groups),
        int(heads_per_group),
        int(nope_dim),
        int(rope_head_dim),
    ):
        from rtp_llm.models_py.modules.dsv4._fused_inv_rope_fp8_quant_triton import (
            fused_inv_rope_fp8_quant,
        )

        # Large-M fallback intentionally materializes the same freqs tensor as
        # the pre-rewrite graph.  The indexed kernel removes the gather, but
        # profiling showed the standalone indexed path is slower once M grows:
        # at M=65536 the original gather+fused kernel was ~1721us while the
        # best indexed CUDA path was ~1765us.  Preserving the original kernel
        # here avoids a GraphFX negative optimization while still letting small
        # and medium M use the indexed CUDA path.
        selected_freqs = freqs_cis.index_select(
            0, position_ids.to(dtype=torch.long).contiguous()
        ).contiguous()
        return fused_inv_rope_fp8_quant(
            o,
            selected_freqs,
            n_groups=int(n_groups),
            heads_per_group=int(heads_per_group),
            nope_dim=int(nope_dim),
            rope_head_dim=int(rope_head_dim),
            quant_group_size=int(quant_group_size),
            eps=float(eps),
            fp8_buf=fp8_buf,
            scale_buf=scale_buf,
            impl="optimized",
        )

    rtp_llm_ops = _load_rtp_llm_ops()
    kernel_mode = _indexed_inv_rope_fp8_quant_kernel_mode(
        m,
        h,
        d,
        int(n_groups),
        int(heads_per_group),
        int(nope_dim),
        int(rope_head_dim),
    )
    o_flat = o.reshape(m, int(o.shape[-2]), int(o.shape[-1]))
    output_q, output_s = _alloc_inv_rope_outputs(
        m,
        int(n_groups),
        int(heads_per_group),
        d,
        o.device,
        fp8_buf=fp8_buf,
        scale_buf=scale_buf,
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
        int(kernel_mode),
    )
    return output_q, output_s
