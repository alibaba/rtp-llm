from __future__ import annotations

import os
import weakref
from typing import Optional

import torch

from rtp_llm.models_py.kernels.cuda.dsv4_indexed_rope import (
    dsv4_indexed_inv_rope_fp8_quant,
    dsv4_indexed_rmsnorm_rope,
)
from rtp_llm.models_py.modules.dsv4._fused_inv_rope_fp8_quant_triton import (
    fused_inv_rope_fp8_quant,
)
from rtp_llm.models_py.modules.dsv4._fused_rmsnorm_rope_triton import fused_rmsnorm_rope

_FreqsTokenEntry = tuple[
    weakref.ReferenceType[torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    bool,
]
_FREQS_TOKEN_REGISTRY: dict[int, _FreqsTokenEntry] = {}
_FREQS_TOKEN_STORAGE_REGISTRY: dict[tuple, _FreqsTokenEntry] = {}
_FREQS_TOKEN_ORDER: list[int] = []
_MAX_FREQS_TOKENS = 4096


def _debug_runtime_enabled() -> bool:
    return os.environ.get("DSV4_INDEXED_ROPE_RUNTIME_DEBUG", "0").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _debug_runtime(message: str) -> None:
    if not _debug_runtime_enabled():
        return
    try:
        import logging

        logging.getLogger(__name__).info(message)
    except Exception:
        return


def _token_enabled() -> bool:
    return os.environ.get("DSV4_INDEXED_ROPE_FREQ_TOKEN", "1").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _tensor_storage_key(tensor: torch.Tensor) -> tuple | None:
    if tensor.is_meta:
        return None
    try:
        return (
            int(tensor.data_ptr()),
            tuple(int(v) for v in tensor.shape),
            tuple(int(v) for v in tensor.stride()),
            str(tensor.dtype),
            str(tensor.device),
        )
    except Exception:
        return None


def _remember_freqs_token(
    token: torch.Tensor,
    freqs_cis: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    poison: bool,
) -> None:
    key = id(token)
    _FREQS_TOKEN_REGISTRY[key] = (weakref.ref(token), freqs_cis, position_ids, poison)
    storage_key = _tensor_storage_key(token)
    if storage_key is not None:
        _FREQS_TOKEN_STORAGE_REGISTRY[storage_key] = (
            weakref.ref(token),
            freqs_cis,
            position_ids,
            poison,
        )
    _FREQS_TOKEN_ORDER.append(key)
    overflow = len(_FREQS_TOKEN_ORDER) - _MAX_FREQS_TOKENS
    if overflow <= 0:
        return
    for old_key in _FREQS_TOKEN_ORDER[:overflow]:
        old_entry = _FREQS_TOKEN_REGISTRY.pop(old_key, None)
        if old_entry is not None:
            old_token = old_entry[0]()
            old_storage_key = _tensor_storage_key(old_token) if old_token is not None else None
            if old_storage_key is not None:
                _FREQS_TOKEN_STORAGE_REGISTRY.pop(old_storage_key, None)
    del _FREQS_TOKEN_ORDER[:overflow]


def _lookup_freqs_token(
    freqs_cis: torch.Tensor,
) -> Optional[tuple[torch.Tensor, torch.Tensor, bool]]:
    key = id(freqs_cis)
    entry = _FREQS_TOKEN_REGISTRY.get(key)
    if entry is None:
        storage_key = _tensor_storage_key(freqs_cis)
        entry = _FREQS_TOKEN_STORAGE_REGISTRY.get(storage_key) if storage_key is not None else None
        if entry is None:
            return None
    token_ref, table, position_ids, poison = entry
    token = token_ref()
    if token is None:
        _FREQS_TOKEN_REGISTRY.pop(key, None)
        storage_key = _tensor_storage_key(freqs_cis)
        if storage_key is not None:
            _FREQS_TOKEN_STORAGE_REGISTRY.pop(storage_key, None)
        return None
    if token is not freqs_cis and _tensor_storage_key(token) != _tensor_storage_key(freqs_cis):
        # Tensor ids can be reused after a previous compiled segment releases a
        # materialized freqs tensor.  Do not let stale provenance redirect an
        # unrelated fallback call into the indexed CUDA path.
        _FREQS_TOKEN_REGISTRY.pop(key, None)
        storage_key = _tensor_storage_key(freqs_cis)
        if storage_key is not None:
            _FREQS_TOKEN_STORAGE_REGISTRY.pop(storage_key, None)
        return None
    return table, position_ids, poison


def _make_poison_freqs_token(freqs_cis: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    # Indexed consumers use the tensor only as a provenance key.  The int8 dtype
    # makes any unfused consumer fail its original complex64 RoPE contract
    # immediately instead of silently using uninitialized rotations.
    return torch.empty(
        (pos.numel(), freqs_cis.shape[-1]),
        dtype=torch.int8,
        device=freqs_cis.device,
    )


def _can_poison_freqs_token(freqs_cis: torch.Tensor, pos: torch.Tensor) -> bool:
    return (
        freqs_cis.is_cuda
        and freqs_cis.dtype == torch.complex64
        and freqs_cis.dim() == 2
        and freqs_cis.is_contiguous()
        and pos.is_cuda
        and pos.is_contiguous()
    )


def _raise_poison_consumer(message: str) -> None:
    raise ValueError(
        "DSV4 indexed RoPE freqs token reached an unsupported or unfused consumer: "
        f"{message}. This indicates the GraphFX indexed RoPE consumer rewrite did "
        "not cover this path."
    )


def _rmsnorm_rope_token_rows(x: torch.Tensor) -> int:
    if x.dim() == 0:
        return 0
    last_dim = int(x.shape[-1])
    if last_dim <= 0:
        return 0
    if x.dim() == 4:
        return int(x.shape[0] * x.shape[1])
    return int(x.numel() // last_dim)


def _try_indexed_grouped_q_rmsnorm_rope(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    table: torch.Tensor,
    position_ids: torch.Tensor,
    rope_head_dim: int,
    eps: float,
) -> torch.Tensor | None:
    if weight is not None or x.dim() not in (2, 3):
        return None
    pos_rows = int(position_ids.numel())
    if pos_rows <= 0:
        return None
    d = int(x.shape[-1])
    if x.dim() == 2:
        rows = int(x.shape[0])
        if rows % pos_rows != 0:
            return None
        group_heads = rows // pos_rows
        x_grouped = x.view(pos_rows, 1, group_heads, d)
    else:
        if int(x.shape[0]) != pos_rows:
            return None
        group_heads = int(x.shape[1])
        x_grouped = x.view(pos_rows, 1, group_heads, d)
    if group_heads <= 1:
        return None
    try:
        out_grouped = dsv4_indexed_rmsnorm_rope(
            x_grouped,
            None,
            table,
            position_ids,
            int(rope_head_dim),
            eps=float(eps),
        )
    except ValueError as exc:
        _debug_runtime(
            "DSV4 indexed RoPE grouped-Q fallback: "
            f"shape={tuple(x.shape)} pos_rows={pos_rows} group_heads={group_heads} "
            f"reason={exc}"
        )
        return None
    _debug_runtime(
        "DSV4 indexed RoPE grouped-Q hit: "
        f"shape={tuple(x.shape)} pos_rows={pos_rows} group_heads={group_heads}"
    )
    return out_grouped.view_as(x)


def _inv_rope_quant_token_rows(o: torch.Tensor) -> int:
    if o.dim() == 4:
        return int(o.shape[0] * o.shape[1])
    if o.dim() == 3:
        return int(o.shape[0])
    return -1


def dsv4_indexed_rope_freqs_token(
    freqs_cis: torch.Tensor, position_ids: torch.Tensor
) -> torch.Tensor:
    """FX-inserted replacement for ``freqs_cis.index_select(...).contiguous()``.

    The producer and consumer currently live in different Dynamo graphs.  This
    token keeps the runtime provenance required by the consumer-side FX rewrite
    without materializing the gather.  It is only inserted by the GraphFX pass;
    model code continues to call the existing unfused reference helpers.
    """
    pos = position_ids.to(device=freqs_cis.device, dtype=torch.long).contiguous()
    if not _token_enabled():
        return freqs_cis.index_select(0, pos).contiguous()
    if _can_poison_freqs_token(freqs_cis, pos):
        token = _make_poison_freqs_token(freqs_cis, pos)
        _remember_freqs_token(token, freqs_cis, pos, poison=True)
        _debug_runtime(
            "DSV4 indexed RoPE freqs poison token: "
            f"rows={int(pos.numel())} table_shape={tuple(freqs_cis.shape)}"
        )
        return token
    materialized = freqs_cis.index_select(0, pos).contiguous()
    _remember_freqs_token(materialized, freqs_cis, pos, poison=False)
    _debug_runtime(
        "DSV4 indexed RoPE freqs materialized token: "
        f"rows={int(pos.numel())} table_shape={tuple(freqs_cis.shape)}"
    )
    return materialized


def dsv4_indexed_rmsnorm_rope_from_freqs(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    freqs_cis: torch.Tensor,
    rope_head_dim: int,
    *,
    eps: float = 1e-6,
    inverse: bool = False,
    out: torch.Tensor | None = None,
    inplace: bool = False,
    group_heads: int | None = None,
) -> torch.Tensor:
    provenance = _lookup_freqs_token(freqs_cis)
    if (
        provenance is not None
        and not inverse
        and out is None
        and not inplace
        and group_heads is None
    ):
        table, position_ids, poison = provenance
        if int(position_ids.numel()) == _rmsnorm_rope_token_rows(x):
            _debug_runtime(
                "DSV4 indexed RoPE direct hit: "
                f"x_shape={tuple(x.shape)} pos_rows={int(position_ids.numel())}"
            )
            return dsv4_indexed_rmsnorm_rope(
                x,
                weight,
                table,
                position_ids,
                int(rope_head_dim),
                eps=float(eps),
            )
        grouped = _try_indexed_grouped_q_rmsnorm_rope(
            x,
            weight,
            table,
            position_ids,
            int(rope_head_dim),
            float(eps),
        )
        if grouped is not None:
            return grouped
        if poison:
            _raise_poison_consumer(
                f"rmsnorm_rope row mismatch: x_shape={tuple(x.shape)} "
                f"x_rows={_rmsnorm_rope_token_rows(x)} pos_rows={int(position_ids.numel())}"
            )
        _debug_runtime(
            "DSV4 indexed RoPE provenance fallback: "
            f"x_shape={tuple(x.shape)} x_rows={_rmsnorm_rope_token_rows(x)} "
            f"pos_rows={int(position_ids.numel())}"
        )
    elif provenance is not None:
        _, _, poison = provenance
        if poison:
            _raise_poison_consumer(
                "rmsnorm_rope unsupported kwargs: "
                f"inverse={inverse} out={out is not None} inplace={inplace} "
                f"group_heads={group_heads}"
            )
    elif provenance is None:
        if freqs_cis.dtype != torch.complex64:
            _raise_poison_consumer(
                f"rmsnorm_rope token provenance missing for dtype={freqs_cis.dtype} "
                f"shape={tuple(freqs_cis.shape)}"
            )
        _debug_runtime(
            "DSV4 indexed RoPE missing provenance fallback: "
            f"x_shape={tuple(x.shape)} freqs_shape={tuple(freqs_cis.shape)}"
        )
    return fused_rmsnorm_rope(
        x,
        weight,
        freqs_cis,
        rope_head_dim,
        eps=eps,
        inverse=inverse,
        out=out,
        inplace=inplace,
        group_heads=group_heads,
    )


def dsv4_indexed_inv_rope_fp8_quant_from_freqs(
    o: torch.Tensor,
    freqs_cis: torch.Tensor,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_head_dim: int,
    quant_group_size: int = 128,
    eps: float = 1e-10,
    fp8_buf: torch.Tensor | None = None,
    scale_buf: torch.Tensor | None = None,
    impl: str | None = None,
    heads_per_cta: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    provenance = _lookup_freqs_token(freqs_cis)
    if provenance is not None:
        table, position_ids, poison = provenance
        if int(position_ids.numel()) == _inv_rope_quant_token_rows(o):
            return dsv4_indexed_inv_rope_fp8_quant(
                o,
                table,
                position_ids,
                int(n_groups),
                int(heads_per_group),
                int(nope_dim),
                int(rope_head_dim),
                quant_group_size=int(quant_group_size),
                eps=float(eps),
                fp8_buf=fp8_buf,
                scale_buf=scale_buf,
            )
        if poison:
            _raise_poison_consumer(
                f"inv_rope_fp8_quant row mismatch: o_shape={tuple(o.shape)} "
                f"o_rows={_inv_rope_quant_token_rows(o)} pos_rows={int(position_ids.numel())}"
            )
    elif freqs_cis.dtype != torch.complex64:
        _raise_poison_consumer(
            f"inv_rope_fp8_quant token provenance missing for dtype={freqs_cis.dtype} "
            f"shape={tuple(freqs_cis.shape)}"
        )
    return fused_inv_rope_fp8_quant(
        o,
        freqs_cis,
        n_groups=n_groups,
        heads_per_group=heads_per_group,
        nope_dim=nope_dim,
        rope_head_dim=rope_head_dim,
        quant_group_size=quant_group_size,
        eps=eps,
        fp8_buf=fp8_buf,
        scale_buf=scale_buf,
        impl=impl,
        heads_per_cta=heads_per_cta,
    )
