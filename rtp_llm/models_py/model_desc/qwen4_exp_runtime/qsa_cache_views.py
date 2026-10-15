"""Strict QSA byte payload, metadata and block geometry validation."""

from __future__ import annotations

import torch

from rtp_llm.models_py.utils.cache_payload_view import (
    CachePayloadGeometry,
    payload_view,
    same_tensor_metadata,
)
from rtp_llm.ops.compute_ops import LayerKVCache, PyAttentionInputs

_same_tensor_metadata = same_tensor_metadata


def _typed_2d_pool(
    cache: LayerKVCache,
    *,
    tag: str,
    storage_dtype: torch.dtype,
    payload_dtype: torch.dtype,
    entries: int,
    width: int,
) -> torch.Tensor:
    if str(cache.tag) != tag:
        raise RuntimeError(
            f"QSA cache tag mismatch: expected {tag!r}, got {cache.tag!r}"
        )
    return payload_view(
        cache.kv_cache_base,
        CachePayloadGeometry(storage_dtype, payload_dtype, entries, width),
        f"QSA cache {tag!r}",
    )


def _block_table(
    inputs: PyAttentionInputs,
    name: str,
    *,
    tag: str,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    table = getattr(inputs, name, None)
    if table is None or table.numel() == 0:
        raise RuntimeError(f"QSA cache {tag!r} requires {name}")
    if table.dim() != 2 or int(table.shape[0]) != batch_size:
        raise RuntimeError(
            f"QSA cache {tag!r} {name} must be [B, max_blocks], got {table.shape}"
        )
    if table.dtype != torch.int32:
        raise RuntimeError(f"QSA cache {tag!r} {name} must be int32, got {table.dtype}")
    if table.device != device:
        raise RuntimeError(
            f"QSA cache {tag!r} {name} is on {table.device}, expected {device}"
        )
    return table


def _validate_required_blocks(
    table: torch.Tensor,
    required_columns: torch.Tensor,
    *,
    pool_blocks: int,
    tag: str,
) -> None:
    """Validate every table entry a paged reader may observe."""
    if required_columns.dim() != 1 or int(required_columns.numel()) != int(
        table.shape[0]
    ):
        raise RuntimeError(f"QSA cache {tag!r} has invalid required-column geometry")
    required_columns = required_columns.to(device=table.device)
    if table.is_cuda and torch.version.hip is None:
        from rtp_llm.models_py.modules.qwen4_exp.qsa.qsa_validation_triton import (
            is_supported,
            required_blocks_are_valid,
        )

        if is_supported(table, required_columns) and required_blocks_are_valid(
            table, required_columns, pool_blocks
        ):
            return
    columns = torch.arange(int(table.shape[1]), device=table.device).unsqueeze(0)
    required = columns < required_columns.unsqueeze(1)
    negative_columns = torch.any(required_columns < 0)
    uncovered_columns = torch.any(required_columns > int(table.shape[1]))
    invalid_required = torch.any(required & ((table <= 0) | (table >= pool_blocks)))
    invalid_physical = torch.any(table >= pool_blocks)
    # Synchronize once on the normal path.  Keep the specific diagnostics on
    # the rare error path, where the extra synchronizations do not affect TPOT.
    if bool(
        (
            negative_columns | uncovered_columns | invalid_required | invalid_physical
        ).item()
    ):
        if bool(negative_columns.item()):
            raise RuntimeError(
                f"QSA cache {tag!r} has a negative required-column count"
            )
        if bool(uncovered_columns.item()):
            raise RuntimeError(
                f"QSA cache {tag!r} block table does not cover the target-verify tail"
            )
        if bool(invalid_required.item()):
            raise RuntimeError(
                f"QSA cache {tag!r} target-verify tail resolves to an unallocated "
                "or out-of-range physical block"
            )
        raise RuntimeError(f"QSA cache {tag!r} contains an out-of-range physical block")
