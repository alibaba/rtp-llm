"""Strict QSA byte payload, metadata and block geometry validation."""

from __future__ import annotations

import torch

from rtp_llm.models_py.modules.qwen4_exp.qsa.validation import DeviceChecks
from rtp_llm.models_py.utils.cache_payload_view import (
    CachePayloadGeometry,
    payload_view,
    same_tensor_metadata,
)
from rtp_llm.ops.compute_ops import LayerKVCache, PyAttentionInputs

_same_tensor_metadata = same_tensor_metadata


def _validate_state_pairs(table, pairs, pool_blocks, label, checks=None):
    if not pairs:
        return
    # Resolve the small tail of the whole batch in one device operation.
    indices = torch.tensor(pairs, device="cpu", dtype=torch.long, pin_memory=True)
    indices = indices.to(device=table.device, non_blocking=True)
    blocks = table[indices[:, 0], indices[:, 1]]
    own_checks = checks is None
    checks = DeviceChecks() if checks is None else checks
    checks.add(
        ((blocks <= 0) | (blocks >= pool_blocks)).any(),
        f"QSA cache 'indexer_state' {label} resolves to "
        "an unallocated or out-of-range physical block",
    )
    if own_checks:
        checks.verify()


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
    checks=None,
) -> None:
    """Validate every table entry a paged reader may observe."""
    if required_columns.dim() != 1 or int(required_columns.numel()) != int(
        table.shape[0]
    ):
        raise RuntimeError(f"QSA cache {tag!r} has invalid required-column geometry")
    required_columns = required_columns.to(device=table.device)
    own_checks = checks is None
    checks = DeviceChecks() if checks is None else checks
    flags = None
    if table.is_cuda and torch.version.hip is None:
        from rtp_llm.models_py.modules.qwen4_exp.qsa.qsa_validation_triton import (
            is_supported,
            required_block_flags,
        )

        if is_supported(table, required_columns):
            flags = required_block_flags(table, required_columns, pool_blocks)
    if flags is not None:
        invalid = None
    else:
        columns = torch.arange(int(table.shape[1]), device=table.device).unsqueeze(0)
        required = columns < required_columns.unsqueeze(1)
        invalid = [
            (required_columns < 0).any(),
            (required_columns > int(table.shape[1])).any(),
            (required & ((table <= 0) | (table >= pool_blocks))).any(),
            (table >= pool_blocks).any(),
        ]
    messages = [
        f"QSA cache {tag!r} has a negative required-column count",
        f"QSA cache {tag!r} block table does not cover the target-verify tail",
        f"QSA cache {tag!r} target-verify tail resolves to an unallocated or out-of-range physical block",
        f"QSA cache {tag!r} contains an out-of-range physical block",
    ]
    if flags is not None:
        checks.add_flags(flags, messages)
    else:
        for condition, message in zip(invalid, messages):
            checks.add(condition, message)
    if own_checks:
        checks.verify()
