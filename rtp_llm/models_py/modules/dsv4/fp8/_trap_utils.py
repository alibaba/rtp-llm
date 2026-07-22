"""Validation helpers for DSV4 FP8 invalid KV/state cache access."""

from __future__ import annotations

import os

DSV4_VALIDATE_INVALID_KV_ACCESS_ENV = "DSV4_VALIDATE_INVALID_KV_ACCESS"
DSV4_INVALID_KV_ACCESS_DUMP_LIMIT_ENV = "DSV4_INVALID_KV_ACCESS_DUMP_LIMIT"


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in ("0", "false", "off", "no")


def trap_invalid_kv_access_enabled() -> bool:
    """Keep device-side invalid-access traps enabled in every kernel specialization."""
    return True


def invalid_kv_access_validation_enabled() -> bool:
    # This diagnostic performs tensor scans and GPU-to-CPU synchronization at
    # every call site, so keep it opt-in rather than part of the serving path.
    return _env_flag(DSV4_VALIDATE_INVALID_KV_ACCESS_ENV, False)


def invalid_kv_access_dump_limit() -> int:
    value = os.environ.get(DSV4_INVALID_KV_ACCESS_DUMP_LIMIT_ENV)
    if value is None:
        return 8
    try:
        return max(1, int(value))
    except ValueError:
        return 8


def validate_slot_mapping(
    site: str,
    slot_mapping,
    *,
    block_size: int,
    num_blocks: int,
    negative_mode: str,
) -> None:
    if not invalid_kv_access_validation_enabled():
        return

    slots = slot_mapping.detach().reshape(-1).long()
    capacity = int(block_size) * int(num_blocks)
    invalid = slots >= capacity
    if negative_mode == "skip_any":
        pass
    elif negative_mode == "skip_minus_one":
        invalid = invalid | (slots < -1)
    elif negative_mode == "invalid":
        invalid = invalid | (slots < 0)
    else:
        raise ValueError(f"unknown negative_mode={negative_mode}")

    if not bool(invalid.any().item()):
        return

    limit = invalid_kv_access_dump_limit()
    bad_idx = invalid.nonzero(as_tuple=False).flatten()[:limit]
    bad_slots = slots[bad_idx].detach().cpu().tolist()
    bad_idx_cpu = bad_idx.detach().cpu().tolist()
    raise RuntimeError(
        f"DSV4 invalid KV access precheck failed at {site}: "
        f"capacity={capacity} block_size={block_size} num_blocks={num_blocks} "
        f"negative_mode={negative_mode} sample_indices={bad_idx_cpu} "
        f"sample_slots={bad_slots} slot_shape={tuple(slot_mapping.shape)}"
    )


def validate_block_table_lookup(
    site: str,
    block_table,
    req_indices,
    block_indices,
    use_mask,
    *,
    num_blocks: int,
) -> None:
    if not invalid_kv_access_validation_enabled():
        return

    req = req_indices.detach().reshape(-1).long()
    blk_idx = block_indices.detach().reshape(-1).long()
    mask = use_mask.detach().reshape(-1).bool()
    if not bool(mask.any().item()):
        return

    rows = int(block_table.shape[0])
    cols = int(block_table.shape[1])
    invalid_index = mask & (
        (req < 0) | (req >= rows) | (blk_idx < 0) | (blk_idx >= cols)
    )
    safe_req = req.clamp(0, max(rows - 1, 0))
    safe_blk_idx = blk_idx.clamp(0, max(cols - 1, 0))
    table_vals = block_table[safe_req, safe_blk_idx]
    invalid_value = (
        mask & ~invalid_index & ((table_vals < 0) | (table_vals >= num_blocks))
    )
    invalid = invalid_index | invalid_value
    if not bool(invalid.any().item()):
        return

    limit = invalid_kv_access_dump_limit()
    bad = invalid.nonzero(as_tuple=False).flatten()[:limit]
    raise RuntimeError(
        f"DSV4 invalid KV access precheck failed at {site}: "
        f"block_table_shape={tuple(block_table.shape)} num_blocks={num_blocks} "
        f"sample_flat_indices={bad.detach().cpu().tolist()} "
        f"sample_req={req[bad].detach().cpu().tolist()} "
        f"sample_block_indices={blk_idx[bad].detach().cpu().tolist()} "
        f"sample_block_values={table_vals[bad].detach().cpu().tolist()}"
    )
