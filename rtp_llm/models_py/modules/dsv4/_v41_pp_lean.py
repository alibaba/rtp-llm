"""Batch-3 A: PP per-chunk host-sync removal (flags default OFF).

Stage-1 runs ``prepare_native_pp_prefill_dispatch`` once per chunk with
~20+ blocking host reads: 5 zigzag-wire ``.cpu().tolist()`` field compares
(three of them 4096-element Python list comparisons), ~12 ``_same_common``
``torch.equal(value.cpu(), other.cpu())`` reads, cu/boundary/length
``torch.equal``s, masked-position ``torch.equal`` (boolean-mask indexing
also runs an implicit nonzero sync), engram ``bool(...)`` reads, and in
``v41_stage_pp`` the row_map build (3 more ``.cpu().tolist()`` + a
1024-iteration Python loop + 24KB H2D). These populate the diffuse
4B/8B/1B DtoH clusters of the R9 attribution on the cadence-setting stage.

``DSV41_PP_ASYNC_PREFLIGHT`` (A1, default ON since Batch-4a decision 1;
``=0`` escape hatch) converts the COMPARISONS to device
predicates routed through the G3 deferred-check machinery (grouped flush
when enabled, immediate ``_assert_async`` otherwise), plus one explicit
``flush_checks("pp_preflight")`` at the end of the dispatch build. Host
structural checks (type/dtype/shape/contiguity/device) still raise
synchronously. Checks-only ⇒ byte-exact outputs on valid runs; a failure
becomes a device-side assert at the flush instead of an immediate host
ValueError — the same accepted trade-off as G3/P2c (both in the BASE
config). Expected wire tensors are cached module-level keyed by
(name, values-tuple, device): the zigzag expectation pattern is
chunk-invariant for steady geometry, so H2Ds are paid once.

``DSV41_PP_DEVICE_ROW_MAP`` (A2, default ON since Batch-4a decision 1;
``=0`` escape hatch) builds the ``v41_stage_pp`` row_map
with a device lookup-gather + where instead of the host loop. Values are
identical int64: valid rows ``[index_by_request[req], position, 1]``,
invalid rows ``[-1, -1, 0]`` (invalid rows' request ids are clamped for
the gather and then discarded by the where — the reference never indexes
them either).
"""

import os

import torch

from rtp_llm.models_py.modules.dsv4._v41_deferred_checks import (
    add_check as _add_deferred_check,
)
from rtp_llm.models_py.modules.dsv4._v41_deferred_checks import (
    flush_checks as _flush_deferred_checks,
)
from rtp_llm.models_py.modules.dsv4._v41_deferred_checks import (
    grouped_asserts_enabled as _grouped_asserts_enabled,
)

_WIRE_CACHE: dict = {}
_ROW_LOOKUP_CACHE: dict = {}
_ANNOUNCED: set = set()


def _announce_once(tag: str) -> None:
    if tag not in _ANNOUNCED:
        _ANNOUNCED.add(tag)
        import sys

        print(f"[PPLEAN] {tag} engaged", file=sys.stderr, flush=True)


def pp_async_preflight_enabled() -> bool:
    """Default ON since Batch-4a decision 1 (byte-exact; A/B pool
    median −8.8 ms, 2/3). ``=0`` (any non-1) restores the blocking reads."""
    return os.environ.get("DSV41_PP_ASYNC_PREFLIGHT", "1") == "1"


def pp_device_rowmap_enabled() -> bool:
    """Default ON since Batch-4a decision 1 (byte-exact; adopted with
    A1). ``=0`` (any non-1) restores the host-loop row_map build."""
    return os.environ.get("DSV41_PP_DEVICE_ROW_MAP", "1") == "1"


def deferred_check(valid: torch.Tensor, message: str) -> None:
    """Device predicate check via the G3 group when available.

    CPU tensors keep the synchronous raise (reference semantics).
    """
    _announce_once("async_preflight")
    if not valid.is_cuda:
        if not bool(valid):
            raise ValueError(message)
        return
    if _grouped_asserts_enabled() and not torch.cuda.is_current_stream_capturing():
        _add_deferred_check(valid, message)
    else:
        torch._assert_async(valid, message)


def flush_preflight_checks() -> None:
    """Tighten the gating window: one stacked assert at preflight end."""
    if pp_async_preflight_enabled():
        _flush_deferred_checks("pp_preflight")


def expected_wire_tensor(name: str, values, device) -> torch.Tensor:
    key = (name, tuple(values), str(device))
    tensor = _WIRE_CACHE.get(key)
    if tensor is None:
        tensor = torch.tensor(list(values), dtype=torch.int32, device=device)
        _WIRE_CACHE[key] = tensor
    return tensor


def check_wire_field(name: str, value: torch.Tensor, values) -> None:
    """Async variant of ``value.cpu().tolist() != values``."""
    expected = expected_wire_tensor(name, values, value.device)
    deferred_check(
        (value == expected).all(),
        f"V4.1 native CP {name} disagrees with the zigzag dispatch",
    )


def _row_lookup(index_by_request: dict, device) -> torch.Tensor:
    key = (tuple(sorted(index_by_request.items())), str(device))
    lookup = _ROW_LOOKUP_CACHE.get(key)
    if lookup is None:
        size = max(index_by_request) + 1 if index_by_request else 1
        lookup = torch.full((int(size),), -1, dtype=torch.long)
        for request, index in index_by_request.items():
            lookup[int(request)] = int(index)
        lookup = lookup.to(device)
        _ROW_LOOKUP_CACHE[key] = lookup
    return lookup


def build_row_map(
    request_ids: torch.Tensor,
    positions: torch.Tensor,
    valid_rows: torch.Tensor,
    index_by_request: dict,
) -> torch.Tensor:
    """Device-built equivalent of the reference list-comprehension row_map."""
    _announce_once("device_row_map")
    device = request_ids.device
    lookup = _row_lookup(index_by_request, device)
    rid = request_ids.long()
    in_range = (rid >= 0) & (rid < lookup.numel())
    mapped = lookup[rid.clamp(0, lookup.numel() - 1)]
    # Fail-loud parity with the reference KeyError: a valid row whose request
    # id is unknown (out of lookup range, or an unmapped slot) must raise /
    # trip the deferred assert — the clamp only protects the gather.
    deferred_check(
        ((~valid_rows.bool()) | (in_range & (mapped >= 0))).all(),
        "V4.1 PP row_map references an unknown request id",
    )
    stacked = torch.stack((mapped, positions.long(), torch.ones_like(mapped)), dim=-1)
    invalid = torch.tensor([-1, -1, 0], dtype=torch.long, device=device)
    return torch.where(valid_rows.bool().unsqueeze(-1), stacked, invalid)
