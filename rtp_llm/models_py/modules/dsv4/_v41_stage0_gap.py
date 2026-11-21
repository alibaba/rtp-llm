# SPDX-License-Identifier: Apache-2.0
"""Stage-0 gap diagnosis — Python-side silent span recorder (DSV41_STAGE0_GAP_AUDIT,
default off). Records the Python prepare/execute boundaries with the SAME
CLOCK_MONOTONIC clock as the C++ recorder (time.monotonic_ns). Preallocated list,
no formatting/printing/JSON on the measured path; flushed at exit to a binary file.

Every record carries the REAL native identity (request_id, generation,
cache_incarnation, chunk_id) read from the admitted inputs' v41_cache_spans row
(CPU int64, host-side read only) or the dispatch's span.wire_values() — never a
clock/forward-counter derivation or a synthetic per-process counter. This stays
correct for overlapping or repeated request ids because the admitted generation /
cache_incarnation / chunk_id disambiguate them. The identity is identical on all
eight ranks because the admitted inputs are synced.
"""

import atexit
import os
import struct
import threading
import time

_MODE = os.environ.get("DSV41_STAGE0_GAP_AUDIT", "off")
_ENV_ENABLED = _MODE in ("cpu", "cpu_gpu")
_DIR = os.environ.get("DSV41_STAGE0_GAP_AUDIT_DIR", "")

_LATCHED_MODE = [None]  # None -> use env; else the latched mode string


def set_mode(mode):
    """Controller-facing per-request mode latch (off/cpu/cpu_gpu). Set once per
    request at the boundary; never changed mid-request."""
    _LATCHED_MODE[0] = mode


def _mode() -> str:
    return _LATCHED_MODE[0] if _LATCHED_MODE[0] is not None else _MODE


# Per-request mode resolution. The mode for a record is ALWAYS derived from the
# record's own request id (the incoming admitted request), never from a stale
# in-flight slot. _rid_mode_map / _mode_for_rid key on the real request_id.
def _mode_for_rid(rid) -> str:
    m = _rid_mode_map()
    if m:
        return m.get(rid, "off")
    return _mode()


def _rid_on(rid) -> bool:
    return _mode_for_rid(rid) in ("cpu", "cpu_gpu")


def _active_globally() -> bool:
    """Fast default-off gate: a per-request ridmap is present, or the global /
    latched mode records. When this is False the recorder is a no-op that does
    NOT read the incoming identity or touch any state (fast default-off path)."""
    return bool(_rid_mode_map()) or (_mode() in ("cpu", "cpu_gpu"))


# kinds (Python-side): keep distinct from the C++ Stage0GapEvent values
PY_PREPARE_ENTER = 100
PY_PREPARE_EXIT = 101
PY_EXECUTE_ENTER = 102
PY_EXECUTE_EXIT = 103
# subregion markers (enter/exit pairs) within the prepare span
PY_AGREE_ENTER = 110  # _agree_dp collective
PY_AGREE_EXIT = 111
PY_DISPATCH_ENTER = 112  # whole prepare_native_pp_prefill_dispatch
PY_DISPATCH_EXIT = 113
PY_D_LEASES_ENTER = 114  # native_allocation_tables + build_block_tables_batched
PY_D_LEASES_EXIT = 115
PY_D_CP_ENTER = 116  # cp_fields clone + build_cp_context_for_forward + geometry
PY_D_CP_EXIT = 117
PY_D_TAGS_ENTER = 118  # per-tag common-field + CP-field + boundary checks
PY_D_TAGS_EXIT = 119
PY_D_FLUSH_ENTER = 120  # _pp_flush_preflight_checks (grouped deferred flush)
PY_D_FLUSH_EXIT = 121
PY_D_ROWMAP_ENTER = 122  # final row-map torch.stack(...).cpu() D2H
PY_D_ROWMAP_EXIT = 123

_MODE_INT = {"off": 0, "cpu": 1, "cpu_gpu": 2}

# Per-request recorder mode map (DSV41_STAGE0_GAP_AUDIT_RIDMAP, default absent):
# "rid mode" lines (mode in off/cpu/cpu_gpu), read once and cached. When present the
# recorder is active only for rids whose mapped mode records — a within-worker
# on/off contrast without any per-request fence.
_RID_MODE_CACHE = None


def _rid_mode_map():
    global _RID_MODE_CACHE
    if _RID_MODE_CACHE is None:
        m = {}
        path = os.environ.get("DSV41_STAGE0_GAP_AUDIT_RIDMAP", "")
        if path:
            try:
                with open(path) as f:
                    for line in f:
                        parts = line.split()
                        if len(parts) == 2:
                            m[int(parts[0])] = parts[1]
            except OSError:
                m = {}
        _RID_MODE_CACHE = m
    return _RID_MODE_CACHE


# per-thread in-flight identity (single executor thread per rank). The identity is
# the REAL admitted span identity, set on every prepare/execute entry.
_tls = threading.local()
_records = []  # list of tuples; flushed at exit


def enabled() -> bool:
    return _ENV_ENABLED or (_LATCHED_MODE[0] in ("cpu", "cpu_gpu"))


def _ns() -> int:
    return time.monotonic_ns()


def _cur():
    """In-flight (request_id, chunk_id, generation, cache_incarnation) for THIS thread."""
    return getattr(_tls, "cur", (-1, -1, 0, 0))


def _set_cur(request_id, chunk_id, generation, cache_incarnation):
    _tls.cur = (request_id, chunk_id, generation, cache_incarnation)


def _record(kind, request_id, chunk_id, generation, cache_incarnation):
    # gate on the RECORD's OWN request id (incoming request), not a stale in-flight slot
    if not _rid_on(request_id):
        return
    _records.append(
        (kind, _ns(), threading.get_native_id(), request_id, chunk_id, generation,
         cache_incarnation, _MODE_INT.get(_mode_for_rid(request_id), 0))
    )


def record(kind: int) -> None:
    """Identity-free record (subregion markers); carries this thread's in-flight identity."""
    r, c, g, ci = _cur()
    _record(kind, r, c, g, ci)


def _span_identity_from_inputs(inputs):
    """(request_id, generation, cache_incarnation, chunk_id) from the admitted inputs'
    v41_cache_spans row 0 (CPU int64, host-side read only). (-1,-1,0,0) if unavailable."""
    try:
        wire = getattr(inputs, "v41_cache_spans", None)
        if wire is not None and wire.numel() >= 6:
            row = wire.reshape(-1, 6)[0].tolist()
            # wire row = (request_id, generation, cache_incarnation, chunk_id, start, count)
            return int(row[0]), int(row[3]), int(row[1]), int(row[2])
    except Exception:
        pass
    return -1, -1, 0, 0


def _span_identity_from_dispatch(dispatch):
    """(request_id, generation, cache_incarnation, chunk_id) from the dispatch's first span."""
    try:
        spans = getattr(dispatch, "spans", None)
        if spans:
            w = spans[0].wire_values()
            # wire_values = (request_id, generation, cache_incarnation, chunk_id, start, count)
            return int(w[0]), int(w[3]), int(w[1]), int(w[2])
    except Exception:
        pass
    return -1, -1, 0, 0


def prepare_entry(inputs) -> None:
    if not _active_globally():
        return  # fast default-off path: no identity read, no state mutation
    rid, chunk, gen, ci = _span_identity_from_inputs(inputs)  # read the INCOMING identity first
    _set_cur(rid, chunk, gen, ci)  # always bind (even off) so nested markers are scoped
    _record(PY_PREPARE_ENTER, rid, chunk, gen, ci)  # gated on the incoming rid's mode


def prepare_exit() -> None:
    if not _active_globally():
        return  # fast default-off path
    r, c, g, ci = _cur()
    _record(PY_PREPARE_EXIT, r, c, g, ci)


def execute_entry(dispatch) -> None:
    if not _active_globally():
        return  # fast default-off path
    rid, chunk, gen, ci = _span_identity_from_dispatch(dispatch)  # bind execute to the dispatch identity
    _set_cur(rid, chunk, gen, ci)
    _record(PY_EXECUTE_ENTER, rid, chunk, gen, ci)  # gated on the dispatch rid's mode


def execute_exit() -> None:
    if not _active_globally():
        return  # fast default-off path
    r, c, g, ci = _cur()
    _record(PY_EXECUTE_EXIT, r, c, g, ci)


def _flush() -> None:
    if not _DIR or not _records:
        return
    path = os.path.join(_DIR, f"stage0-gap-py-{os.getpid()}.bin")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(_records)))
        for kind, ns, tid, rid, chunk, gen, ci, mode in _records:
            f.write(struct.pack("<IQq q i i i i", kind, tid, ns, rid, chunk, gen, ci, mode))


atexit.register(_flush)
