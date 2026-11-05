"""R19/C diagnostic: per-rank MoE expert-count dump (DSV41_MOE_COUNT_DUMP).

Set DSV41_MOE_COUNT_DUMP=<dir> to record every contiguous-layout host count
read (already materialized for routing — no extra sync is introduced) as
(seq, expert_start_id, counts) entries, flushed to <dir>/counts-rank<R>-
pid<P>.jsonl at exit. Offline analysis derives the per-layer/per-chunk EP
load imbalance (max/avg across the 4 EP ranks) that feeds the #20 EPLB
redundant-expert design estimate. Diagnostic only: no behavior change, no
numerics involvement; default off (env absent -> the record call is a
single None compare).
"""

import atexit
import json
import os
import sys

_DUMP_DIR = None
_ENTRIES = None
_SEQ = 0
_ANNOUNCED = False
_DIR_RESOLVED = False


def _resolve_dir():
    global _DUMP_DIR, _DIR_RESOLVED
    if _DIR_RESOLVED:
        return _DUMP_DIR
    _DIR_RESOLVED = True
    value = os.environ.get("DSV41_MOE_COUNT_DUMP")
    if value:
        _DUMP_DIR = value
    return _DUMP_DIR


def count_dump_record(expert_start_id: int, counts) -> None:
    """Record one host-count read. ``counts`` is the already-materialized
    host tuple from the contiguous routing path."""
    global _SEQ, _ANNOUNCED, _ENTRIES
    if not _resolve_dir():
        return
    if _ENTRIES is None:
        _ENTRIES = []
        atexit.register(_flush)
    _SEQ += 1
    _ENTRIES.append((_SEQ, int(expert_start_id), list(counts)))
    if not _ANNOUNCED:
        _ANNOUNCED = True
        print(
            f"[COUNTDUMP] recording to {_DUMP_DIR} pid={os.getpid()} "
            f"rank={_resolve_rank()}",
            file=sys.stderr,
            flush=True,
        )


def _resolve_rank() -> str:
    rank = os.environ.get("RANK") or os.environ.get("LOCAL_RANK")
    if rank:
        return rank
    try:
        import torch

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return str(torch.distributed.get_rank())
    except Exception:
        pass
    return "na"


def _flush() -> None:
    if not _ENTRIES:
        return
    rank = _resolve_rank()
    try:
        os.makedirs(_DUMP_DIR, exist_ok=True)
        path = os.path.join(_DUMP_DIR, f"counts-rank{rank}-pid{os.getpid()}.jsonl")
        with open(path, "w") as handle:
            for seq, esi, counts in _ENTRIES:
                handle.write(json.dumps({"s": seq, "e": esi, "c": counts}) + "\n")
        print(
            f"[COUNTDUMP] wrote {len(_ENTRIES)} entries to {path}",
            file=sys.stderr,
            flush=True,
        )
    except OSError as exc:  # never fail a run on diagnostics
        print(f"[COUNTDUMP] flush failed: {exc}", file=sys.stderr, flush=True)


# The v41_moe call site imports this name directly.
_count_dump_record = count_dump_record
