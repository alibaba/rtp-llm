"""R25/O1: mHC stats side-stream policy + resources (OSS-HINTS-20261030 §2).

Flag ``DSV41_MHC_STATS_SIDE_STREAM`` (default "1" since R27 adoption;
"0" = escape: every kernel stays on
the current stream, exactly as today). Under "1", ``DelayedHCUnit._pre_impl``
issues the stats kernels — prenorm/``_mhc_pre_fn_mul``, ``mhc_pre_split_mixes``,
``sinkhorn_normalize`` and the ``pre_mix_out`` materialization (~90 µs per
boundary, 40 boundaries/chunk) — on a shared per-device side stream and
event-joins them back to the main stream before every consumer:
``*_hc_post`` (post/comb), ``DelayedHCHead.head`` and the PP export of
``pre_mix`` (``pack_v41_hidden_frame``). The collapse stays on the main
stream: it consumes the PREVIOUS boundary's pre_mix, which the previous
post already joined.

Byte-exact by construction: same kernels, same inputs, same op order —
only the issuing stream changes, with event edges reproducing the serial
order at every consumption point. Allocator safety: outputs are recorded
onto the main stream (record_stream) and Python references are retained
by the unit until the post consumes them; the input residual is recorded
onto the side stream.

Optional diagnostics ``DSV41_MHC_SIDE_STATS=1`` (verify arms only; adds
event elapsed-time work): reports the join-already-done rate — the direct
overlap proof without nsys (CUPTI is blocked on this stack, R20) — plus
mean side-region GPU busy per boundary.
"""

import os

import torch

MHC_SIDE_STREAM_FLAG = "DSV41_MHC_STATS_SIDE_STREAM"
MHC_SIDE_STATS_FLAG = "DSV41_MHC_SIDE_STATS"

_STREAMS = {}
_ANNOUNCED = [False]
_STATS = {"joins": 0, "already_done": 0, "busy_ms": 0.0, "timed": 0}


def mhc_side_stream_enabled() -> bool:
    # R27: default flipped to "1" (adopted; R25 evidence: stack -42.4 ms,
    # 3/3, byte-exact). Escape: DSV41_MHC_STATS_SIDE_STREAM=0.
    return os.environ.get(MHC_SIDE_STREAM_FLAG, "1") == "1"


def mhc_side_stats_enabled() -> bool:
    return os.environ.get(MHC_SIDE_STATS_FLAG, "0") == "1"


def mhc_side_stream(device):
    """Shared per-device side stream, created on first use."""
    key = (device.type, device.index)
    stream = _STREAMS.get(key)
    if stream is None:
        stream = torch.cuda.Stream(device=device)
        _STREAMS[key] = stream
    return stream


def mhc_side_announce():
    if not _ANNOUNCED[0]:
        _ANNOUNCED[0] = True
        import sys

        print(
            f"[MHCSIDE] engaged ({MHC_SIDE_STREAM_FLAG}=1): stats on side stream, "
            "event-joined before hc_post / head / PP export",
            file=sys.stderr,
            flush=True,
        )


def mhc_side_note_join(already_done: bool, busy_ms: float = None):
    """Accumulate overlap diagnostics; print every 512 joins (stats flag only)."""
    _STATS["joins"] += 1
    if already_done:
        _STATS["already_done"] += 1
    if busy_ms is not None:
        _STATS["busy_ms"] += busy_ms
        _STATS["timed"] += 1
    if _STATS["joins"] % 512 == 0:
        import sys

        pct = 100.0 * _STATS["already_done"] / max(_STATS["joins"], 1)
        busy = _STATS["busy_ms"] / max(_STATS["timed"], 1) * 1000.0
        print(
            "[MHCSIDE-STATS] joins=%d already_done=%.1f%% side_busy_avg_us=%.1f"
            % (_STATS["joins"], pct, busy),
            file=sys.stderr,
            flush=True,
        )
