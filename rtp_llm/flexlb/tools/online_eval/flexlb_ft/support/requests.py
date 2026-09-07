"""Shared request lifecycle for KV and balance scenarios.

Batch dispatch defers Fetch until drain; NON_BATCH starts its stream at fire.
"""

import time


def fire_request(ops, rid: int, fired: list, fired_handles: dict, **kwargs):
    """Schedule without consuming the stream — the ledger entry stays
    live until the case drain.  Returns (engine_name, error).  Under
    NON_BATCH dispatch the engine only sees the request when the CLIENT
    opens the stream, so the direct stream is opened fire-and-forget."""
    try:
        resp = ops.schedule(rid, **kwargs)
    except Exception as exc:
        return None, repr(exc)
    if resp.code != 200 or not resp.success:
        return None, f"schedule failed: {resp.error_message}"
    addr = ops.role_addr(resp, "PREFILL")
    name = ops.addr_to_name().get(addr, addr)
    fired.append((rid, resp))
    if not resp.enqueued_by_master:
        try:
            input_pb = ops.build_generate_input(rid, **kwargs)
            fired_handles[rid] = ops.start_stream(resp, rid, input_pb=input_pb)
        except Exception as exc:
            return name, f"direct stream failed to open: {exc!r}"
    return name, None


def drain_fired(ops, fired: list, fired_handles: dict, wait_s: float = 30.0) -> list:
    """Consume every fired request to terminal state (cancel fallback).
    Returns [(rid, engine_name, completed, err)]."""
    outcomes = []
    for rid, resp in fired:
        name = ops.addr_to_name().get(ops.role_addr(resp, "PREFILL"), "")
        completed = False
        err = None
        try:
            handle = (
                fired_handles[rid]
                if rid in fired_handles
                else ops.start_stream(resp, rid)
            )
            ended = handle.wait_end(wait_s)
            completed = ended and handle.snap.completed and not handle.snap.error
            if not completed:
                err = handle.snap.error or "stream did not complete"
        except Exception as exc:
            err = repr(exc)
        if not completed:
            try:
                ops.cancel(rid, resp)
            except Exception:
                pass
        outcomes.append((rid, name, completed, err))
    return outcomes


def wait_engine_pending(
    ops, engine_name: str, min_pending: int, timeout_s: float = 6.0
) -> bool:
    """Engine-side proof that a fired request was really dispatched."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        info = ops.snapshot_by_name().get(engine_name, {})
        if info.get("waiting", 0) + info.get("running", 0) >= min_pending:
            return True
        time.sleep(0.1)
    return False
