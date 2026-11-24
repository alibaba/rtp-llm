"""Shared gang bookkeeping, without a second execution state machine."""

import copy
from datetime import datetime, timezone

from rtp_llm.test.perf_test.state import RunState

STAGE_ORDER = {
    name: index
    for index, name in enumerate(("rendezvous", "build", "submit", "collect", "report"))
}


def write_manifest(
    store, manifest, *, state=None, stage=None, status=None, details=None
):
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    if state is not None:
        manifest["state"] = state.value
    if stage is not None:
        payload = {"status": status, "updated_at": now}
        payload.update(copy.deepcopy(dict(details or {})))
        manifest.setdefault("stages", {})[stage] = payload
    manifest["updated_at"] = now
    store.write_manifest(manifest)


def record_stage(store, manifest, *, stage, status, state=None, details=None):
    if stage not in STAGE_ORDER:
        raise ValueError(f"unknown gang stage: {stage}")
    if status not in {"running", "completed", "failed", "partial", "skipped"}:
        raise ValueError(f"unknown stage status: {status}")
    write_manifest(
        store,
        manifest,
        state=state,
        stage=stage,
        status=status,
        details={**(details or {}), "stage_index": STAGE_ORDER[stage]},
    )


def reconcile_terminal_state(
    master_state, submit_failure=None, collection_failure=None
):
    """A successful payload never overrides an observed execution failure."""
    if master_state == RunState.INTERRUPTED:
        return RunState.INTERRUPTED
    if submit_failure is not None or collection_failure is not None:
        return RunState.FAILED
    return master_state or RunState.FAILED
