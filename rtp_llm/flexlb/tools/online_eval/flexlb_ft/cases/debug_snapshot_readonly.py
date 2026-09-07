"""Independent status case; the case-registry owner wires CASE_DEF into the suite."""

import json
import time
from dataclasses import replace

from ..context import CaseDef, rid_base
from ..debug_client import DebugClient, DebugUnavailable, check_scheduler_tombstone


def master_debug_snapshot(ctx):
    """One completed cohort member must remain queryable as a resource-free tombstone."""
    spec = replace(
        ctx.smoke_spec(),
        label=f"debug_snapshot_{ctx.profile}",
        master_env={"FLEXLB_DEBUG_ENABLED": "true"},
    )
    env = ctx.env_manager.ensure(spec)
    ops = ctx.engine_ops(env)
    client = DebugClient(f"http://127.0.0.1:{env.master_http_port}")
    artifact = ctx.case_dir("master_debug_snapshot") / "master_debug_snapshots.ndjson"
    try:
        before = client.snapshot()
        before.component(
            "scheduler"
        )  # Required source, including schema/coverage validation.
        instance = before.payload["instanceId"]
        rid = ops.next_request_id(rid_base(ctx, "status"))
        _, error = ops.run_one_request(rid, output_len=2, typed_stream_error=True)
        if error is not None:
            return False, f"nonempty cohort request failed: {error}"
        deadline = time.monotonic() + 15.0
        with artifact.open("w", encoding="utf-8") as output:
            while time.monotonic() < deadline:
                client.timeout_s = min(5.0, max(0.001, deadline - time.monotonic()))
                capture = client.snapshot(request_id=rid)
                output.write(
                    json.dumps(
                        {
                            "source": "master_debug",
                            "request_id": str(rid),
                            "started_mono": capture.started_mono,
                            "finished_mono": capture.finished_mono,
                            "snapshot": capture.payload,
                        }
                    )
                    + "\n"
                )
                if capture.payload["instanceId"] != instance:
                    raise DebugUnavailable("Master instance changed during observation")
                rows = capture.component("scheduler")["rows"]
                if len(rows) != 1 or rows[0]["request_id"] != str(rid):
                    raise DebugUnavailable("issued cohort member is not queryable")
                if rows[0]["storage_phase"] == "TOMBSTONE":
                    passed, detail = check_scheduler_tombstone(rows[0])
                    return passed, f"rid={rid}: {detail}; evidence={artifact}"
                time.sleep(min(0.2, max(0, deadline - time.monotonic())))
        return (
            False,
            f"scheduler tombstone not observed before deadline; evidence={artifact}",
        )
    except DebugUnavailable as error:
        # Legacy CaseDef's boolean contract: unavailable evidence cannot return True.
        return False, f"ERROR debug observation: {error}; evidence={artifact}"


CASE_DEF = CaseDef(
    "master_debug_snapshot",
    "status",
    master_debug_snapshot,
    profiles=["batch-window"],
    requires=["enqueue_batch"],
    source="debug_snapshot_readonly.py",
)
