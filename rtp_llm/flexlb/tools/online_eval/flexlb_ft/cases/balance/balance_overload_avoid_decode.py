from __future__ import annotations

import json
import time

from ...context import CaseContext, rid_base
from ...grade import GradeReport
from ...registry import case
from ...support.balance import STREAM_TIMEOUT_S, _decode_names


@case(
    "balance_overload_avoid_decode",
    category="balance",
    source="scheduling_smoke.py S11 (strengthened)",
)
def balance_overload_avoid_decode(ctx: CaseContext):
    """Decode KV exhaustion: the pressured engine stops taking new work and
    the healthy engines absorb the traffic.

    Result properties: P5 overload-avoidance in the *delta caliber* (graded:
    how many of the n requests still complete on the KV-exhausted engine),
    P6 completeness, P2 no-starvation takeover assertions.

    P5 band note (case override, absolute-delta caliber): the global P5
    share bands translate awkwardly to 10 samples (0.05*10 = 0.5); the
    delta bands strict=0 / normal=1 / loose=2 carry the same intent with
    the historical calibration that exactly one straggler request can
    already be in prefill→decode handoff when the pressure snapshot lands
    (the legacy S11 delta<=1 was the stable-pass baseline).

    Takeover strengthening (vs legacy S11, which only bounded the target
    delta): the non-pressured decode engines must actually absorb the
    diverted load — at least two of them take requests, and every one of
    the n requests completes somewhere (no loss).
    """
    ops = ctx.ops()
    report = GradeReport(run_grade=ctx.grade)
    injected: str | None = None
    try:
        decode_names = _decode_names(ops)
        if len(decode_names) < 2:
            return False, "need >=2 decode workers"
        target = decode_names[0]
        info = ops.snapshot_by_name()[target]
        total_kv = int(info.get("available_kv_tokens", 0)) + int(
            info.get("active_kv_tokens", 0)
        )
        ops.set_kv_pressure(target, total_kv)  # available -> 0
        injected = target
        time.sleep(1.0)  # master worker-status sync

        snap_sync = ops.snapshot_by_name()
        completed_before = {
            name: snap_sync.get(name, {}).get("completed", 0) for name in decode_names
        }

        n = 10
        failures = []
        for _ in range(n):
            rid = ops.next_request_id(rid_base(ctx, "balance"))
            keys = [rid * 100 + j for j in range(3)]
            _, err = ops.run_one_request(
                rid, output_len=2, block_keys=keys, stream_timeout_s=STREAM_TIMEOUT_S
            )
            if err:
                failures.append(f"rid={rid}: {err}")

        snap2 = ops.snapshot_by_name()
        deltas = {
            name: snap2[name].get("completed", 0) - completed_before.get(name, 0)
            for name in decode_names
        }
        target_delta = deltas.get(target, 0)
        others = {name: d for name, d in deltas.items() if name != target}
        others_used = sum(1 for v in others.values() if v > 0)
        others_total = sum(others.values())

        # P6: every request completed somewhere — no loss under pressure.
        report.invariant(
            "P6",
            not failures and others_total + target_delta >= n,
            detail=f"failures={failures[:2]}, total_delta={others_total + target_delta}/{n}",
        )
        # P5: hot-engine delta caliber (graded, case override — see docstring).
        report.check(
            "P5",
            float(target_delta),
            context="decode_kv_pressure",
            bands={"strict": 0.0, "normal": 1.0, "loose": 2.0},
            detail=f"target={target}(delta={target_delta}), "
            f"deltas={json.dumps(deltas, sort_keys=True)}",
        )
        # P2: takeover — the diverted load actually lands on the healthy
        # engines (>=2 of them used), i.e. nobody is starved by the pressure.
        report.invariant(
            "P2",
            others_used >= 2 and others_total >= n - target_delta,
            context="decode_takeover",
            detail=f"others_used={others_used}, others_total={others_total}",
        )

        return report.finish(
            f"target={target}(delta={target_delta}), others_used={others_used}, "
            f"grades: {report.summary()}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        if injected:
            try:
                ops.set_kv_pressure(injected, 0)
            except Exception:
                pass
