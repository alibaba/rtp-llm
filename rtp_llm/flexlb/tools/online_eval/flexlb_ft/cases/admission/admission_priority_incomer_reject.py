from __future__ import annotations

import time

from ...context import CaseContext, rid_base
from ...engine_ops import engine_inflight_clean
from ...harness import AssertUtils, wait_for
from ...registry import case
from ...support.admission import (
    _decode_names,
    _incomer_spec,
    _master_http,
    _prefill_names,
    _schedule_with_priority,
)


@case(
    "admission_priority_incomer_reject",
    category="admission",
    profiles=["batch-window"],
    requires=["enqueue_batch"],
    source=(
        "admission wave-2 W3: PRIORITY incomer under new-B permit "
        "semantics — permit frees at DecodeAccepted, incomer admitted "
        "fast alongside the running victim (name kept for history)"
    ),
)
def admission_priority_incomer_reject(ctx: CaseContext):
    """PRIORITY incomer without preemption: admitted alongside the
    running victim (new-B permit semantics; the case name keeps its
    historical "_reject" suffix from the pre-intake3 contract).

    Scenario: dedicated 1P+1D env, PRIORITY ordering, NO preemption
    block (allowedVictimStages unset — EvictionManager.tryAdmit is a
    no-op) and lifecycle.maxDeliveredNotAcceptedRequestsGlobal=1, so
    exactly one acceptance permit exists.  A low-priority victim
    (priority 30, output_len=200 — decode runs ~1.5s) is scheduled
    first; once it is RUNNING on decode a higher-priority incomer
    (priority 70, output_len=2) arrives.

    Behaviour (new-B semantics): the decode acceptance permit is
    released at the DecodeAccepted EVENT, not held to the occupant's
    terminal — by the time the victim is observably RUNNING its permit
    is already back in the pool.  The incomer's route selection and
    permit acquisition both succeed: the Schedule RPC returns FAST
    (code 200) and the incomer executes ALONGSIDE the victim in a
    parallel decode slot.  With no preemption block nothing disturbs
    the victim either way.

    Expected (contract): the incomer's Schedule RPC returns FAST (< 3s)
    with code 200 and the incomer's stream completes normally; the
    victim is NOT preempted (its stream completes normally, no 8429
    anywhere) — the original queue is unaffected by the incomer; a
    fresh request succeeds (recovery); master inflight and engine
    ledgers drain clean.

    Prediction: measured contract (2026-09-04 run: incomer code=200 at
    0.02s, victim 2 outputs, clean ledgers) — the permit-release-at-
    DecodeAccepted semantics makes the old 8431 outcome unreachable at
    this probe point.  Complement of cancel_preemption_victim:
    preemption ON there (victim 8429, incomer wins) vs OFF here (victim
    lives, incomer admitted alongside).
    """
    env = ctx.env_manager.ensure(_incomer_spec(ctx))
    ops = ctx.engine_ops(env)
    base = rid_base(ctx, "admission")
    decode_engines = _decode_names(ops)
    if not decode_engines:
        return False, "no decode engines found"
    victim_handle = None
    incomer_handle = None
    try:
        # Victim takes the single admission permit (held until terminal).
        victim_rid = ops.next_request_id(base)
        victim_keys = [victim_rid * 100 + 1]
        victim_resp = _schedule_with_priority(
            ops,
            victim_rid,
            30,
            input_len=512,
            output_len=200,
            block_keys=victim_keys,
        )
        if victim_resp.code != 200 or not victim_resp.success:
            return False, (f"victim schedule failed: {victim_resp.error_message}")
        victim_input = (
            None
            if victim_resp.enqueued_by_master
            else ops.build_generate_input(
                victim_rid,
                input_len=512,
                output_len=200,
                block_keys=victim_keys,
            )
        )
        victim_handle = ops.start_stream(victim_resp, victim_rid, input_pb=victim_input)

        # Victim must be past prefill and RUNNING on decode before the
        # incomer arrives — otherwise the probe would race the transfer.
        def victim_running() -> bool:
            snap = ops.snapshot_by_name()
            return any(
                int(snap.get(n, {}).get("running", 0)) >= 1 for n in decode_engines
            )

        running = wait_for(victim_running, 10.0, 0.1)
        if not running:
            return False, "victim never reached RUNNING on decode"

        # Incomer: the acceptance permit freed at the victim's
        # DecodeAccepted event, so the incomer is ADMITTED fast and runs
        # alongside the victim (new-B semantics).
        incomer_rid = ops.next_request_id(base)
        incomer_keys = [incomer_rid * 100 + 1]
        t0 = time.monotonic()
        incomer_resp = _schedule_with_priority(
            ops,
            incomer_rid,
            70,
            input_len=512,
            output_len=2,
            block_keys=incomer_keys,
        )
        accept_latency = time.monotonic() - t0
        incomer_msg = str(incomer_resp.error_message)
        incomer_accepted = incomer_resp.code == 200 and incomer_resp.success
        if incomer_accepted:
            incomer_input = (
                None
                if incomer_resp.enqueued_by_master
                else ops.build_generate_input(
                    incomer_rid,
                    input_len=512,
                    output_len=2,
                    block_keys=incomer_keys,
                )
            )
            incomer_handle = ops.start_stream(
                incomer_resp, incomer_rid, input_pb=incomer_input
            )

        # Victim finishes unmolested (no preemption -> no 8429); the
        # incomer completes alongside it.
        victim_ended = victim_handle.wait_end(30.0)
        victim_completed = (
            victim_ended
            and victim_handle.snap.completed
            and not victim_handle.snap.error
        )
        incomer_completed = False
        if incomer_handle is not None:
            incomer_ended = incomer_handle.wait_end(30.0)
            incomer_completed = (
                incomer_ended
                and incomer_handle.snap.completed
                and not incomer_handle.snap.error
            )
        incomer_outputs = (
            len(incomer_handle.snap.outputs) if incomer_handle is not None else 0
        )

        # Post-wave hygiene: a fresh request must still succeed.
        recovery_ok, recovery_msg = ops.verify_recovery()
        inflight_ok, inflight_detail = AssertUtils.inflight_clean(
            _master_http(ops), 30.0
        )
        engine_clean, engine_detail = engine_inflight_clean(
            ops, _prefill_names(ops) + decode_engines, 15.0
        )

        accepted_fast = accept_latency < 3.0
        passed = (
            incomer_accepted
            and accepted_fast
            and incomer_completed
            and victim_completed
            and recovery_ok
            and inflight_ok
            and engine_clean
        )
        return passed, (
            f"incomer_accepted={incomer_accepted} "
            f"(code={incomer_resp.code}, latency={accept_latency:.2f}s, "
            f"msg={incomer_msg[:80]}), "
            f"incomer_completed={incomer_completed} "
            f"(outputs={incomer_outputs}), "
            f"victim_completed={victim_completed} "
            f"(outputs={len(victim_handle.snap.outputs)}), "
            f"recovery={recovery_msg}, "
            f"inflight_clean={inflight_ok}({inflight_detail}), "
            f"engine_clean={engine_clean}({engine_detail})"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        if incomer_handle is not None:
            incomer_handle.cancel()
        if victim_handle is not None:
            victim_handle.cancel()
