from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from ...context import CaseContext, rid_base
from ...engine_ops import engine_inflight_clean
from ...harness import AssertUtils, ConfigOverride, EnvSpec, default_perf, wait_for
from ...registry import case
from ...support.cancel import (
    _PB_ERROR_CANCELLED,
    CODE_ENGINE_CANCELLED,
    _all_engine_names,
    _cancel_rpc_total,
    _master_http,
    _schedule_with_priority,
)


@case("cancel_preemption_victim", category="cancel")
def cancel_preemption_victim(ctx: CaseContext):
    """M3 preemption: a P70 arrival evicts a RUNNING P30 victim through the
    master → original-Prefill weak-Cancel protocol.

    Scenario: dedicated 1P+1D environment, PRIORITY ordering with the
    victim stages {PREFILL_QUEUED, DECODE_RESERVED, DECODE_ENGINE_OWNED}
    plus the engineCancellation block (the engine-owned stage REQUIRES
    it — FlexlbConfigValidator rejects the stage set otherwise; see
    flexlb_cfg._build_preemption_cfg for the schema) and decode
    maxEngineRequests=1 so the single decode slot makes the capacity
    contest deterministic.  The victim (priority 30, input_len=512 /
    output_len=200 — long decode) is scheduled first and waits RUNNING
    on the decode engine; the preemptor (priority 70, output_len=2)
    then arrives.

    Stage contract (EvictionPlanner/PreemptionConfig): a RUNNING
    victim is engine-confirmed — only the DECODE_ENGINE_OWNED admission
    path can reach it (planDecodeOne gates the ENGINE_CANCEL ownership
    on preemption.allows(DECODE_ENGINE_OWNED) + the cancel channel).
    The 2026-09-04 run proved the legacy set
    {PREFILL_QUEUED, DECODE_RESERVED} (master_fixed_window.json values)
    wrong for this choreography: a RUNNING victim is invisible to both
    master-local layers, so it simply ran to completion and the weak
    cancel never fired (delta=0) — correct behaviour for THAT config,
    which makes this a construction fix, not an assertion relaxation.

    Behaviour: the preemptor's ordinary placement is BLOCKED (decode
    capacity exhausted), so RequestScheduler.attemptPlacement escalates
    to EvictionManager.tryAdmit; slotDeficit=1 (engineLoad 1, limit 1)
    makes the planner select the engine-owned victim and the master
    sends the real (weak) Cancel to the ORIGINAL prefill (tokenized
    Cancel coordinator, GrpcEngineCancelChannel), which propagates to
    decode (P→D stream-cancel conduction).

    Expected (contract): the victim's stream terminates in a
    non-completion terminal carrying the typed 8429
    (PRIORITY_PREEMPTED / CANCELLED) error; the engine records the
    victim as cancelled (cancelled_rids / lifecycle); the original
    prefill's Cancel RPC counter increased (the weak cancel really
    went out); the P70 request completes normally once the slot frees;
    the master inflight ledger drains with no leak; recovery works.

    Prediction: this is the legacy priority-preemption smoke scenario
    (RUNNING decode victim, batch default) ported onto the flexlb_ft
    framework; capacity here comes from maxEngineRequests=1 instead of
    the smoke line's KV pressure so the eviction trigger is
    deterministic.  Priority rides the Schedule proto's priority field
    (see _schedule_with_priority).
    """
    # Victim-stage contract: the victim is polled to RUNNING on the
    # decode engine (= engine-confirmed), so the stage set MUST include
    # DECODE_ENGINE_OWNED — and that stage requires the engineCancellation
    # block (ackTimeoutMs / completionTimeoutMs, the same values the
    # priority family's _PREEMPT_DECODE spec uses).  The legacy
    # {PREFILL_QUEUED, DECODE_RESERVED} set left the RUNNING victim
    # unreachable by every eviction layer (see the docstring's stage-
    # contract note).
    spec = EnvSpec(
        label=f"cancel_preempt_{ctx.profile}",
        n_prefill=1,
        n_decode=1,
        perf=default_perf(),
        master_profile=ctx.profile,
        config_overrides=ConfigOverride(
            ordering="priority",
            default_priority=50,
            preemption={
                "allowed_victim_stages": [
                    "PREFILL_QUEUED",
                    "DECODE_RESERVED",
                    "DECODE_ENGINE_OWNED",
                ],
                "engine_cancellation": {
                    "ack_timeout_ms": 50,
                    "completion_timeout_ms": 1000,
                },
            },
            decode_max_engine_requests=1,
        ),
    )
    env = ctx.env_manager.ensure(spec)
    ops = ctx.engine_ops(env)
    base = rid_base(ctx, "cancel")
    victim_handle = None
    high_handle = None
    try:
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
            return False, f"victim schedule failed: {victim_resp.error_message}"
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

        # Victim must be RUNNING on the decode engine before the preemptor
        # arrives — otherwise the preemptor would simply take the free slot.
        def victim_running() -> bool:
            snap = ops.snapshot_by_name()
            return any(
                e.get("role") == "decode"
                and e.get("request_lifecycle", {})
                .get(str(victim_rid), {})
                .get("end_state")
                == "running"
                for e in snap.values()
            )

        running = wait_for(victim_running, 10.0, 0.1)
        if not running:
            victim_handle.cancel()
            return False, "victim never reached RUNNING on decode"

        baseline_cancel = _cancel_rpc_total(ops)
        high_rid = ops.next_request_id(base)
        high_keys = [high_rid * 100 + 1]
        with ThreadPoolExecutor(max_workers=1) as pool:
            high_future = pool.submit(
                _schedule_with_priority,
                ops,
                high_rid,
                70,
                input_len=512,
                output_len=2,
                block_keys=high_keys,
            )
            # The victim's terminal: preemption ends its stream in a
            # non-completion state (typed 8429 surfaces as the stream
            # error under both dispatch modes).
            victim_ended = victim_handle.wait_end(20.0)
            try:
                high_resp = high_future.result(timeout=40.0)
            except Exception as exc:
                return False, f"high-priority schedule failed: {exc!r}"
        if high_resp.code != 200 or not high_resp.success:
            return False, f"high schedule failed: {high_resp.error_message}"
        high_input = (
            None
            if high_resp.enqueued_by_master
            else ops.build_generate_input(
                high_rid,
                input_len=512,
                output_len=2,
                block_keys=high_keys,
            )
        )
        high_handle = ops.start_stream(high_resp, high_rid, input_pb=high_input)
        high_handle.wait_end(30.0)

        victim_cancelled, victim_cancel_detail = ops.verify_engine_cancelled(victim_rid)
        weak_cancel_delta = _cancel_rpc_total(ops) - baseline_cancel
        if victim_resp.enqueued_by_master or high_resp.enqueued_by_master:
            inflight_ok, inflight_detail = AssertUtils.inflight_clean(
                _master_http(ops), 15.0
            )
        else:
            inflight_ok, inflight_detail = True, "N/A"
        engine_clean, engine_clean_detail = engine_inflight_clean(
            ops, _all_engine_names(ops), 15.0
        )
        recovery_ok, recovery_msg = ops.verify_recovery()
        # Victim terminal hard gate: EXACTLY the typed 8429 engine-cancelled
        # terminal, not merely "not completed".  Stage discriminator 8400 vs
        # 8429: 8400 is the master-local atomic eviction (PREFILL_QUEUED /
        # DECODE_RESERVED victims — never engine-confirmed, settles on the
        # schedule RPC before any stream exists), while this RUNNING victim
        # is DECODE_ENGINE_OWNED, so eviction must ride the engine Cancel and
        # surface as the stream's in-band CANCELLED frame (raw enum 2 = the
        # 8429 family's client-side form; the numeric rides the TaskInfoPB
        # master channel only — same mapping as priority.py's
        # _StreamTerminal).
        victim_typed = victim_handle.snap.stream_error_code
        if victim_typed == _PB_ERROR_CANCELLED:
            victim_typed = CODE_ENGINE_CANCELLED
        passed = (
            victim_ended
            and victim_typed == CODE_ENGINE_CANCELLED
            and victim_cancelled
            and weak_cancel_delta >= 1
            and high_handle.snap.completed
            and not high_handle.snap.error
            and inflight_ok
            and engine_clean
            and recovery_ok
        )
        return passed, (
            f"preemption_victim: victim_terminated={victim_ended}"
            f"(completed={victim_handle.snap.completed}, "
            f"typed={victim_typed}, error={victim_handle.snap.error}), "
            f"victim_engine_cancelled={victim_cancelled}"
            f"({victim_cancel_detail}), weak_cancel_delta={weak_cancel_delta}, "
            f"high_completed={high_handle.snap.completed}"
            f"(outputs={len(high_handle.snap.outputs)}), "
            f"inflight_clean={inflight_ok}({inflight_detail}), "
            f"engine_clean={engine_clean}({engine_clean_detail}), "
            f"recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        if victim_handle is not None:
            victim_handle.cancel()
        if high_handle is not None:
            high_handle.cancel()
