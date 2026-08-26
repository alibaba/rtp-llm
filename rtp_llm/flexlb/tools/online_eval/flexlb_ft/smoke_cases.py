"""Smoke test cases (functional correctness e2e).

Ported 1:1 from the legacy scripts (assertion thresholds preserved):

  smoke_cancel_t1..t6        <- cancel_smoke.py    T1-T6
  smoke_scheduling_s1..s12   <- scheduling_smoke.py S1-S12
  smoke_anomaly_e1..e3       <- anomaly_smoke.py   E1-E3

Mode adaptation follows run_matrix_smoke.sh grouping:
  * S4/S5/S6 assert only under batch (COST_BASED_PREFILL)
  * S7/S8/S9 assert only under direct/queue (SHORTEST_TTFT)
  * everything else runs under all modes

Behavioural corrections for the Java mock (documented inline):
  * S4 — legacy injected a fake ``queue_depth`` display value (80000) that the
    Java mock implements as a *real* enqueue rejection gate
    (FaultInjectionConfig.queueDepthLimit); a huge limit never triggers, so
    the legacy "requests avoid the hot worker" assertion no longer holds via
    that knob.  The Java-true way to build a hotspot is real pending load:
    both prefill engines are slowed, a seed request plus a 4-request burst
    pile real pendingCount onto one engine, and the next burst must bias
    away from it while that pending persists.  All waves are
    fire-and-forget inside the slowed prefill window — waiting for wave-2
    completion (the original port) drained the hotspot back to zero before
    the assertion had any basis.  Assertion form: hot_count <= 2 of 5
    (soft RANDOM_WITHIN_TOLERANCE preference).
  * S9 — 50000 is far above any reachable pending count, so the gate never
    fires and requests still route to the target worker; the legacy
    ``target_count > 0`` assertion is kept and now documents exactly that.
  * role_addr is called with the proto role *string* (the legacy code passed
    the ROLE_TYPE enum int, which never matched — see engine_ops.py).
"""

from __future__ import annotations

import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from .context import CaseContext, CaseDef, rid_base
from .harness import AssertUtils

KV_CACHE_SYNC_WAIT_S = 2.0
STREAM_TIMEOUT_S = 15.0

SMOKE_CASES: list[CaseDef] = []


def case(name: str, modes=None, source: str = ""):
    def deco(fn):
        SMOKE_CASES.append(
            CaseDef(name=name, suite="smoke", fn=fn, modes=modes, source=source)
        )
        return fn

    return deco


def _master_http(ops) -> str:
    return f"http://127.0.0.1:{ops.master_http_port}"


# ===========================================================================
# Cancel cases (cancel_smoke.py T1-T6)
# ===========================================================================


def _engine_side_checks(ops, rid: int, response) -> tuple[str, str, str, str]:
    """Common engine-side verification (recv / cancelled / inflight-clean)."""
    method = "enqueue_batch" if response.enqueued_by_master else "generate_stream"
    engine_recv, recv_detail = ops.verify_engine_received(rid, method)
    engine_cancelled, cancel_detail = ops.verify_engine_cancelled(rid)
    if response.enqueued_by_master:
        inflight_ok, inflight_detail = AssertUtils.inflight_clean(
            _master_http(ops), 10.0
        )
    else:
        inflight_ok, inflight_detail = True, "N/A"
    return (
        f"engine_recv={engine_recv}({recv_detail}), "
        f"engine_cancelled={engine_cancelled}({cancel_detail}), "
        f"inflight_clean={inflight_ok}({inflight_detail}), ",
        recv_detail,
        "",
        "",
    )


@case("smoke_cancel_t1", source="cancel_smoke.py T1")
def t1_basic_cancel(ctx: CaseContext):
    ops = ctx.ops()
    rid = ops.next_request_id(rid_base(ctx, "cancel"))
    try:
        response = ops.schedule(rid)
        if response.code != 200 or not response.success:
            return False, f"schedule failed: {response.error_message}"
        input_pb = (
            None if response.enqueued_by_master else ops.build_generate_input(rid)
        )
        handle = ops.start_stream(response, rid, input_pb=input_pb)
        if not handle.wait_first_output():
            handle.cancel()
            return False, "no output received before cancel window"
        cancel_at = time.monotonic()
        ops.cancel(rid, response)
        ended = handle.wait_end(5.0)
        cancel_latency = time.monotonic() - cancel_at
        recovery_ok, recovery_msg = ops.verify_recovery()
        method = "enqueue_batch" if response.enqueued_by_master else "generate_stream"
        engine_recv, recv_detail = ops.verify_engine_received(rid, method)
        engine_cancelled, cancel_detail = ops.verify_engine_cancelled(rid)
        if response.enqueued_by_master:
            inflight_ok, inflight_detail = AssertUtils.inflight_clean(
                _master_http(ops), 10.0
            )
        else:
            inflight_ok, inflight_detail = True, "N/A"
        passed = ended and recovery_ok
        return passed, (
            f"cancel_latency={cancel_latency:.3f}s, stream_terminated={ended}, "
            f"outputs={len(handle.snap.outputs)}, "
            f"engine_recv={engine_recv}({recv_detail}), "
            f"engine_cancelled={engine_cancelled}({cancel_detail}), "
            f"inflight_clean={inflight_ok}({inflight_detail}), recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"


@case("smoke_cancel_t2", source="cancel_smoke.py T2")
def t2_cancel_idempotency(ctx: CaseContext):
    ops = ctx.ops()
    rid = ops.next_request_id(rid_base(ctx, "cancel"))
    try:
        response = ops.schedule(rid)
        if response.code != 200 or not response.success:
            return False, f"schedule failed: {response.error_message}"
        input_pb = (
            None if response.enqueued_by_master else ops.build_generate_input(rid)
        )
        handle = ops.start_stream(response, rid, input_pb=input_pb)
        if not handle.wait_first_output():
            handle.cancel()
            return False, "no output received before cancel window"
        ops.cancel(rid, response)
        second_cancel_ok, second_cancel_err = True, ""
        try:
            ops.cancel(rid, response)
        except Exception as exc:
            second_cancel_ok, second_cancel_err = False, repr(exc)
        ended = handle.wait_end(5.0)
        recovery_ok, recovery_msg = ops.verify_recovery()
        method = "enqueue_batch" if response.enqueued_by_master else "generate_stream"
        engine_recv, recv_detail = ops.verify_engine_received(rid, method)
        engine_cancelled, cancel_detail = ops.verify_engine_cancelled(rid)
        if response.enqueued_by_master:
            inflight_ok, inflight_detail = AssertUtils.inflight_clean(
                _master_http(ops), 10.0
            )
        else:
            inflight_ok, inflight_detail = True, "N/A"
        passed = second_cancel_ok and ended and recovery_ok
        return passed, (
            f"second_cancel_ok={second_cancel_ok} {second_cancel_err}, "
            f"stream_terminated={ended}, "
            f"engine_recv={engine_recv}({recv_detail}), "
            f"engine_cancelled={engine_cancelled}({cancel_detail}), "
            f"inflight_clean={inflight_ok}({inflight_detail}), recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"


@case("smoke_cancel_t3", source="cancel_smoke.py T3")
def t3_multi_request_isolation(ctx: CaseContext):
    ops = ctx.ops()
    base = rid_base(ctx, "cancel")
    rids = [ops.next_request_id(base) for _ in range(3)]
    cancel_rid = rids[1]  # B
    # B runs a long decode so the cancel lands while it is still decoding.
    # The default output_len=10 finishes decode in ~200ms and races the
    # cancel: the engine finishes first, the master then (correctly) returns
    # the terminal state idempotently without forwarding an engine cancel,
    # and verify_engine_cancelled flaps.  200 tokens ≈ 4-5s of decode at the
    # default perf step_ms, comfortably spanning the cancel window.
    long_output_len = 200
    try:

        def _schedule(rid: int):
            if rid == cancel_rid:
                return ops.schedule(rid, output_len=long_output_len)
            return ops.schedule(rid)

        with ThreadPoolExecutor(max_workers=3) as pool:
            responses = list(pool.map(_schedule, rids))
        for i, resp in enumerate(responses):
            if resp.code != 200 or not resp.success:
                return False, f"schedule failed for rid={rids[i]}: {resp.error_message}"

        handles = []
        for rid, resp in zip(rids, responses):
            input_pb = (
                None if resp.enqueued_by_master else ops.build_generate_input(rid)
            )
            handles.append(ops.start_stream(resp, rid, input_pb=input_pb))

        # Wait for the SHORT requests' (A, C) first output only.  In batch
        # mode the mock engine's FetchResponse surfaces the first message
        # only after decode completes, so waiting for B (output_len=200)
        # would mean B is already terminal when the cancel fires — the
        # master then (correctly) answers REQUEST_STATE_COMPLETED
        # idempotently and never forwards an engine cancel.  A/C finish in
        # ~1s while B still has ~3.5s of decode left, so cancelling right
        # after A/C's first output lands the cancel mid-decode.
        if not all(handles[i].wait_first_output(15.0) for i in (0, 2)):
            for h in handles:
                h.cancel()
            return False, "short requests (A, C) did not receive first output"

        ops.cancel(cancel_rid, responses[1])
        b_ended = handles[1].wait_end(5.0)
        a_complete = handles[0].wait_end(30.0)
        c_complete = handles[2].wait_end(30.0)

        recovery_ok, recovery_msg = ops.verify_recovery()
        method = (
            "enqueue_batch" if responses[1].enqueued_by_master else "generate_stream"
        )
        engine_recv, recv_detail = ops.verify_engine_received(cancel_rid, method)
        engine_cancelled, cancel_detail = ops.verify_engine_cancelled(cancel_rid)
        if responses[1].enqueued_by_master:
            inflight_ok, inflight_detail = AssertUtils.inflight_clean(
                _master_http(ops), 10.0
            )
        else:
            inflight_ok, inflight_detail = True, "N/A"

        a_snap, b_snap, c_snap = handles[0].snap, handles[1].snap, handles[2].snap
        passed = (
            b_ended
            and a_complete
            and a_snap.completed
            and c_complete
            and c_snap.completed
            and not b_snap.completed
            and engine_cancelled
            and recovery_ok
        )
        return passed, (
            f"A_completed={a_snap.completed}(outputs={len(a_snap.outputs)}), "
            f"B_cancelled={b_ended}(completed={b_snap.completed}), "
            f"C_completed={c_snap.completed}(outputs={len(c_snap.outputs)}), "
            f"engine_recv={engine_recv}({recv_detail}), "
            f"engine_cancelled={engine_cancelled}({cancel_detail}), "
            f"inflight_clean={inflight_ok}({inflight_detail}), recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"


@case("smoke_cancel_t4", source="cancel_smoke.py T4")
def t4_cancel_after_completion(ctx: CaseContext):
    ops = ctx.ops()
    rid = ops.next_request_id(rid_base(ctx, "cancel"))
    try:
        response = ops.schedule(rid, output_len=1)
        if response.code != 200 or not response.success:
            return False, f"schedule failed: {response.error_message}"
        input_pb = (
            None if response.enqueued_by_master else ops.build_generate_input(rid)
        )
        handle = ops.start_stream(response, rid, input_pb=input_pb)
        deadline = time.monotonic() + 30.0
        while not handle.snap.completed and time.monotonic() < deadline:
            time.sleep(0.05)
        if not handle.snap.completed:
            handle.cancel()
            return False, "request did not complete before timeout"
        cancel_ok, cancel_err = True, ""
        try:
            ops.cancel(rid, response)
        except Exception as exc:
            cancel_ok, cancel_err = False, repr(exc)
        handle.wait_end(2.0)
        recovery_ok, recovery_msg = ops.verify_recovery()
        method = "enqueue_batch" if response.enqueued_by_master else "generate_stream"
        engine_recv, recv_detail = ops.verify_engine_received(rid, method)
        engine_cancelled, cancel_detail = ops.verify_engine_cancelled(rid)
        if response.enqueued_by_master:
            inflight_ok, inflight_detail = AssertUtils.inflight_clean(
                _master_http(ops), 10.0
            )
        else:
            inflight_ok, inflight_detail = True, "N/A"
        passed = cancel_ok and recovery_ok
        return passed, (
            f"cancel_ok={cancel_ok} {cancel_err}, completed={handle.snap.completed}, "
            f"engine_recv={engine_recv}({recv_detail}), "
            f"engine_cancelled={engine_cancelled}({cancel_detail}), "
            f"inflight_clean={inflight_ok}({inflight_detail}), recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"


@case("smoke_cancel_t5", source="cancel_smoke.py T5")
def t5_cancel_nonexistent(ctx: CaseContext):
    ops = ctx.ops()
    try:
        fake_rid = 99999
        cancel_ok, cancel_err = True, ""
        try:
            ops.cancel(fake_rid)
        except Exception as exc:
            cancel_ok, cancel_err = False, repr(exc)
        return cancel_ok, (
            f"cancel(rid={fake_rid}) ok={cancel_ok} {cancel_err}, "
            f"engine_verify=N/A (nonexistent request)"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"


@case("smoke_cancel_t6", source="cancel_smoke.py T6")
def t6_cancel_at_prefill_vs_decode(ctx: CaseContext):
    ops = ctx.ops()
    base = rid_base(ctx, "cancel")
    try:
        # A: cancel in prefill phase (before any output)
        rid_a = ops.next_request_id(base)
        resp_a = ops.schedule(rid_a)
        if resp_a.code != 200 or not resp_a.success:
            return False, f"schedule A failed: {resp_a.error_message}"
        input_pb_a = (
            None if resp_a.enqueued_by_master else ops.build_generate_input(rid_a)
        )
        handle_a = ops.start_stream(resp_a, rid_a, input_pb=input_pb_a)
        time.sleep(0.1)
        a_in_prefill = not handle_a.snap.first_received
        ops.cancel(rid_a, resp_a)
        a_ended = handle_a.wait_end(5.0)

        # B: cancel in decode phase (after first output)
        rid_b = ops.next_request_id(base)
        resp_b = ops.schedule(rid_b)
        if resp_b.code != 200 or not resp_b.success:
            return False, f"schedule B failed: {resp_b.error_message}"
        input_pb_b = (
            None if resp_b.enqueued_by_master else ops.build_generate_input(rid_b)
        )
        handle_b = ops.start_stream(resp_b, rid_b, input_pb=input_pb_b)
        b_got_first = handle_b.wait_first_output()
        if not b_got_first:
            handle_b.cancel()
            return False, "B never received first output (decode phase)"
        ops.cancel(rid_b, resp_b)
        b_ended = handle_b.wait_end(5.0)

        recovery_ok, recovery_msg = ops.verify_recovery()
        method_a = "enqueue_batch" if resp_a.enqueued_by_master else "generate_stream"
        method_b = "enqueue_batch" if resp_b.enqueued_by_master else "generate_stream"
        engine_recv_a, _ = ops.verify_engine_received(rid_a, method_a)
        engine_cancelled_a, _ = ops.verify_engine_cancelled(rid_a)
        engine_recv_b, _ = ops.verify_engine_received(rid_b, method_b)
        engine_cancelled_b, _ = ops.verify_engine_cancelled(rid_b)
        if resp_a.enqueued_by_master or resp_b.enqueued_by_master:
            inflight_ok, inflight_detail = AssertUtils.inflight_clean(
                _master_http(ops), 10.0
            )
        else:
            inflight_ok, inflight_detail = True, "N/A"

        passed = a_ended and b_ended and a_in_prefill and recovery_ok
        return passed, (
            f"A_prefill_phase={a_in_prefill}, A_terminated={a_ended}, "
            f"A_outputs={len(handle_a.snap.outputs)}, "
            f"B_decode_phase={b_got_first}, B_terminated={b_ended}, "
            f"B_outputs={len(handle_b.snap.outputs)}, "
            f"engine_recv_A={engine_recv_a}, engine_cancel_A={engine_cancelled_a}, "
            f"engine_recv_B={engine_recv_b}, engine_cancel_B={engine_cancelled_b}, "
            f"inflight_clean={inflight_ok}({inflight_detail}), recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"


# ===========================================================================
# Scheduling cases (scheduling_smoke.py S1-S12)
# ===========================================================================


def _prefill_names(ops) -> list[str]:
    snap = ops.snapshot_by_name()
    return sorted(name for name, e in snap.items() if e.get("role") == "prefill")


@case("smoke_scheduling_s1", source="scheduling_smoke.py S1")
def s1_load_balance(ctx: CaseContext):
    ops = ctx.ops()
    n = 10
    is_batch = ctx.mode == "batch"
    perf_engine = None
    try:
        if is_batch:
            prefill_names = _prefill_names(ops)
            if len(prefill_names) >= 2:
                ops.set_perf(prefill_names[1], prefill_fixed_ms=200.0)
                perf_engine = prefill_names[1]
                # Let the master sync the changed perf metrics before routing
                # (legacy code raced this; its assertion was inert due to the
                # role_addr bug — here it is effective and needs the settle).
                time.sleep(1.5)

        addrs = []
        for _ in range(n):
            rid = ops.next_request_id(rid_base(ctx, "scheduling"))
            keys = [rid * 100 + j for j in range(3)]
            addr, err = ops.run_one_request(
                rid, output_len=2, block_keys=keys, stream_timeout_s=STREAM_TIMEOUT_S
            )
            if err:
                return False, f"rid={rid} failed: {err}"
            addrs.append(addr)

        counts = Counter(addrs)
        num_workers = len(counts)
        max_ratio = max(counts.values()) / n if n else 1.0
        addr_map = ops.addr_to_name()
        dist_names = {addr_map.get(a, a): c for a, c in counts.items()}
        snap = ops.snapshot_by_name()
        accepted = {
            name: info.get("accepted", 0)
            for name, info in snap.items()
            if info.get("role") == "prefill"
        }

        if is_batch and perf_engine:
            slow_count = dist_names.get(perf_engine, 0)
            passed = slow_count == 0
            batch_detail = (
                f", slow_worker={perf_engine}({slow_count}), "
                f"batch_deterministic={'yes' if passed else 'no'}"
            )
        else:
            passed = True
            batch_detail = ""
        return passed, (
            f"requests={n}, workers={num_workers}, max_ratio={max_ratio:.2f}, "
            f"distribution={json.dumps(dist_names, sort_keys=True)}, "
            f"snapshot_accepted={json.dumps(accepted, sort_keys=True)}{batch_detail}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        if perf_engine:
            try:
                ops.set_perf(perf_engine, prefill_fixed_ms=100.0)
            except Exception:
                pass


@case("smoke_scheduling_s2", source="scheduling_smoke.py S2")
def s2_kv_cache_affinity(ctx: CaseContext):
    ops = ctx.ops()
    base = rid_base(ctx, "scheduling")
    keys = [1001, 1002, 1003]
    try:
        rid_a = ops.next_request_id(base)
        addr_a, err_a = ops.run_one_request(
            rid_a,
            input_len=2048,
            output_len=2,
            block_keys=keys,
            stream_timeout_s=STREAM_TIMEOUT_S,
        )
        if err_a:
            return False, f"request A failed: {err_a}"
        time.sleep(KV_CACHE_SYNC_WAIT_S)
        rid_b = ops.next_request_id(base)
        addr_b, err_b = ops.run_one_request(
            rid_b,
            input_len=2048,
            output_len=2,
            block_keys=keys,
            stream_timeout_s=STREAM_TIMEOUT_S,
        )
        if err_b:
            return False, f"request B failed: {err_b}"
        if addr_a == addr_b:
            return True, f"affinity confirmed: A=B={addr_a}"
        # retry once (cache sync may lag)
        time.sleep(KV_CACHE_SYNC_WAIT_S)
        rid_c = ops.next_request_id(base)
        addr_c, err_c = ops.run_one_request(
            rid_c,
            input_len=2048,
            output_len=2,
            block_keys=keys,
            stream_timeout_s=STREAM_TIMEOUT_S,
        )
        if err_c:
            return False, f"request C failed: {err_c}"
        passed = addr_a == addr_c
        return passed, (
            f"retry: A={addr_a}, B={addr_b}, C={addr_c}, "
            f"match={'A==C' if passed else 'none'}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"


@case("smoke_scheduling_s3", source="scheduling_smoke.py S3")
def s3_decode_balance(ctx: CaseContext):
    ops = ctx.ops()
    n = 10
    try:
        for _ in range(n):
            rid = ops.next_request_id(rid_base(ctx, "scheduling"))
            keys = [rid * 100 + j for j in range(3)]
            _, err = ops.run_one_request(
                rid, output_len=2, block_keys=keys, stream_timeout_s=STREAM_TIMEOUT_S
            )
            if err:
                return False, f"rid={rid} failed: {err}"
        snap = ops.snapshot_by_name()
        completed = {
            name: info.get("completed", 0)
            for name, info in snap.items()
            if info.get("role") == "decode"
        }
        total = sum(completed.values())
        used = sum(1 for v in completed.values() if v > 0)
        passed = used >= 2 and total >= n
        return passed, (
            f"requests={n}, decode_workers={used}, "
            f"total_completed={total}, "
            f"distribution={json.dumps(completed, sort_keys=True)}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"


@case(
    "smoke_scheduling_s4",
    modes=["batch"],
    source="scheduling_smoke.py S4 (Java-corrected)",
)
def s4_hotspot_filter(ctx: CaseContext):
    """Hotspot bias: requests divert away from the engine with high real pending.

    Java-corrected scenario (see module docstring): the hotspot is built from
    *real* pending load, not the legacy fake queue_depth knob.  Race fix:
    the previous flow waited for the wave-2 burst to complete and slept
    1.2s, so the hotspot pending had drained back to zero by the time the
    verification burst asserted anything — the filter had no basis at
    assertion time.  All waves are now fire-and-forget inside the 5s
    prefill window so the pending persists:

      1. slow both prefill engines to 5s (keeps pendingCount alive)
      2. seed one request (fire-and-forget); it lands on engine S and its
         pending steers the next wave away from S
      3. wave 2: 4 concurrent fire-and-forget requests concentrate on the
         engine the seed avoided (S's pending=1 pushes its cost score past
         the RANDOM_WITHIN_TOLERANCE window) — that engine is the hotspot
      4. wave 3: 5 requests fired immediately (no waiting, no sleep) while
         the hotspot still holds the wave-2 pending; they must bias away
      5. assertion (soft preference): hot_count <= 2 of 5 — the tolerance
         window admits a few requests once the other side's pending catches
         up, but the hotspot must never take the majority
    """
    ops = ctx.ops()
    base = rid_base(ctx, "scheduling")
    try:
        prefill_names = _prefill_names(ops)
        if len(prefill_names) < 2:
            return False, "need >=2 prefill workers"

        ops.set_perf(prefill_names[0], prefill_fixed_ms=5000.0)
        ops.set_perf(prefill_names[1], prefill_fixed_ms=5000.0)
        time.sleep(1.5)  # master syncs the slowed perf before we seed

        addr_map = ops.addr_to_name()

        def fire(rid: int):
            """Schedule without consuming the stream — keeps it pending."""
            resp = ops.schedule(
                rid, output_len=2, block_keys=[rid * 100 + j for j in range(3)]
            )
            if resp.code != 200 or not resp.success:
                return None, f"schedule failed: {resp.error_message}"
            return addr_map.get(ops.role_addr(resp, "PREFILL"), ""), None

        # -- wave 1: seed one request (fire-and-forget).  Its pending steers
        #    wave 2 deterministically onto the other engine.
        rid1 = ops.next_request_id(base)
        seed_name, err1 = fire(rid1)
        if err1:
            return False, f"seed request failed: {err1}"
        if seed_name not in prefill_names:
            return False, f"seed request went to unknown worker {seed_name}"

        # -- wave 2: 4 concurrent fire-and-forget requests concentrate on
        #    the engine the seed avoided — that engine is now the hotspot
        rids2 = [ops.next_request_id(base) for _ in range(4)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            wave2 = list(pool.map(fire, rids2))
        for rid, (name, err) in zip(rids2, wave2):
            if err:
                return False, f"wave2 rid={rid} failed: {err}"
        wave2_names = [name for name, _ in wave2 if name]
        hot, hot_votes = Counter(wave2_names).most_common(1)[0]
        if hot_votes < 3:
            return False, (
                f"wave2 did not concentrate on one engine "
                f"({json.dumps(dict(Counter(wave2_names)), sort_keys=True)}) "
                f"— hotspot precondition failed"
            )
        cool = next(n for n in prefill_names if n != hot)

        # -- wave 3 (verification): fire immediately — no waiting, no sleep —
        #    so the hotspot pending from wave 2 still exists at routing time
        wave3_names = []
        for _ in range(5):
            rid = ops.next_request_id(base)
            name, err = fire(rid)
            if err:
                return False, f"rid={rid} failed: {err}"
            wave3_names.append(name)
            time.sleep(0.05)

        dist = Counter(wave3_names)
        hot_count = dist.get(hot, 0)
        cool_count = dist.get(cool, 0)
        passed = hot_count <= 2
        return passed, (
            f"hot={hot}({hot_count}), cool={cool}({cool_count}), "
            f"dist={json.dumps(dict(dist), sort_keys=True)}, "
            f"wave2={json.dumps(dict(Counter(wave2_names)), sort_keys=True)}, "
            f"assertion=hot_count<=2 (soft preference while pending persists)"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        try:
            ops.set_perf(prefill_names[0], prefill_fixed_ms=100.0)
            ops.set_perf(prefill_names[1], prefill_fixed_ms=100.0)
        except Exception:
            pass
        # Drain the fire-and-forget requests (each finishes after its ~5s
        # prefill + decode) so the shared env is clean for the next cases.
        try:
            AssertUtils.inflight_clean(_master_http(ops), 90.0)
        except Exception:
            pass


@case("smoke_scheduling_s5", modes=["batch"], source="scheduling_smoke.py S5")
def s5_kv_cache_hit_preference(ctx: CaseContext):
    ops = ctx.ops()
    base = rid_base(ctx, "scheduling")
    keys = [999]
    try:
        rid_a = ops.next_request_id(base)
        addr_a, err_a = ops.run_one_request(
            rid_a,
            input_len=2048,
            output_len=2,
            block_keys=keys,
            stream_timeout_s=STREAM_TIMEOUT_S,
        )
        if err_a:
            return False, f"request A failed: {err_a}"
        time.sleep(KV_CACHE_SYNC_WAIT_S)
        rid_b = ops.next_request_id(base)
        addr_b, err_b = ops.run_one_request(
            rid_b,
            input_len=2048,
            output_len=2,
            block_keys=keys,
            stream_timeout_s=STREAM_TIMEOUT_S,
        )
        if err_b:
            return False, f"request B failed: {err_b}"
        addr_map = ops.addr_to_name()
        name_a = addr_map.get(addr_a, addr_a)
        name_b = addr_map.get(addr_b, addr_b)
        snap = ops.snapshot_by_name()
        cache_count = snap.get(name_a, {}).get("cache_keys", 0)
        passed = addr_a == addr_b and cache_count > 0
        return passed, (
            f"A={name_a}, B={name_b}, "
            f"same={'yes' if addr_a == addr_b else 'no'}, "
            f"cache_keys={cache_count}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"


@case("smoke_scheduling_s6", modes=["batch"], source="scheduling_smoke.py S6")
def s6_cost_based_determinism(ctx: CaseContext):
    ops = ctx.ops()
    try:
        addrs = []
        for _ in range(5):
            rid = ops.next_request_id(rid_base(ctx, "scheduling"))
            keys = [rid * 100 + j for j in range(3)]
            addr, err = ops.run_one_request(
                rid, output_len=2, block_keys=keys, stream_timeout_s=STREAM_TIMEOUT_S
            )
            if err:
                return False, f"rid={rid} failed: {err}"
            addrs.append(addr)
        addr_map = ops.addr_to_name()
        dist = Counter(addr_map.get(a, a) for a in addrs)
        num_workers = len(dist)
        snap = ops.snapshot_by_name()
        accepted = {
            name: info.get("accepted", 0)
            for name, info in snap.items()
            if info.get("role") == "prefill"
        }
        passed = num_workers == 1
        return passed, (
            f"workers={num_workers}, "
            f"dist={json.dumps(dict(dist), sort_keys=True)}, "
            f"accepted={json.dumps(accepted, sort_keys=True)}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"


@case("smoke_scheduling_s7", modes=["direct", "queue"], source="scheduling_smoke.py S7")
def s7_cas_fairness(ctx: CaseContext):
    ops = ctx.ops()
    base = rid_base(ctx, "scheduling")
    try:
        rids = [ops.next_request_id(base) for _ in range(20)]

        def run(rid):
            keys = [rid * 100 + j for j in range(3)]
            return ops.run_one_request(
                rid, output_len=2, block_keys=keys, stream_timeout_s=STREAM_TIMEOUT_S
            )

        with ThreadPoolExecutor(max_workers=20) as pool:
            results = list(pool.map(run, rids))
        addrs = []
        for rid, (addr, err) in zip(rids, results):
            if err:
                return False, f"rid={rid} failed: {err}"
            addrs.append(addr)
        addr_map = ops.addr_to_name()
        dist = Counter(addr_map.get(a, a) for a in addrs)
        num_workers = len(dist)
        snap = ops.snapshot_by_name()
        accepted = {
            name: info.get("accepted", 0)
            for name, info in snap.items()
            if info.get("role") == "prefill"
        }
        passed = num_workers >= 2
        return passed, (
            f"workers={num_workers}, "
            f"dist={json.dumps(dict(dist), sort_keys=True)}, "
            f"accepted={json.dumps(accepted, sort_keys=True)}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"


@case("smoke_scheduling_s8", modes=["direct", "queue"], source="scheduling_smoke.py S8")
def s8_ttft_sorting(ctx: CaseContext):
    ops = ctx.ops()
    perf_engines = []
    try:
        prefill_names = _prefill_names(ops)
        if len(prefill_names) < 2:
            return False, "need >=2 prefill workers"
        fast, slow = prefill_names[0], prefill_names[1]
        ops.set_perf(fast, prefill_fixed_ms=10.0)
        ops.set_perf(slow, prefill_fixed_ms=500.0)
        perf_engines = [fast, slow]
        time.sleep(1.5)  # master perf sync

        addrs = []
        for _ in range(10):
            rid = ops.next_request_id(rid_base(ctx, "scheduling"))
            keys = [rid * 100 + j for j in range(3)]
            addr, err = ops.run_one_request(
                rid, output_len=2, block_keys=keys, stream_timeout_s=STREAM_TIMEOUT_S
            )
            if err:
                return False, f"rid={rid} failed: {err}"
            addrs.append(addr)
        addr_map = ops.addr_to_name()
        dist = Counter(addr_map.get(a, a) for a in addrs)
        fast_count = dist.get(fast, 0)
        slow_count = dist.get(slow, 0)
        passed = fast_count >= slow_count
        return passed, (
            f"fast={fast}({fast_count}), slow={slow}({slow_count}), "
            f"dist={json.dumps(dict(dist), sort_keys=True)}, "
            f"assertion=fast>=slow"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        for eng in perf_engines:
            try:
                ops.set_perf(eng, prefill_fixed_ms=100.0)
            except Exception:
                pass


@case(
    "smoke_scheduling_s9",
    modes=["direct", "queue"],
    source="scheduling_smoke.py S9 (Java-corrected)",
)
def s9_no_hard_filter(ctx: CaseContext):
    """High queue-depth limit must not block routing.

    Java-corrected semantics: /set_queue_depth is a real enqueue rejection
    gate (reject when pendingRequests >= limit).  50000 is far above any
    reachable pending count, so the gate never fires and requests still
    route to the target worker — the legacy ``target_count > 0`` assertion
    is preserved and now documents exactly that behaviour.
    """
    ops = ctx.ops()
    injected_engine = None
    try:
        prefill_names = _prefill_names(ops)
        if len(prefill_names) < 2:
            return False, "need >=2 prefill workers"
        target = prefill_names[0]
        ops.set_queue_depth(target, 50000)
        injected_engine = target

        addrs = []
        for _ in range(5):
            rid = ops.next_request_id(rid_base(ctx, "scheduling"))
            keys = [rid * 100 + j for j in range(3)]
            addr, err = ops.run_one_request(
                rid, output_len=2, block_keys=keys, stream_timeout_s=STREAM_TIMEOUT_S
            )
            if err:
                return False, f"rid={rid} failed: {err}"
            addrs.append(addr)
        addr_map = ops.addr_to_name()
        dist = Counter(addr_map.get(a, a) for a in addrs)
        target_count = dist.get(target, 0)
        passed = target_count > 0
        return passed, (
            f"target={target}({target_count}), "
            f"dist={json.dumps(dict(dist), sort_keys=True)}, "
            f"note=queue_depth_limit=50000 is a real Java reject gate; "
            f"never fires at this height so routing is unaffected"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        if injected_engine:
            try:
                ops.set_queue_depth(injected_engine, 0)
            except Exception:
                pass


@case("smoke_scheduling_s10", source="scheduling_smoke.py S10")
def s10_weighted_random(ctx: CaseContext):
    ops = ctx.ops()
    try:
        for _ in range(50):
            rid = ops.next_request_id(rid_base(ctx, "scheduling"))
            keys = [rid * 100 + j for j in range(3)]
            _, err = ops.run_one_request(
                rid, output_len=2, block_keys=keys, stream_timeout_s=STREAM_TIMEOUT_S
            )
            if err:
                return False, f"rid={rid} failed: {err}"
        snap = ops.snapshot_by_name()
        completed = {
            name: info.get("completed", 0)
            for name, info in snap.items()
            if info.get("role") == "decode"
        }
        used = sum(1 for v in completed.values() if v > 0)
        passed = used >= 3
        return passed, (
            f"decode_workers={used}, "
            f"distribution={json.dumps(completed, sort_keys=True)}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"


@case("smoke_scheduling_s11", source="scheduling_smoke.py S11")
def s11_kv_capacity_filter(ctx: CaseContext):
    ops = ctx.ops()
    injected_engine = None
    try:
        snap = ops.snapshot_by_name()
        decode_names = sorted(
            name for name, e in snap.items() if e.get("role") == "decode"
        )
        if len(decode_names) < 2:
            return False, "need >=2 decode workers"
        target = decode_names[0]
        target_info = snap[target]
        avail = target_info.get("available_kv_tokens", 0)
        active = target_info.get("active_kv_tokens", 0)
        total_kv = avail + active
        ops.set_kv_pressure(target, total_kv)
        injected_engine = target
        time.sleep(1.0)  # master sync

        snap_sync = ops.snapshot_by_name()
        completed_before = snap_sync.get(target, {}).get("completed", 0)

        for _ in range(10):
            rid = ops.next_request_id(rid_base(ctx, "scheduling"))
            keys = [rid * 100 + j for j in range(3)]
            _, err = ops.run_one_request(
                rid, output_len=2, block_keys=keys, stream_timeout_s=STREAM_TIMEOUT_S
            )
            if err:
                return False, f"rid={rid} failed: {err}"

        snap2 = ops.snapshot_by_name()
        completed = {
            name: info.get("completed", 0)
            for name, info in snap2.items()
            if info.get("role") == "decode"
        }
        target_delta = completed.get(target, 0) - completed_before
        passed = target_delta <= 1
        return passed, (
            f"target={target}(delta={target_delta}), "
            f"distribution={json.dumps(completed, sort_keys=True)}, "
            f"assertion=delta<=1"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        if injected_engine:
            try:
                ops.set_kv_pressure(injected_engine, 0)
            except Exception:
                pass


@case("smoke_scheduling_s12", source="scheduling_smoke.py S12")
def s12_reserve_weight_change(ctx: CaseContext):
    ops = ctx.ops()
    base = rid_base(ctx, "scheduling")
    try:
        snap0 = ops.snapshot_by_name()
        decode_names = sorted(
            name for name, e in snap0.items() if e.get("role") == "decode"
        )
        if len(decode_names) < 2:
            return False, "need >=2 decode workers"
        baseline = {name: snap0[name].get("completed", 0) for name in decode_names}

        rid_a = ops.next_request_id(base)
        keys_a = [rid_a * 100 + j for j in range(3)]
        _, err_a = ops.run_one_request(
            rid_a, output_len=2, block_keys=keys_a, stream_timeout_s=STREAM_TIMEOUT_S
        )
        if err_a:
            return False, f"request A failed: {err_a}"

        snap_a = ops.snapshot_by_name()
        delta_a = {
            name: snap_a[name].get("completed", 0) - baseline[name]
            for name in decode_names
        }
        a_worker = max(delta_a, key=delta_a.get) if any(delta_a.values()) else None
        if a_worker is None:
            return False, "could not identify A's decode worker"

        for _ in range(10):
            rid = ops.next_request_id(base)
            keys = [rid * 100 + j for j in range(3)]
            _, err = ops.run_one_request(
                rid, output_len=2, block_keys=keys, stream_timeout_s=STREAM_TIMEOUT_S
            )
            if err:
                return False, f"subsequent request failed: {err}"

        snap_f = ops.snapshot_by_name()
        total_delta = {
            name: snap_f[name].get("completed", 0) - baseline[name]
            for name in decode_names
        }
        a_total = total_delta.get(a_worker, 0)
        other_max = max(
            (total_delta[n] for n in decode_names if n != a_worker), default=0
        )
        passed = a_total <= other_max + 1
        return passed, (
            f"a_worker={a_worker}(total={a_total}), other_max={other_max}, "
            f"delta={json.dumps(total_delta, sort_keys=True)}, "
            f"assertion=a_total<=other_max+1"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"


# ===========================================================================
# Anomaly cases (anomaly_smoke.py E1-E3)
# ===========================================================================

TIMEOUT_WAIT_S = 5.0
ANOMALY_STREAM_TIMEOUT_S = 10.0
WORKER_RECOVERY_WAIT_S = 3.0


def _inject_all_prefill(ops, config: dict) -> list[str]:
    snap = ops.snapshot()
    prefill_names = [
        e["name"] for e in snap.get("engines", []) if e.get("role") == "prefill"
    ]
    for name in prefill_names:
        ops.inject(name, config)
    return prefill_names


def _clear_all_prefill_inject(ops, names: list[str]) -> None:
    for name in names:
        try:
            ops.clear_inject(name)
        except Exception:
            pass


@case("smoke_anomaly_e1", source="anomaly_smoke.py E1")
def e1_cancel_path(ctx: CaseContext):
    ops = ctx.ops()
    rid = ops.next_request_id(rid_base(ctx, "anomaly"))
    try:
        response = ops.schedule(rid)
        if response.code != 200 or not response.success:
            return False, f"schedule failed: {response.error_message}"
        input_pb = (
            None if response.enqueued_by_master else ops.build_generate_input(rid)
        )
        handle = ops.start_stream(response, rid, input_pb=input_pb)
        if not handle.wait_first_output():
            handle.cancel()
            return False, "no output received before cancel window"
        cancel_at = time.monotonic()
        ops.cancel(rid, response)
        ended = handle.wait_end(5.0)
        cancel_latency = time.monotonic() - cancel_at
        recovery_ok, recovery_msg = ops.verify_recovery()
        if ctx.mode == "batch":
            inflight_ok, inflight_detail = AssertUtils.inflight_clean(
                _master_http(ops), 10.0
            )
        else:
            inflight_ok, inflight_detail = True, "N/A (non-batch path)"
        passed = ended and recovery_ok
        return passed, (
            f"cancel_latency={cancel_latency:.3f}s, stream_terminated={ended}, "
            f"outputs={len(handle.snap.outputs)}, "
            f"inflight_clean={inflight_ok}({inflight_detail}), recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"


def _anomaly_error_case(
    ctx: CaseContext, inject_config: dict, wait_s: float, require_error_detail: bool
) -> tuple[bool, str]:
    ops = ctx.ops()
    rid = ops.next_request_id(rid_base(ctx, "anomaly"))
    error_observed = False
    error_detail = "no error observed"
    injected_names: list[str] = []
    response = None
    try:
        injected_names = _inject_all_prefill(ops, inject_config)
        try:
            response = ops.schedule(rid)
            if response.code != 200 or not response.success:
                error_observed = True
                error_detail = f"schedule error: {response.error_message}"
            else:
                input_pb = (
                    None
                    if response.enqueued_by_master
                    else ops.build_generate_input(rid)
                )
                handle = ops.start_stream(response, rid, input_pb=input_pb)
                ended = handle.wait_end(wait_s)
                if not ended:
                    error_observed = True
                    error_detail = f"stream timed out (no response within {wait_s}s)"
                if handle.snap.error:
                    error_observed = True
                    error_detail = f"stream error: {handle.snap.error}"
                elif require_error_detail and not handle.snap.completed:
                    error_observed = True
                    error_detail = "stream did not complete"
        except Exception as exc:
            error_observed = True
            error_detail = f"exception: {exc!r}"
        finally:
            _clear_all_prefill_inject(ops, injected_names)

        # Explicitly cancel the failed request to clean up server-side
        # inflight (scheduler keeps the entry until TTL eviction otherwise).
        if response is not None and response.success:
            try:
                ops.cancel(rid, response)
            except Exception:
                pass

        time.sleep(WORKER_RECOVERY_WAIT_S)
        recovery_ok, recovery_msg = ops.verify_recovery()
        if ctx.mode == "batch":
            inflight_ok, inflight_detail = AssertUtils.inflight_clean(
                _master_http(ops), 10.0
            )
        else:
            inflight_ok, inflight_detail = True, "N/A (non-batch path)"
        passed = error_observed and recovery_ok
        return passed, (
            f"error_observed={error_observed} ({error_detail}), "
            f"inflight_clean={inflight_ok}({inflight_detail}), "
            f"recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"


@case("smoke_anomaly_e2", source="anomaly_smoke.py E2")
def e2_timeout(ctx: CaseContext):
    return _anomaly_error_case(ctx, {"no_respond": True}, TIMEOUT_WAIT_S, False)


@case("smoke_anomaly_e3", source="anomaly_smoke.py E3")
def e3_worker_fail(ctx: CaseContext):
    return _anomaly_error_case(
        ctx, {"enqueue_error": True}, ANOMALY_STREAM_TIMEOUT_S, True
    )
