"""Smoke test cases (functional correctness e2e).

Ported 1:1 from the legacy scripts (assertion thresholds preserved):

  smoke_cancel_t1..t6        <- cancel_smoke.py    T1-T6
  smoke_scheduling_s1..s12   <- scheduling_smoke.py S1-S12
  smoke_anomaly_e1..e3       <- anomaly_smoke.py   E1-E3

Profile adaptation (task #55; supersedes the legacy run_matrix_smoke.sh
mode grouping): the v1 batch/direct/queue split existed to vary the LB
strategy per mode, which v2 removed — all four profiles (batch-window /
single-nonbatch / single-batch / window-nonbatch) share QUEUE + FIFO
ordering and the same ESTIMATED_TTFT prefill selector, so scoping is by
delivery-path semantics instead:
  * S9 requires enqueue_batch — the queue-depth gate lives at the
    engine's EnqueueBatch entry only (structurally unreachable on the
    NON_BATCH GenerateStreamCall path)
  * everything else runs under all profiles; S4 opens the seed's client
    stream itself under NON_BATCH (see its docstring)

Behavioural corrections for the Java mock (documented inline):
  * S1 — the legacy "slow worker gets zero of 10 requests" assertion never
    held on the Java stack (it was inert in the legacy code due to the
    role_addr bug): COST_BASED_PREFILL's score is ledgerWaitMs + FORMULA
    estimate (a pure function of the request's token shape) + batcherWaitMs,
    so an engine's *speed* (set_perf 100 -> 200ms) is invisible to the score
    and serial requests leave no backlog — both engines score identically
    and RANDOM_WITHIN_TOLERANCE samples uniformly within the tie window.
    S1 now asserts the balance contract (both engines used, neither above
    80% of 20 requests); slow-engine avoidance is a *backlog* signal and is
    asserted by S4 (real pending hotspot).
  * S6 — the legacy "all 5 requests land on one worker" assertion assumed
    deterministic selection, but RANDOM_WITHIN_TOLERANCE uniformly samples
    the tie window per request by design (determinism needs BEST_ONLY), so
    5 requests only co-located with probability 2*(1/2)^4 = 12.5%.  S6 now
    asserts the same balance contract as S1 on 20 equivalent requests.
  * S4 — legacy injected a fake ``queue_depth`` display value (80000) that the
    Java mock implements as a *real* enqueue rejection gate
    (FaultInjectionConfig.queueDepthLimit); a huge limit never triggers, so
    the legacy "requests avoid the hot worker" assertion no longer holds via
    that knob.  The Java-true way to build a hotspot is real ledger load:
    a big seed request (input_len 49152 -> FORMULA estimate ~49s) lands on
    one engine, the mock snapshot confirms the seed is really enqueued
    there, and every subsequent request must deterministically avoid it
    (score gap ~49s >> the ~205ms tolerance window; outlier rejection also
    removes the loaded engine).  The engine that did NOT get the seed is
    restored to fast perf so it drains instantly and its ledger wait stays
    ~0 — otherwise the verification wave itself piles pending onto the cool
    engine and the load balancer correctly sees-saws back (an earlier port
    asserted "4 concurrent requests concentrate 4/0 on one engine", which
    only holds when the whole burst is evaluated against one stale ledger
    snapshot; when the master processes it in two groups the second group
    sees the first group's pending and splits 2/2 — correct balancing, not
    a defect).  The verification wave is serial with spacing so every
    request is evaluated against live ledger state: 0 of 5 requests may
    land on the heavy engine (deterministic, no concurrency window).
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


def case(name: str, profiles=None, requires=None, source: str = ""):
    def deco(fn):
        SMOKE_CASES.append(
            CaseDef(
                name=name,
                suite="smoke",
                fn=fn,
                profiles=profiles,
                requires=requires,
                source=source,
            )
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


@case("smoke_scheduling_s1", source="scheduling_smoke.py S1 (Java-corrected)")
def s1_load_balance(ctx: CaseContext):
    """Basic load balance: equivalent-score engines share traffic evenly.

    Java-corrected semantics: COST_BASED_PREFILL scores each engine as
    ledgerWaitMs + FORMULA estimate + batcherWaitMs.  The FORMULA estimate
    (default ``sum(computeTokens) + 0.3*sum(hitCacheTokens)``) depends only
    on the request's token shape, and serial requests complete before the
    next is scheduled, so a *speed* difference (set_perf 100 -> 200ms) never
    shows up in the score — with no backlog both engines tie and
    RANDOM_WITHIN_TOLERANCE picks uniformly.  Slow-engine avoidance is a
    backlog signal and belongs to S4.  Balance contract: of 20 serial
    requests both prefills are used and neither takes more than 80%.
    """
    ops = ctx.ops()
    n = 20
    is_batch = ctx.batch_dispatch()
    perf_engine = None
    try:
        if is_batch:
            prefill_names = _prefill_names(ops)
            if len(prefill_names) >= 2:
                ops.set_perf(prefill_names[1], prefill_fixed_ms=200.0)
                perf_engine = prefill_names[1]
                # Keep the slow injection as a regression guard: it must not
                # break the even distribution (the score model ignores
                # engine speed, only backlog matters — see S4).
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

        passed = num_workers >= 2 and max_ratio <= 0.8
        slow_detail = ""
        if is_batch and perf_engine:
            slow_count = dist_names.get(perf_engine, 0)
            slow_detail = f", slow_worker={perf_engine}({slow_count}) — observational: engine speed is not part of the COST_BASED score"
        return passed, (
            f"requests={n}, workers={num_workers}, max_ratio={max_ratio:.2f}, "
            f"distribution={json.dumps(dist_names, sort_keys=True)}, "
            f"snapshot_accepted={json.dumps(accepted, sort_keys=True)}, "
            f"assertion=workers>=2 and max_ratio<=0.8 (tie-window uniform "
            f"sampling){slow_detail}"
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
    source="scheduling_smoke.py S4 (Java-corrected)",
)
def s4_hotspot_filter(ctx: CaseContext):
    """Hotspot avoidance: requests divert away from the engine with real load.

    Java-corrected scenario (see module docstring): the hotspot is built
    from *real* ledger load, not the legacy fake queue_depth knob.  Two
    earlier race flaws, both fixed here:

    * waiting for a burst to complete drained the pending back to zero
      before the assertion had any basis (the original port);
    * asserting that a 4-request concurrent burst "concentrates 4/0" only
      holds when the whole burst is evaluated against one stale ledger
      snapshot — when the master processes the burst in two groups the
      second group sees the first group's fresh pending and splits 2/2.
      That seesaw is *correct* load balancing, not a defect, so the
      deterministic form is a serial wave with spacing:

      1. slow both prefill engines to 5s (the seed must survive on either
         landing spot) and let the master sync
      2. seed one fire-and-forget request with input_len=49152: the ledger
         predicts ~49s, so whichever engine X it lands on stays heavy for
         the whole observation window
      3. poll the mock snapshot until a prefill engine reports
         waiting+running >= 1 — that engine is X, and the poll proves the
         batcher really dispatched the seed (no master-ledger guesswork)
      4. restore the other engine T to 100ms so T drains instantly
         (ledger wait ~0); X stays slow at 5s
      5. wave: 5 serial requests (0.4s spacing), default shape — each is
         evaluated against live ledger state: X's wait (~49s) vs T's (~0)
         is far past the tolerance window (~205ms), and outlier rejection
         removes X outright
      6. assertion (deterministic): 0 of 5 requests land on X

    Profile semantics (v2, task #55): the hotspot lives in the master-side
    request ledger, which books BOTH delivery modes — BATCH groups through
    the inflight-batch accounting and NON_BATCH/direct requests through
    PrefillEndpoint.registerDirectRequest — so the deterministic-avoidance
    contract holds under every profile.  The only profile-dependent step is
    the seed's dispatch proof: under BATCH the master enqueues the
    fire-and-forget seed itself (the engine shows pending with no client
    action), while under NON_BATCH the master only publishes a route
    decision, so the seed's client stream is OPENED at fire time (never
    consumed) for the engine-side pending the hotspot poll needs.
    """
    ops = ctx.ops()
    base = rid_base(ctx, "scheduling")
    try:
        prefill_names = _prefill_names(ops)
        if len(prefill_names) < 2:
            return False, "need >=2 prefill workers"

        for name in prefill_names:
            ops.set_perf(name, prefill_fixed_ms=5000.0)
        time.sleep(1.5)  # master syncs the slowed perf before we seed

        addr_map = ops.addr_to_name()
        fired: list[tuple[int, object]] = []  # (rid, response) — cancelled in finally
        fired_handles: dict[int, object] = {}  # rid -> stream handle (NON_BATCH)

        def fire(rid: int, **kwargs):
            """Schedule without consuming the stream — keeps it pending."""
            resp = ops.schedule(rid, **kwargs)
            if resp.code != 200 or not resp.success:
                return None, f"schedule failed: {resp.error_message}"
            fired.append((rid, resp))
            if resp.enqueued_by_master:
                return addr_map.get(ops.role_addr(resp, "PREFILL"), ""), None
            # NON_BATCH: the master only published the route decision; the
            # engine sees the seed when the CLIENT opens the stream.  Open it
            # fire-and-forget (never wait) so the engine-side pending the
            # hotspot poll needs really exists.
            input_pb = ops.build_generate_input(rid, **kwargs)
            try:
                fired_handles[rid] = ops.start_stream(resp, rid, input_pb=input_pb)
            except Exception as exc:
                return None, f"seed direct stream failed to open: {exc!r}"
            return addr_map.get(ops.role_addr(resp, "PREFILL"), ""), None

        # -- seed: big ledger footprint, fire-and-forget.  input_len=49152
        #    keeps the seed's predicted wait (~49s) far above anything the
        #    wave can pile onto the cool engine (~10s max) for the whole
        #    observation window.
        seed_rid = ops.next_request_id(base)
        seed_name, err = fire(seed_rid, input_len=49152, output_len=2)
        if err:
            return False, f"seed request failed: {err}"
        if seed_name not in prefill_names:
            return False, f"seed request went to unknown worker {seed_name}"

        # -- poll the mock snapshot until the seed really shows up on an
        #    engine: proves the batcher dispatched it and identifies the
        #    hot engine without guessing at master-side ledger state.
        deadline = time.monotonic() + 6.0
        hot = None
        while time.monotonic() < deadline and hot is None:
            snap = ops.snapshot_by_name()
            for name in prefill_names:
                info = snap.get(name, {})
                if info.get("waiting", 0) + info.get("running", 0) >= 1:
                    hot = name
                    break
            if hot is None:
                time.sleep(0.1)
        if hot is None:
            return False, "seed never appeared on any engine (engine side)"
        if hot != seed_name:
            return False, (f"seed routed to {seed_name} but pending showed up on {hot}")
        cool = next(n for n in prefill_names if n != hot)

        # -- restore the cool engine to fast so it drains instantly; its
        #    ledger wait stays ~0 and the wave cannot pile up on it.
        ops.set_perf(cool, prefill_fixed_ms=100.0)
        time.sleep(0.3)

        # -- verification wave: serial with spacing — every request is
        #    evaluated against live ledger state (no stale-snapshot race).
        wave_names = []
        for i in range(5):
            rid = ops.next_request_id(base)
            name, err = fire(rid, output_len=2)
            if err:
                return False, f"rid={rid} failed: {err}"
            wave_names.append(name)
            if i < 4:
                time.sleep(0.4)

        dist = Counter(wave_names)
        hot_count = dist.get(hot, 0)
        cool_count = dist.get(cool, 0)
        passed = hot_count == 0
        return passed, (
            f"hot={hot}({hot_count}), cool={cool}({cool_count}), "
            f"dist={json.dumps(dict(dist), sort_keys=True)}, "
            f"seed={seed_name}, "
            f"assertion=0 of 5 requests land on the heavy engine "
            f"(deterministic avoidance; serial wave, no concurrency window)"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        try:
            ops.set_perf(prefill_names[0], prefill_fixed_ms=100.0)
            ops.set_perf(prefill_names[1], prefill_fixed_ms=100.0)
        except Exception:
            pass
        # Fire-and-forget requests never consume FetchResponse, so their
        # master-side inflight/ledger entries can linger long after the
        # engine finished (observed: inflight_clean timing out at 90s; the
        # seed's ~49s ledger prediction then kept one engine's wait high
        # for the rest of the suite and poisoned later balance assertions —
        # S5 lost its cache affinity and S6 drew 20/20 onto the other
        # engine).  Cancelling alone does not reliably clear those entries
        # either: for a stream nobody ever fetched, the cancel/batch-
        # completion race is nondeterministic.  The deterministic cleanup
        # is the normal completion path every other case uses — consume
        # each fired request's FetchResponse to terminal state (the seed's
        # ~5s prefill is the only slow one), with cancel as fallback.
        for rid, resp in fired:
            try:
                if rid in fired_handles:
                    # NON_BATCH: the direct stream opened at fire time IS the
                    # completion path — consume it to terminal state.
                    fired_handles[rid].wait_end(20.0)
                else:
                    ops.start_stream(resp, rid).wait_end(20.0)
            except Exception:
                try:
                    ops.cancel(rid, resp)
                except Exception:
                    pass
        try:
            AssertUtils.inflight_clean(_master_http(ops), 30.0)
        except Exception:
            pass


@case("smoke_scheduling_s5", source="scheduling_smoke.py S5")
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


@case(
    "smoke_scheduling_s6",
    source="scheduling_smoke.py S6 (Java-corrected)",
)
def s6_cost_based_tie_window_balance(ctx: CaseContext):
    """Equivalent-cost engines are tied, not deterministic — assert balance.

    The legacy ``all 5 requests land on one worker`` assertion assumed
    COST_BASED_PREFILL selects deterministically, but the selector is
    RANDOM_WITHIN_TOLERANCE by design: equal scores form a tie window
    (max(score*10%, 20ms)) and the winner is uniformly sampled per request
    (reservoir sampling in selectBaselineCandidate) — 5 requests only
    co-located with probability 2*(1/2)^4 = 12.5%.  A deterministic pick
    would need candidateChoice BEST_ONLY.  The real contract: equivalent
    engines share traffic; of 20 requests both prefills are used and
    neither takes more than 80%.
    """
    ops = ctx.ops()
    try:
        addrs = []
        for _ in range(20):
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
        max_ratio = max(dist.values()) / len(addrs)
        snap = ops.snapshot_by_name()
        accepted = {
            name: info.get("accepted", 0)
            for name, info in snap.items()
            if info.get("role") == "prefill"
        }
        passed = num_workers >= 2 and max_ratio <= 0.8
        return passed, (
            f"workers={num_workers}, max_ratio={max_ratio:.2f}, "
            f"dist={json.dumps(dict(dist), sort_keys=True)}, "
            f"accepted={json.dumps(accepted, sort_keys=True)}, "
            f"assertion=workers>=2 and max_ratio<=0.8 (tie-window uniform "
            f"sampling, not determinism)"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"


@case("smoke_scheduling_s7", source="scheduling_smoke.py S7")
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


@case(
    "smoke_scheduling_s8",
    source="scheduling_smoke.py S8 (Java-corrected)",
)
def s8_ttft_sorting(ctx: CaseContext):
    """ESTIMATED_TTFT tie-window balance: equivalent-score engines share traffic.

    Java-corrected semantics: the legacy S8 asserted that a fast engine
    (set_perf 10ms) receives at least as many requests as a slow one
    (500ms), assuming the TTFT sort "sees" engine speed.  It does not:
    the v2 prefill selector (ESTIMATED_TTFT in the harness config) scores
    ledger wait + FORMULA estimate (a pure function of the request's token
    shape) — engine perf is not a score input, exactly like COST_BASED
    (see S1).  Serial requests with no backlog leave both engines tied,
    so the split is a uniform draw (P(fast < slow) ≈ 38% for 10 requests);
    the historical passes leaned on a settle-latency side effect — a
    request on the slow engine leaves its ledger entry alive longer,
    nudging later requests away — which prompt settling does not
    reproduce.  The real contract is the same balance as S1: of 20
    requests both prefills are used and neither takes more than 80%; the
    perf split is observational only.  Dispatcher/decision axes are
    invisible to the selector, so the case runs under all profiles.
    """
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
        for _ in range(20):
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
        num_workers = len(dist)
        max_ratio = max(dist.values()) / len(addrs)
        passed = num_workers >= 2 and max_ratio <= 0.8
        return passed, (
            f"fast={fast}({fast_count}), slow={slow}({slow_count}), "
            f"dist={json.dumps(dict(dist), sort_keys=True)}, "
            f"assertion=workers>=2 and max_ratio<=0.8 (tie-window uniform "
            f"sampling; engine perf is not part of the SHORTEST_TTFT score — "
            f"perf split is observational)"
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
    requires=["enqueue_batch"],
    source="scheduling_smoke.py S9 (Java-corrected)",
)
def s9_no_hard_filter(ctx: CaseContext):
    """High queue-depth limit must not block routing.

    Java-corrected semantics: /set_queue_depth is a real enqueue rejection
    gate (reject when pendingRequests >= limit).  50000 is far above any
    reachable pending count, so the gate never fires and requests still
    route to the target worker — the legacy ``target_count > 0`` assertion
    is preserved and now documents exactly that behaviour.

    Profile semantics (v2, task #55): the gate is checked ONLY at the
    engine's EnqueueBatch entry (JavaMockEngineCluster.enqueueBatch);
    the GenerateStreamCall path (NON_BATCH dispatcher) never consults it,
    so "the high gate does not block routing" is only observable where
    the master actually enqueues — requires=["enqueue_batch"] keeps the
    case to the BATCH-dispatch profiles (batch-window, single-batch).
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


@case("smoke_scheduling_s12", source="scheduling_smoke.py S12 (Java-corrected)")
def s12_reserve_weight_change(ctx: CaseContext):
    """Decode single-point collapse guard after request A.

    Java-corrected semantics: the legacy S12 asserted COST_BASED_DECODE's
    reserve weight-lowering (after request A, later requests lean away from
    A's worker), but the Java stack's decode selector is
    KV_USAGE_WEIGHTED_RANDOM — the reserve weight-lowering mechanism does
    not exist here.  11 requests are weighted-random draws whose weights
    track per-worker KV residue left by earlier cases, so *any* bound on
    the per-worker share is a coin flip (observed flaps: {6,4,1,0} then
    {0,3,7,1} with the busiest worker taking 7 while a fresh-env run
    passed 3/3 with near-uniform draws).  The mechanism-backed assertions
    live elsewhere — S10 (distribution sanity over 50 requests) and S11
    (KV-pressure avoidance) — so what remains uniquely assertable here is
    the collapse guard: 11 requests must not all land on one decode
    worker (>=2 workers used), which would only happen if the weighted
    random selector degenerated to a single candidate.
    """
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
        used = sum(1 for v in total_delta.values() if v > 0)
        busiest = max(total_delta.values())
        passed = used >= 2
        return passed, (
            f"a_worker={a_worker}(total={a_total}), other_max={other_max}, "
            f"busiest={busiest}, used={used}/{len(decode_names)}, "
            f"delta={json.dumps(total_delta, sort_keys=True)}, "
            f"assertion=used>=2 (collapse guard; Java decode selector is "
            f"KV_USAGE_WEIGHTED_RANDOM with no reserve weight-lowering — "
            f"distribution sanity is S10, KV-pressure avoidance is S11)"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"


# ===========================================================================
# Anomaly cases (anomaly_smoke.py E1-E3)
# ===========================================================================

TIMEOUT_WAIT_S = 5.0
ANOMALY_STREAM_TIMEOUT_S = 10.0
WORKER_RECOVERY_WAIT_S = 3.0
# E4 probe: short client-side gRPC deadline proving the parked Schedule RPC
# stays pending (the master's own scheduling deadline is queueTimeoutMs,
# default 1h — far beyond any useful probe).
E4_PROBE_DEADLINE_S = 5.0


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
        if response.enqueued_by_master:
            inflight_ok, inflight_detail = AssertUtils.inflight_clean(
                _master_http(ops), 10.0
            )
        else:
            # NON_BATCH: a client Cancel on a delivered request cannot
            # safely release the master ledger entry (fence probe NOT_FOUND
            # is not a safe-release fact), so immediate-zero is not a
            # contract here — see E4's watermark rationale.
            inflight_ok, inflight_detail = True, "N/A (NON_BATCH residue contract)"
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
        if response is not None and response.success and response.enqueued_by_master:
            inflight_ok, inflight_detail = AssertUtils.inflight_clean(
                _master_http(ops), 10.0
            )
        else:
            # NON_BATCH: see E1 — client Cancel cannot safely release a
            # delivered ledger entry, so immediate-zero is not asserted.
            inflight_ok, inflight_detail = True, "N/A (NON_BATCH residue contract)"
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


@case(
    "smoke_anomaly_e4",
    source="decode-side anomaly gap (G1) — new",
)
def e4_decode_capacity_exhausted(ctx: CaseContext):
    """Decode-side anomaly: every decode engine KV-exhausted -> the request
    is parked undelivered, a master Cancel releases it without residue, and
    clearing the pressure recovers routing.

    Why KV pressure and not the E-series ops.inject faults: a decode engine
    in the Java mock never receives traffic through any gRPC entry point.
    After prefill completes, the request is handed off IN-PROCESS
    (JavaMockEngineCluster.FastRpcService.startDecode ->
    scheduleDecodeCompletion), so enqueue_error / generate_error /
    fetch_error — all checked at the enqueueBatch / generateStreamCall /
    fetchResponse RPC entries — never fire for a decode engine, and
    no_respond on decode only suppresses the "intermediate first-step
    output" which the mock never produces (each request yields exactly one
    finished message).  The one decode-side anomaly observable e2e is KV
    capacity: the delivery-capacity admission hard-filters every decode
    endpoint whose available_kv_tokens < seq_len, so exhausting every
    decode engine's KV must block delivery.

    v2 contract (task #55, source-verified — supersedes the v1 fail-fast
    assertion): the QUEUE scheduler treats decode KV exhaustion as a WAIT
    condition, not a fail-fast rejection:
      * FixedWindowBatcherAlgorithm parks the head when delivery capacity
        cannot be reserved ("Dynamic KV pressure is a wait condition, not a
        rejection"; BatcherContext.admitAndDeliverCapacityFeasiblePrefix
        returns CapacityBlocked and the worker loop waits for the exact
        resource-change event).
      * The scheduling deadline is owned by the queue config
        (QueueSchedulerConfig.queueTimeoutMs, default 1h), not the caller,
        so the Schedule RPC stays pending while parked — the client
        observes its own gRPC DEADLINE_EXCEEDED instead of a rejection
        response.  The pre-v2 fail-fast NO_AVAILABLE_WORKER contract
        belonged to the v1 non-QUEUE flow and does not exist in v2.
      * A client-side RPC deadline/cancellation does NOT release the
        parked entry (it lingered until the stale-inflight TTL eviction in
        the repro); an explicit master Cancel does
        (PriorityScheduler.cancelRequest -> isLocallyReversible -> local
        cleanup).

    Scenario:
      1. set active_kv_tokens = total on every decode engine
      2. probe Schedule with a short client-side deadline: it must stay
         pending (client DEADLINE_EXCEEDED, no rejection response) and the
         parked rid must NOT be delivered to any engine
      3. master Cancel must release the parked request and leave no
         inflight residue
      4. clear the pressure -> a fresh request must complete again

    Profile semantics (v2): the decision and dispatcher axes are invisible
    to the decode-side delivery capacity gate — both delivery modes share
    the per-worker batcher and the same capacity admission — so the case
    runs under all profiles.  The no-residue assertion is a pre-probe
    WATERMARK comparison rather than a global zero check: under NON_BATCH
    dispatch a client-side Cancel cannot safely release a delivered
    request's master ledger entry (the fence probe's NOT_FOUND ack is not
    a safe-release fact — the client connects to the engine
    asynchronously after RouteDecision), so earlier E-series requests on
    the shared env may leave contract-parked entries; this case only owns
    the residue of ITS OWN parked probe.
    """
    ops = ctx.ops()
    base = rid_base(ctx, "anomaly")
    injected: list[str] = []
    try:
        snap = ops.snapshot_by_name()
        decode_names = sorted(
            name for name, e in snap.items() if e.get("role") == "decode"
        )
        if not decode_names:
            return False, "no decode workers found"

        # Exhaust every decode engine: active = total -> available = 0.
        for name in decode_names:
            info = snap[name]
            total_kv = int(info.get("available_kv_tokens", 0)) + int(
                info.get("active_kv_tokens", 0)
            )
            ops.set_kv_pressure(name, total_kv)
            injected.append(name)
        time.sleep(1.5)  # master worker-status sync

        # 1. The probe stays pending: a short client-side deadline fires
        #    instead of the master returning a rejection.
        base_view = ops.master_inflight() or {}

        def _inflight_totals(view: dict) -> tuple[int, int, int]:
            return (
                int(view.get("scheduler_inflight", 0) or 0),
                sum(
                    int(ep.get("inflight_batches", 0) or 0)
                    for ep in view.get("prefill_endpoints", []) or []
                ),
                sum(
                    int(ep.get("inflight_requests", 0) or 0)
                    for ep in view.get("decode_endpoints", []) or []
                ),
            )

        base_sched, base_prefill, base_decode = _inflight_totals(base_view)
        rid = ops.next_request_id(base)
        probe: dict = {}

        def _probe() -> None:
            try:
                resp = ops.schedule(rid, timeout_s=E4_PROBE_DEADLINE_S)
                probe["returned"] = (
                    f"code={resp.code}, success={resp.success}, "
                    f"error={resp.error_message!r}"
                )
            except Exception as exc:  # client deadline while parked
                code_fn = getattr(exc, "code", None)
                probe["grpc_code"] = str(code_fn()) if callable(code_fn) else ""
                probe["exc"] = repr(exc)

        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(_probe).result(timeout=E4_PROBE_DEADLINE_S + 10.0)
        parked_ok = probe.get("grpc_code") == "StatusCode.DEADLINE_EXCEEDED"
        parked_detail = probe.get("returned") or probe.get(
            "grpc_code", probe.get("exc", "no outcome")
        )

        # 2. The parked request must not have been delivered to any engine.
        time.sleep(0.5)
        snap2 = ops.snapshot()
        delivered = [
            engine["name"]
            for engine in snap2.get("engines", [])
            if str(rid) in engine.get("request_lifecycle", {})
        ]
        not_delivered_ok = not delivered

        # 3. An explicit master Cancel releases the parked request: master
        #    inflight must return to the pre-probe watermark (scheduler
        #    entry + decode shadow reservation both released).
        cancel_err = None
        try:
            ops.cancel(rid, None)
        except Exception as exc:
            cancel_err = repr(exc)
        time.sleep(0.5)
        inflight_ok, inflight_detail = False, "no inflight view"
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            view = ops.master_inflight()
            if view is not None:
                sched, pre, dec = _inflight_totals(view)
                if sched <= base_sched and pre <= base_prefill and dec <= base_decode:
                    inflight_ok = True
                    inflight_detail = (
                        f"back to pre-probe watermark "
                        f"(scheduler={sched}/{base_sched}, "
                        f"prefill_batches={pre}/{base_prefill}, "
                        f"decode_reservations={dec}/{base_decode})"
                    )
                    break
                inflight_detail = (
                    f"scheduler={sched} (base {base_sched}), "
                    f"prefill_batches={pre} (base {base_prefill}), "
                    f"decode_reservations={dec} (base {base_decode})"
                )
            time.sleep(0.5)

        # 4. Clear the pressure on every decode engine; recovery must be
        #    functional, not just cosmetic: a fresh request must schedule
        #    and complete again.
        for name in injected:
            try:
                ops.set_kv_pressure(name, 0)
            except Exception:
                pass
        time.sleep(2.0)  # master worker-status sync (recovery view)

        rid_rec = ops.next_request_id(base)
        rec_addr, rec_err = ops.run_one_request(
            rid_rec, output_len=2, stream_timeout_s=STREAM_TIMEOUT_S
        )
        recovery_ok, recovery_msg = ops.verify_recovery()

        passed = (
            parked_ok
            and not_delivered_ok
            and cancel_err is None
            and inflight_ok
            and rec_err is None
            and recovery_ok
        )
        return passed, (
            f"parked_pending={parked_ok} ({parked_detail}), "
            f"delivered_while_parked={delivered or 'none'}, "
            f"cancel_err={cancel_err}, "
            f"recovered_request_ok={rec_err is None}"
            f"(prefill={rec_addr}, err={rec_err}), "
            f"inflight_clean={inflight_ok}({inflight_detail}), "
            f"recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        for name in injected:
            try:
                ops.set_kv_pressure(name, 0)
            except Exception:
                pass
