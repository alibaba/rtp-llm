from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from ...context import CaseContext, rid_base
from ...registry import case
from ...support.kv import E4_PROBE_DEADLINE_S, STREAM_TIMEOUT_S


@case(
    "kv_decode_capacity_park",
    category="kv",
    source="decode-side anomaly gap (G1) — new (anomaly E4 until the category reorg)",
)
def kv_decode_capacity_park(ctx: CaseContext):
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
    finished message).  The one decode-side anomaly observable end-to-end
    is KV capacity: the delivery-capacity admission hard-filters every
    decode endpoint whose available_kv_tokens < seq_len, so exhausting
    every decode engine's KV must block delivery.

    v2 contract (source-verified — supersedes the v1 fail-fast
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
    asynchronously after RouteDecision), so earlier requests on the
    shared env may leave contract-parked entries; this case only owns
    the residue of ITS OWN parked probe.
    """
    ops = ctx.ops()
    base = rid_base(ctx, "kv")
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
