from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from ...context import CaseContext, rid_base
from ...engine_ops import engine_inflight_clean
from ...grade import GradeReport
from ...harness import AssertUtils
from ...registry import case
from ...support.priority import (
    CODE_ENGINE_CANCELLED,
    CODE_OK,
    CODE_YIELDED,
    FIRE_GAP_S,
    PERF_SETTLE_S,
    _all_engine_names,
    _dr_live_spec,
    _engine_saw,
    _finally_hygiene,
    _master_http,
    _metric_lines,
    _metric_sum,
    _poll_engine_pending,
    _prefill_names,
    _scrape_master_metrics,
)


@case(
    "atpm_preempt_decode_reserved_live",
    category="priority",
    profiles=["single-batch"],
    source="preemption-stages audit (2026-09) — live DECODE_RESERVED eviction",
)
def atpm_preempt_decode_reserved_live(ctx: CaseContext):
    """LIVE DECODE_RESERVED eviction: the incoming's decode placement
    fails on the victim's shadow reservation and the master's local
    atomic eviction settles the victim at EXACTLY 8400 — never 8429
    (the discriminating feature of this path: an engine-owned victim
    would surface the typed-cancel 8429 terminal).

    ENV: BATCH dispatcher + the production-baseline stage set
    {PREFILL_QUEUED, DECODE_RESERVED} (master_fixed_window.json values —
    no engineCancellation because no engine-owned stage is enabled), a
    4-block decode KV pool (4096 tokens at blockSize=1024) on a SINGLE
    decode engine, 1P+1D, prefill slowed to 4s.

    Choreography (both shadows land on the one decode pool):
      * P90 placeholder input=512 dispatches to prefill — its decode
        shadow (512) is reserved on the single decode engine and P90
        stands above the incoming, so it is never a candidate;
      * P30 victim input=512 queues MASTER_QUEUED_NOT_DISPATCHED
        (prefill slot busy) — its decode shadow (512) is ALSO reserved
        (the precise AdmissionCapacity check runs because the stage set
        contains a decode stage), leaving hardAvailable = 4096−1024 =
        3072;
      * P70 incoming input=3500 > 3072 → BLOCKED(decode) → kvDeficit =
        428 ≤ victim freedKv = 512 → EvictionPlanner picks the P30
        shadow → master-local atomic eviction (8400), post-eviction
        capacity 4096−512 = 3584 ≥ 3500 → the incoming places.

    Contract:
      * victim terminal EXACTLY 8400 (asserted NOT 8429 — the stage's
        discriminating feature) and no engine ever saw the rid;
      * placeholder + incoming complete 200 with real FetchResponse
        output;
      * metric plane: auto_tpm.victim.count{stage=decode_reserved} == 1
        and auto_tpm.victim.kv_tokens{stage=decode_reserved} >= 428
        (the released shadow covers the incoming's kvDeficit);
      * the ledger closes clean and recovery works.
    """
    env = ctx.env_manager.ensure(_dr_live_spec(ctx))
    ops = ctx.engine_ops(env)
    report = GradeReport(run_grade=ctx.grade)
    base = rid_base(ctx, "priority")
    handles: dict = {}
    prefill_names: list = []
    try:
        prefill_names = _prefill_names(ops)
        for name in prefill_names:
            ops.set_perf(name, prefill_fixed_ms=4000.0)
        time.sleep(PERF_SETTLE_S)

        ph = ops.next_request_id(base)
        ph_resp = ops.schedule(
            ph, priority=90, input_len=512, output_len=2, timeout_s=90.0
        )
        if ph_resp.code != CODE_OK or not ph_resp.success:
            return False, f"placeholder schedule failed: {ph_resp.error_message}"
        if not _poll_engine_pending(ops, prefill_names[0], 1):
            return False, "placeholder never dispatched"

        victim = ops.next_request_id(base)
        inc = ops.next_request_id(base)
        with ThreadPoolExecutor(max_workers=2) as pool:
            v_future = pool.submit(
                ops.schedule,
                victim,
                priority=30,
                input_len=512,
                output_len=2,
                timeout_s=90.0,
            )
            time.sleep(FIRE_GAP_S)
            inc_future = pool.submit(
                ops.schedule,
                inc,
                priority=70,
                input_len=3500,
                output_len=2,
                timeout_s=90.0,
            )
            v_resp = v_future.result(timeout=95.0)
            inc_resp = inc_future.result(timeout=95.0)

        victim_evicted = v_resp.code == CODE_YIELDED and not v_resp.success
        victim_not_cancelled = v_resp.code != CODE_ENGINE_CANCELLED
        inc_ok = inc_resp.code == CODE_OK and inc_resp.success

        completed = {}
        for rid, resp in ((ph, ph_resp), (inc, inc_resp)):
            if resp.code != CODE_OK or not resp.success:
                continue
            handle = ops.start_stream(resp, rid)
            handles[rid] = handle
            ended = handle.wait_end(45.0)
            completed[rid] = ended and handle.snap.completed
        survivors_completed = all(completed.get(r) for r in (ph, inc))
        victim_never_seen = not _engine_saw(ops, victim)

        samples = _scrape_master_metrics(ops)
        dr_victim = _metric_sum(
            samples, "auto_tpm_victim_count", {"stage": "decode_reserved"}
        )
        dr_kv = _metric_sum(
            samples, "auto_tpm_victim_kv_tokens", {"stage": "decode_reserved"}
        )

        report.invariant(
            "PR10",
            victim_evicted and victim_not_cancelled and inc_ok and survivors_completed,
            context="live_decode_reserved_replacement",
            detail=(
                f"victim terminal={v_resp.code} (expected exactly "
                f"{CODE_YIELDED}, NEVER {CODE_ENGINE_CANCELLED} — the "
                f"stage's discriminating feature), incoming schedule="
                f"{inc_resp.code}, FetchResponse completed="
                f"{ {r % 1_000_000: completed.get(r) for r in (ph, inc)} }"
            ),
        )
        report.invariant(
            "PR5",
            victim_never_seen and victim_evicted,
            context="live_victim_shadow_never_delivered",
            detail=(
                f"victim terminal={v_resp.code}, engine ever saw rid="
                f"{_engine_saw(ops, victim)} (DECODE_RESERVED eviction is "
                f"a master-local atomic transaction on the shadow "
                f"reservation — the request itself was never dispatched)"
            ),
        )
        report.invariant(
            "PR6",
            victim_evicted and dr_victim == 1.0 and (dr_kv or 0.0) >= 428.0,
            context="live_decode_reserved_metrics",
            detail=(
                f"auto_tpm.victim.count{{stage=decode_reserved}}="
                f"{dr_victim} (expected 1), victim.kv_tokens="
                f"{dr_kv} (>=428 — the released shadow covers the "
                f"incoming's kvDeficit), samples="
                f"{_metric_lines(samples, 'auto_tpm_victim')}"
            ),
        )
        clean_ok, clean_detail = AssertUtils.inflight_clean(_master_http(ops), 30.0)
        engine_clean, engine_detail = engine_inflight_clean(
            ops, _all_engine_names(ops), 30.0
        )
        recovery_ok, recovery_msg = ops.verify_recovery()
        report.invariant(
            "P6",
            clean_ok and engine_clean and recovery_ok,
            detail=(
                f"inflight={'ok' if clean_ok else clean_detail}, "
                f"engine={'ok' if engine_clean else engine_detail}, "
                f"recovery={recovery_msg}"
            ),
        )
        return report.finish(
            f"live decode-reserved eviction: victim={v_resp.code} "
            f"(not 8429), survivors completed={survivors_completed}, "
            f"metric victim.count={dr_victim} kv_tokens={dr_kv}, "
            f"grades: {report.summary()}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        for handle in handles.values():
            handle.cancel()
        _finally_hygiene(ops, [], prefill_names)
