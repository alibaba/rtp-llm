from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from ...context import CaseContext, rid_base
from ...engine_ops import clear_type_all, engine_inflight_clean, inject_type
from ...harness import AssertUtils
from ...registry import case
from ...support.admission import (
    STREAM_TIMEOUT_S,
    _all_engines_busy,
    _master_http,
    _prefill_names,
)


@case(
    "admission_queue_depth_reject",
    category="admission",
    requires=["enqueue_batch"],
    source="gap G8: engine queue_depth admission gate (fast reject + recovery)",
)
def admission_queue_depth(ctx: CaseContext):
    """Engine-side queue_depth gate: once every prefill holds >=1 slow
    pending request, the next enqueue is rejected FAST with "queue depth
    limit exceeded" (-> BATCH_DISPATCH_FAILED schedule response), NOT an
    unbounded pile-up; after the gate is lifted the occupiers finish and
    a fresh request succeeds with no inflight leak.

    Profile semantics (v2): the gate is checked only at the
    engine's EnqueueBatch entry (BATCH dispatcher) — the
    GenerateStreamCall path never consults it — so
    requires=["enqueue_batch"] keeps the case to the BATCH-dispatch
    profiles (batch-window, single-batch).  The case runs on the shared
    default env (real per-profile config), so single-batch exercises the
    SINGLE decision axis on the same gate path.
    """
    ops = ctx.ops()
    base = rid_base(ctx, "admission")
    names = _prefill_names(ops)
    if not names:
        return False, "no prefill engines found"

    seed_pool: Optional[ThreadPoolExecutor] = None
    futures = []
    try:
        # Slow prefills keep the seed requests pending long enough.
        for n in names:
            ops.set_perf(n, prefill_fixed_ms=4000.0)
            inject_type(ops, n, "queue_depth", depth=1)

        # Fire seeds until EVERY prefill holds >=1 pending request, so the
        # probe request cannot dodge the gate by landing on a free engine.
        # With depth=1 some seeds may themselves be rejected — their errors
        # are expected and not part of the verdict.
        seed_pool = ThreadPoolExecutor(max_workers=8)
        deadline = time.monotonic() + 15.0
        while not _all_engines_busy(ops, names) and time.monotonic() < deadline:
            rid = ops.next_request_id(base)
            futures.append(
                seed_pool.submit(
                    ops.run_one_request,
                    rid,
                    input_len=512,
                    output_len=2,
                    stream_timeout_s=20.0,
                )
            )
            time.sleep(0.3)
        occupied = _all_engines_busy(ops, names)
        if not occupied:
            return False, "seeds never occupied every prefill engine"

        # Probe: must be fast-rejected with the queue-depth error.
        rid_probe = ops.next_request_id(base)
        t0 = time.monotonic()
        _, err_probe = ops.run_one_request(
            rid_probe, input_len=512, output_len=2, stream_timeout_s=20.0
        )
        reject_latency = time.monotonic() - t0

        # Lift the gate, let the seeds drain, then verify recovery.
        clear_type_all(ops, names, "queue_depth")
        for fut in futures:
            try:
                fut.result(timeout=30.0)
            except Exception:
                pass

        rid_after = ops.next_request_id(base)
        _, err_after = ops.run_one_request(
            rid_after, input_len=512, output_len=2, stream_timeout_s=STREAM_TIMEOUT_S
        )

        inflight_ok, inflight_detail = AssertUtils.inflight_clean(
            _master_http(ops), 30.0
        )
        engine_clean, engine_detail = engine_inflight_clean(ops, names)

        rejected = (
            err_probe is not None
            and "queue depth" in str(err_probe)
            and reject_latency < 3.0
        )
        passed = (
            occupied and rejected and err_after is None and inflight_ok and engine_clean
        )
        return passed, (
            f"all_engines_occupied={occupied}, "
            f"probe_rejected={rejected} "
            f"(latency={reject_latency:.2f}s, err={str(err_probe)[:80]}), "
            f"recovered={err_after is None}, "
            f"master_inflight_clean={inflight_ok}({inflight_detail}), "
            f"engine_inflight_clean={engine_clean}({engine_detail})"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        clear_type_all(ops, names, "queue_depth")
        try:
            for n in names:
                ops.set_perf(n, prefill_fixed_ms=100.0)
        except Exception:
            pass
        if seed_pool is not None:
            seed_pool.shutdown(wait=True)
