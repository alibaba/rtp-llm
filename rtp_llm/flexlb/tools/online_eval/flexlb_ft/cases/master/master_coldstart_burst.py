from __future__ import annotations

import json
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from ...context import CaseContext, rid_base
from ...registry import case
from ...support.master import STREAM_TIMEOUT_S, _coldstart_spec


@case(
    "master_coldstart_burst",
    category="master",
    profiles=["batch-window", "single-nonbatch", "single-batch", "window-nonbatch"],
    source="intake defect regression probe (cold-start first-connect storm)",
)
def coldstart_burst(ctx: CaseContext):
    """Fire 20 requests the instant the master reports ready.

    Regression probe for the three intake defects: CONNECT_TIMEOUT 20ms,
    3-strike dead marking on first connect, non-atomic getOrCreateWorkerStatus.
    Expected to FAIL or pass marginally today — the failure rate and the
    marked-dead sample count are recorded as the baseline for the intake fix.

    Profile semantics (v2): _coldstart_spec carries NO config
    override, so the case would genuinely exercise each profile's config
    (unlike the pinned-spec cases above); it stays scoped to batch-window
    this round as a deliberate scope decision — spreading the intake probe
    across profiles is later-phase work.
    """
    env = ctx.env_manager.ensure(_coldstart_spec(ctx))
    ops = ctx.engine_ops(env)
    base = rid_base(ctx, "master")
    expected = {"PREFILL": env.spec.n_prefill, "DECODE": env.spec.n_decode}
    try:
        # Sample worker_summary every 0.5s while the burst runs and for 10s
        # after — the cold-start window where engines get marked dead.
        samples: list[tuple[float, dict]] = []
        stop = threading.Event()

        def sample() -> Optional[dict]:
            info = ops.master_info()
            if not info:
                return None
            summary = info.get("worker_summary", {}) or {}
            return {
                role: (
                    int((summary.get(role) or {}).get("discovered", -1)),
                    int((summary.get(role) or {}).get("alive", -1)),
                )
                for role in ("PREFILL", "DECODE")
            }

        def sampler() -> None:
            t0 = time.monotonic()
            while not stop.is_set():
                s = sample()
                if s is not None:
                    samples.append((round(time.monotonic() - t0, 1), s))
                stop.wait(0.5)

        poller = threading.Thread(target=sampler, name="coldstart-sampler", daemon=True)
        poller.start()

        # Burst: 20 requests (10-way concurrent) immediately after ready.
        # The prefill address of every request is kept for the load-balance
        # assertion (balance_uniform_serial P1 contract) below.
        def run(rid: int):
            addr, err = ops.run_one_request(
                rid,
                output_len=2,
                block_keys=[rid * 100 + 1],
                stream_timeout_s=STREAM_TIMEOUT_S,
            )
            return addr, err

        rids = [ops.next_request_id(base) for _ in range(20)]
        with ThreadPoolExecutor(max_workers=10) as pool:
            results = list(pool.map(run, rids))
        ok = sum(1 for _, e in results if e is None)
        error_types = sorted({str(e)[:60] for _, e in results if e is not None})

        # Keep sampling 10s past the burst: transient 3-strike marks recover,
        # permanent ones stay dead (that is the S10-class regression).
        time.sleep(10.0)
        stop.set()
        poller.join(timeout=2.0)

        final = sample()
        dead_samples = sum(1 for _, s in samples if any(a < d for d, a in s.values()))
        final_ok = bool(
            final
            and all(d == expected[role] and a == d for role, (d, a) in final.items())
        )

        # Load-balance contract (user-mandated): under the cold-start burst
        # traffic must still spread across the engines.  Same calibration
        # as the balance suite (balance_uniform_serial / P1, with
        # the balance_concurrent_mix relaxed-caliber note): 20 requests over
        # 2 prefills (10-way concurrent), both engines used, no engine above
        # 80% of the *successful* requests — COST_BASED_PREFILL scores the
        # two prefills identically on an empty cold ledger and
        # RANDOM_WITHIN_TOLERANCE samples the tie window uniformly, so a
        # one-sided distribution can only come from an engine being
        # 3-strike-marked dead (the intake defect this probe guards).
        # 80% of 20 = 16 requests, i.e. the same "no engine eats the burst"
        # bound as the balance suite's P1 (loose floor 0.85 over the
        # uniform-random calibration; this probe keeps the historical 0.80
        # as its hard bound — semantics unchanged by the rework).
        addr_map = ops.addr_to_name()
        dist = Counter(addr_map.get(a, a) for a, e in results if e is None and a)
        n_ok = sum(dist.values())
        workers_used = len(dist)
        max_share = (max(dist.values()) / n_ok) if n_ok else 1.0
        balance_ok = workers_used >= 2 and max_share <= 0.80

        success_rate = ok / 20 * 100.0
        passed = (
            ok >= 16  # >=80% success + no permanent eviction
            and final_ok
            and balance_ok
        )
        return passed, (
            f"burst_ok={ok}/20 ({success_rate:.0f}%), "
            f"dead_samples={dead_samples}/{len(samples)}, final={final}, "
            f"error_types={error_types[:3]}, "
            f"balance: workers={workers_used}/{env.spec.n_prefill}, "
            f"max_share={max_share:.0%} (need >=2 workers and <=80%), "
            f"dist={json.dumps(dict(sorted(dist.items())))}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
