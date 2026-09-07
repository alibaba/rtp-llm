from __future__ import annotations

import json
import time
from collections import Counter

from ...context import CaseContext, rid_base
from ...grade import GradeReport
from ...registry import case
from ...support.kv import KV_CACHE_SYNC_WAIT_S, STREAM_TIMEOUT_S


@case(
    "kv_match_mixed",
    category="kv",
    source="hit-rate tier contrast M3",
)
def kv_match_mixed(ctx: CaseContext):
    """Prefix hit-rate tiers: full-hit and half-hit traffic concentrate on
    the holder while zero-hit traffic spreads — a graded contrast.

    Result properties: M3 soft contrast bound (graded lower band on the
    same-engine concentration of the full-hit and half-hit tiers), P2
    zero-hit multi-engine spread, P6 completeness.  Hit-latency benefits
    are NOT asserted (mock execution time is length/cache-blind).

    Construction (fixed input_len=8192, three tiers, all serial):
      * full-hit tier — seed family keys 4001-4008 on engine X1, then 10
        continuations reusing the SAME 8 blocks: hitTokens = 7168 (the last
        partial block is excluded: rawHit >= seqLen -> seqLen - blockSize),
        estimate discount ~5.0s vs tie window ~0.3s -> deterministic
        concentration on X1;
      * half-hit tier — seed keys 5001-5004 (input_len=4096, 4 blocks) on
        X2, then 10 requests carrying [5001-5004 + 4 fresh keys]: the
        continuous prefix match stops at 4 blocks -> hitTokens = 4096,
        discount ~2.9s vs tie window ~0.5s -> deterministic concentration
        on X2 (a 50% hit rate still clears the affinity threshold — the
        contrast with the zero-hit tier is the point, not a partial
        stickiness);
      * zero-hit tier — 10 requests with fresh unique keys on both
        engines: no discount anywhere -> uniform tie-window spread.

    Why P2 covers only the zero-hit tier: P2 forbids starving an engine
    with INDISTINGUISHABLE traffic; full/half-hit requests landing on
    their holder is correct affinity routing, not starvation.  The
    zero-hit tier is exactly the indistinguishable population, so its
    spread carries the P2 contract (probability of a single-engine
    collapse under correct spread: 2 * 0.5**10 ~= 0.2%).
    """
    ops = ctx.ops()
    report = GradeReport(run_grade=ctx.grade)
    base = rid_base(ctx, "kv")
    full_keys = list(range(4001, 4009))
    half_shared_keys = list(range(5001, 5005))
    try:

        def run_tier_cont(n: int, keys_fn, label: str):
            """Serial run of *n* requests, each keys from keys_fn(rid, i)."""
            addrs, failures = [], []
            for i in range(n):
                rid = ops.next_request_id(base)
                addr, err = ops.run_one_request(
                    rid,
                    input_len=8192,
                    output_len=2,
                    block_keys=keys_fn(rid, i),
                    stream_timeout_s=STREAM_TIMEOUT_S,
                )
                if err:
                    failures.append(f"{label} rid={rid}: {err}")
                else:
                    addrs.append(addr)
            return addrs, failures

        # -- tier 1: full-hit (8-block family).
        rid_seed1 = ops.next_request_id(base)
        seed1_addr, seed1_err = ops.run_one_request(
            rid_seed1,
            input_len=8192,
            output_len=2,
            block_keys=full_keys,
            stream_timeout_s=STREAM_TIMEOUT_S,
        )
        if seed1_err:
            report.invariant("P6", False, detail=f"seed1 failed: {seed1_err}")
            return report.finish(f"full-hit seed failed: {seed1_err}")
        time.sleep(KV_CACHE_SYNC_WAIT_S)
        full_addrs, full_fail = run_tier_cont(10, lambda rid, i: full_keys, "full")

        # -- tier 2: half-hit (4 shared + 4 fresh per request).
        rid_seed2 = ops.next_request_id(base)
        seed2_addr, seed2_err = ops.run_one_request(
            rid_seed2,
            input_len=4096,
            output_len=2,
            block_keys=half_shared_keys,
            stream_timeout_s=STREAM_TIMEOUT_S,
        )
        if seed2_err:
            report.invariant("P6", False, detail=f"seed2 failed: {seed2_err}")
            return report.finish(f"half-hit seed failed: {seed2_err}")
        time.sleep(KV_CACHE_SYNC_WAIT_S)
        half_addrs, half_fail = run_tier_cont(
            10,
            lambda rid, i: half_shared_keys + [rid * 100 + 40 + j for j in range(4)],
            "half",
        )

        # -- tier 3: zero-hit (fresh unique keys everywhere).
        zero_addrs, zero_fail = run_tier_cont(
            10, lambda rid, i: [rid * 100 + j for j in range(8)], "zero"
        )

        addr_map = ops.addr_to_name()
        failures = full_fail + half_fail + zero_fail

        def concentration(addrs, anchor_addr) -> float:
            if not addrs:
                return 0.0
            anchor = addr_map.get(anchor_addr, anchor_addr)
            return sum(1 for a in addrs if addr_map.get(a, a) == anchor) / len(addrs)

        full_conc = concentration(full_addrs, seed1_addr)
        half_conc = concentration(half_addrs, seed2_addr)
        zero_dist = Counter(addr_map.get(a, a) for a in zero_addrs)
        zero_engines = len(zero_dist)
        zero_max = max(zero_dist.values()) / len(zero_addrs) if zero_addrs else 1.0

        report.invariant(
            "P6",
            not failures
            and len(full_addrs) == 10
            and len(half_addrs) == 10
            and len(zero_addrs) == 10,
            detail=f"failures={failures[:2]}",
        )
        report.check(
            "M3",
            full_conc,
            context="full_hit",
            detail=(
                f"concentration on full-hit seed engine={full_conc:.2f} "
                f"(vs zero-hit baseline ~0.5)"
            ),
        )
        report.check(
            "M3",
            half_conc,
            context="half_hit",
            detail=(
                f"concentration on half-hit seed engine={half_conc:.2f} "
                f"(50% hit still clears the affinity threshold)"
            ),
        )
        report.invariant(
            "P2",
            zero_engines >= 2,
            context="zero_hit",
            detail=(
                f"zero-hit spread: engines={zero_engines}, "
                f"max_share={zero_max:.2f} (observational, expected ~0.5-0.7)"
            ),
        )
        return report.finish(
            f"full_conc={full_conc:.2f}, half_conc={half_conc:.2f}, "
            f"zero_dist={json.dumps(dict(zero_dist), sort_keys=True)}, "
            f"grades: {report.summary()}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
